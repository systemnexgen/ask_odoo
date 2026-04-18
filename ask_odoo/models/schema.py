from odoo import models, fields, api
import logging
import base64
import re
import numpy as np
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate

_logger = logging.getLogger(__name__)

class AskOdooModel(models.Model):
    _inherit = 'ask.odoo.model'

    @api.model
    def _vector_shortlist_fields(self, question, model_metadata, embeddings_model, top_n=15):
        """
        Pure Vector Field Shortlisting:
        Embeds the question and every field (description + name + type) to
        dynamically pick the top N most semantically relevant fields in-memory.
        Mathmatically limits vertical scope to avoid exceeding token window limits.
        """
        model_fields = model_metadata.get('fields', [])
        if not model_fields:
            return model_metadata

        # Ensure 'id' is always kept since it's the primary key and often needed
        # We manually reserve it, so we only need to score the rest for top_N - 1
        id_field = next((f for f in model_fields if f.get('name') == 'id'), None)
        other_fields = [f for f in model_fields if f.get('name') != 'id']

        if not other_fields:
            return model_metadata

        # 1. Embed query
        query_emb = embeddings_model.embed_query(question)

        # 2. Embed all field strings
        field_strings = [
            f"{f.get('string', '')} ({f.get('name', '')}): {f.get('type', '')}"
            for f in other_fields
        ]
        
        # Batch embed is much faster
        try:
            field_embs = embeddings_model.embed_documents(field_strings)
        except Exception as e:
            _logger.error(f"AskOdoo: Failed to embed fields for {model_metadata.get('model')}: {e}")
            return model_metadata
            
        # 3. Vectorised cosine similarity via numpy
        query_vec = np.array(query_emb)
        field_mat = np.array(field_embs)

        # Normalise to unit vectors (avoid division by zero)
        query_norm = query_vec / (np.linalg.norm(query_vec) + 1e-10)
        field_norms = field_mat / (np.linalg.norm(field_mat, axis=1, keepdims=True) + 1e-10)
        scores = field_norms @ query_norm  # shape: (n_fields,)

        # 4. Pick top-N field indices
        allocated_slots = top_n - (1 if id_field else 0)
        top_indices = np.argsort(scores)[::-1][:allocated_slots]
        top_fields = [other_fields[i] for i in top_indices]
        
        if id_field:
            top_fields.insert(0, id_field)

        return {**model_metadata, 'fields': top_fields}

    @api.model
    def _get_relevant_schema(self, question):
        """
        Two-phase schema retrieval.

        Phase 1: Vector similarity search retrieves the top-K most relevant
                 *model* documents from the schema vector store.

        Phase 2: For each retrieved model, apply _trim_fields_for_question()
                 to remove fields that are irrelevant to the question. Relational
                 fields and generic business fields are always preserved.

        Returns a single context string (trimmed schema text) ready to inject
        into the LLM prompt.
        """
        # ── Phase 1: Retrieve top-K relevant models ───────────────────────────
        retriever = self._get_schema_retriever()
        docs = retriever.invoke(question)

        if not docs:
            _logger.warning("AskOdoo: Phase 1 retrieval returned no schema docs.")
            return "No relevant schema found."

        # Extract the technical model names from the retrieved doc text.
        # Each doc's first line is "Model: <human name> (<tech.name>)"
        retrieved_model_names = []
        for doc in docs:
            first_line = doc.page_content.split('\n')[0]  # e.g. "Model: Sales Order (sale.order)"
            match = re.search(r'\(([a-z][a-z0-9_.]+)\)', first_line)
            if match:
                retrieved_model_names.append(match.group(1))

        _logger.info(
            f"AskOdoo: [Phase 1] Retrieved {len(docs)} models: {retrieved_model_names}"
        )

        if not retrieved_model_names:
            # Fallback: return raw doc text if we can't parse model names
            _logger.warning("AskOdoo: Could not parse model names from docs. Using raw text.")
            return "\n\n".join(doc.page_content for doc in docs)

        # ── Phase 2: Trim fields for each retrieved model ─────────────────────
        context_parts = []
        for model_name in retrieved_model_names:
            if model_name not in self.env:
                _logger.warning(f"AskOdoo: Model '{model_name}' not in registry, skipping.")
                continue

            current_model = self.env[model_name]
            if current_model._abstract or current_model._transient:
                continue

            # Build the structured metadata dict for this model on the fly
            # (same structure as _get_db_schema produces)
            model_record = self.env['ir.model'].search(
                [('model', '=', model_name)], limit=1
            )
            model_fields = []
            for field_name, field_obj in current_model._fields.items():
                model_fields.append({
                    'name': field_name,
                    'type': field_obj.type,
                    'string': field_obj.string,
                    'relation': getattr(field_obj, 'comodel_name', None),
                })

            model_metadata = {
                'model': model_name,
                'name': model_record.name if model_record else model_name,
                'fields': model_fields,
            }

            # Apply Phase 2 trimming (Strict Vector Top-15)
            original_count = len(model_fields)
            embeddings = self._get_embeddings()
            trimmed_metadata = self._vector_shortlist_fields(question, model_metadata, embeddings, top_n=15)
            trimmed_count = len(trimmed_metadata['fields'])

            _logger.info(
                f"AskOdoo: [Phase 2] {model_name}: "
                f"{original_count} fields → {trimmed_count} fields kept"
            )

            context_parts.append(self._schema_to_text(trimmed_metadata))

        if not context_parts:
            return "No relevant schema found."
                
        return "\n\n".join(context_parts)

    @api.model
    def refresh_schema_index(self):
        """
        Re-indexes the entire Odoo database schema into the vector store.
        Generates a single canonical text corpus (snapshot), saves it as an attachment,
        then chunks and embeds from that corpus.
        """
        _logger.info("AskOdoo: Starting Schema Indexing...")
        
        # 1. Get Metadata
        schemas = self._get_db_schema()
        _logger.info(f"AskOdoo: Found {len(schemas)} models.")
        
        # 2. Generate Canonical Text Corpus
        # We join all model descriptions into one large text
        corpus_parts = []
        for s in schemas:
            text_content = self._schema_to_text(s)
            if text_content:
                corpus_parts.append(text_content)
            
        # Join with a separator that helps visualization handling
        full_corpus = ("\n\n" + "="*50 + "\n\n").join(corpus_parts)
        full_corpus += "\n\n" + "="*50 + "\n=== END OF SCHEMA SNAPSHOT ===\n"
        
        # 3. Save as Snapshot (ir.attachment) for Audit/Debug
        # This serves as the 'source of truth' for the embedding
        try:
            # Cleanup old snapshots to avoid clutter
            self.env['ir.attachment'].search([
                ('description', '=', 'Canonical schema snapshot for AskOdoo RAG')
            ]).unlink()
            _logger.info("AskOdoo: Removed old schema snapshots.")

            attachment_name = f"odoo_schema_snapshot_{fields.Datetime.now().isoformat().replace(':','-')}.txt"
            b64_data = base64.b64encode(full_corpus.encode('utf-8'))
            
            self.env['ir.attachment'].create({
                'name': attachment_name,
                'type': 'binary',
                'datas': b64_data,
                'mimetype': 'text/plain',
                'description': 'Canonical schema snapshot for AskOdoo RAG',
            })
            _logger.info(f"AskOdoo: Saved schema snapshot: {attachment_name}")
        except Exception as e:
            _logger.warning(f"AskOdoo: Failed to save schema snapshot attachment: {e}")

        # 4. Chunk and Embed FROM THE CORPUS
        # Split strictly by the 50-equals separator so that each chunk corresponds to exactly ONE model.
        separator = "\n\n" + "="*50 + "\n\n"
        
        # Remove the EOF marker before splitting to keep the list clean
        eof_marker = "\n\n" + "="*50 + "\n=== END OF SCHEMA SNAPSHOT ===\n"
        clean_corpus = full_corpus.replace(eof_marker, "")
        
        # Split directly using python to bypass CharacterTextSplitter chunk_size warnings
        raw_chunks = clean_corpus.split(separator)
        documents = []
        for chunk in raw_chunks:
            chunk = chunk.strip()
            if chunk:
                documents.append(Document(page_content=chunk, metadata={'source': 'schema_snapshot'}))
                
        _logger.info(f"AskOdoo: Created {len(documents)} document chunks locally.")
        
        # 5. Add to Vector Store
        if documents:
            _logger.info("AskOdoo: Calling _get_schema_vector_store()...")
            v_store = self._get_schema_vector_store()
            _logger.info("AskOdoo: Finished _get_schema_vector_store(). Deleting collection...")
            
            # Clear existing collection to prevent duplicates
            try:
                v_store.delete_collection()
                _logger.info("AskOdoo: Cleared existing schema collection. Re-initializing PGVector...")
                
                # IMPORTANT: The collection is now gone from the DB.
                # We MUST re-initialize the PGVector store so it recreates the collection.
                import odoo.addons.ask_odoo.models.llm as _llm_mod
                _llm_mod._schema_vector_store_instance = None
                v_store = self._get_schema_vector_store()
                _logger.info("AskOdoo: PGVector re-initialized.")
                
            except Exception as e:
                _logger.warning(f"AskOdoo: Warning during collection cleanup: {e}")
                if "not found" in str(e).lower():
                    import odoo.addons.ask_odoo.models.llm as _llm_mod
                    _llm_mod._schema_vector_store_instance = None
                    v_store = self._get_schema_vector_store()
            
            _logger.info(f"AskOdoo: Starting add_documents with {len(documents)} chunks...")
            batch_size = 50
            for i in range(0, len(documents), batch_size):
                batch = documents[i:i+batch_size]
                _logger.info(f"AskOdoo: Indexing batch {i//batch_size + 1}/{(len(documents)-1)//batch_size + 1} ({len(batch)} chunks)...")
                v_store.add_documents(batch)
            _logger.info(f"AskOdoo: Indexed {len(documents)} schema chunks from snapshot.")
            
        return True

    def _get_db_schema(self):
        """
        Dynamically extracts schema metadata for all available models.
        Returns a list of dictionaries describing models and their fields.
        """
        schema_data = []

        # Iterate over all models registered in the database (ir.model)
        # This ensures we get descriptions and module info from the database records
        model_records = self.env['ir.model'].search([])

        for record in model_records:
            model_name = record.model
            
            # Ensure the model is currently accessible in the environment registry
            if model_name not in self.env:
                continue

            current_model = self.env[model_name]
            
            # Skip Abstract and Transient models (Schema RAG should focus on persistent data)
            # This filters out technical mixins like ir.websocket, base_import.mapping, etc.
            if current_model._abstract or current_model._transient:
                continue
            
            # Extract fields dynamically from the model class
            # This includes custom fields (x_) and fields added by Studio
            model_fields = []
            for field_name, field_obj in current_model._fields.items():
                field_data = {
                    'name': field_name,
                    'type': field_obj.type,
                    'string': field_obj.string,
                    'relation': getattr(field_obj, 'comodel_name', None),
                }
                model_fields.append(field_data)

            # Extract methods dynamically
            # We look for methods starting with 'action_' or 'button_' which usually denote business logic
            model_methods = []
            # Use dir() to get all attributes, but we need to be careful about what we access
            # We only check attributes that exist on the class/recordset
            for attr_name in dir(current_model):
                if attr_name.startswith(('action_', 'button_')) and not attr_name.startswith('_'):
                    try:
                        attr = getattr(current_model, attr_name)
                        if callable(attr):
                            # Get the first line of the docstring as description
                            doc = (attr.__doc__ or "").strip().split('\n')[0]
                            model_methods.append({
                                'name': attr_name,
                                'description': doc or "No description available."
                            })
                    except (AttributeError, TypeError, Exception) as e:
                        _logger.debug("Could not inspect method %s: %s", attr_name, e)
                        continue

            # Build metadata dictionary
            model_metadata = {
                'model': model_name,
                'name': record.name,          # Human-readable description
                'module': record.modules,     # Comma-separated list of modules defining this model
                'fields': model_fields,
                'methods': model_methods      # Include the extracted methods
            }
            
            schema_data.append(model_metadata)

        return schema_data

    def _schema_to_text(self, model_metadata):
        """
        Converts model schema to text with ALL fields included.
        No filters applied — every field is preserved.
        """
        # 1. Basic Info
        model_name = model_metadata.get('name', model_metadata.get('model'))
        model_tech_name = model_metadata.get('model')

        # 2. Include ALL Fields (no filtering)
        all_fields = []
        field_names = []
        for f in model_metadata.get('fields', []):
            # Formatting: "Name (tech_name): type"
            # If relation exists, add "-> RelatedModel"
            desc = f"- {f['string']} ({f['name']}): {f['type']}"
            if f.get('relation'):
                desc += f" -> {f['relation']}"
            all_fields.append(desc)
            field_names.append(f['name'])

        # 3. Construct Text
        text = f"Model: {model_name} ({model_tech_name})\n"
        text += f"All field names: {', '.join(field_names)}\n"
        text += f"Fields details:\n" + "\n".join(all_fields)

        return text

