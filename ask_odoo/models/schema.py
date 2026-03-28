from odoo import models, fields, api
import logging
import base64
from langchain_core.documents import Document
from langchain_text_splitters import CharacterTextSplitter

_logger = logging.getLogger(__name__)

class AskOdooModel(models.Model):
    _inherit = 'ask.odoo.model'

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
        
        # 1 doc for the entire schema only
        schema_doc = Document(page_content=clean_corpus, metadata={'source': 'schema_snapshot'})
        
        # Chunking based on ==================================================
        text_splitter = CharacterTextSplitter(
            separator=separator,
            chunk_size=1,
            chunk_overlap=0,
            keep_separator=False
        )
        documents = text_splitter.split_documents([schema_doc])
        
        # 5. Add to Vector Store
        # 5. Add to Vector Store
        if documents:
            v_store = self._get_schema_vector_store()
            
            # Clear existing collection to prevent duplicates
            try:
                v_store.delete_collection()
                _logger.info("AskOdoo: Cleared existing schema collection.")
                
                # IMPORTANT: The collection is now gone from the DB.
                # We MUST re-initialize the PGVector store so it recreates the collection.
                # If we use the old 'v_store' object, it errors with "Collection not found".
                type(self)._schema_vector_store = None
                v_store = self._get_schema_vector_store()
                
            except Exception as e:
                # If collection didn't exist, that's fine.
                # But we should ensure we have a valid store for adding.
                _logger.warning(f"AskOdoo: Warning during collection cleanup: {e}")
                # Optional: Force re-init if unsure, but usually only needed on success
                if "not found" in str(e).lower():
                     type(self)._schema_vector_store = None
                     v_store = self._get_schema_vector_store()
            
            v_store.add_documents(documents)
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
                    except:
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

