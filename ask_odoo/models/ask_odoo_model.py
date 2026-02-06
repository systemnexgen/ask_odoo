from odoo import models, fields, api
from odoo.tools import config
from odoo.tools.safe_eval import safe_eval
import re
import logging
import json
import base64
import os
from dotenv import load_dotenv
load_dotenv()

from langchain_postgres.vectorstores import PGVector
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_community.chat_models import ChatOllama
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

_logger = logging.getLogger(__name__)

class AskOdooModel(models.Model):
    _name = 'ask.odoo.model'
    _description = 'Ask Odoo Model'

    name = fields.Char(required=True)
    description = fields.Text()
    user_id = fields.Many2one('res.users', default=lambda self: self.env.user)
    active = fields.Boolean(default=True)

    gemini_api_key = fields.Char(required=False)

    # Embedding Model Cache
    _vector_store = None
    _schema_vector_store = None
    _embeddings = None


    # ==========================
    # PUBLIC ENTRY POINT (RPC)
    # ==========================
    @api.model
    def process_message(self, message, conversation_id=None, chat_mode='conversation'):
        if chat_mode == 'conversation':
            return self.process_message_rag(message, conversation_id, chat_mode)
        else:
            return self.process_message_db(message, conversation_id)

    def process_message_db(self, message, conversation_id=None):
        """
        Handles database / schema related queries using Schema RAG.
        This does NOT execute CRUD yet — only explains models, fields, relations.
        """
        chat_mode = "database"
        _logger.info("AskOdoo: Processing message in DATABASE mode")

        # 1. Handle Conversation
        if conversation_id:
            conversation = self.env['ask.odoo.conversation'].browse(conversation_id)
        else:
            conversation = self.env['ask.odoo.conversation'].create({
                'name': 'New DB Chat',
                'user_id': self.env.user.id
            })

        # 2. Save User Message
        user_msg = self.env['ask.odoo.message'].create({
            'conversation_id': conversation.id,
            'type': 'user',
            'content': message,
        })

        try:
            # 3. Components
            llm = self._get_llm()
            retriever = self._get_schema_retriever()

            history = self._get_history(conversation.id, exclude_id=user_msg.id)

            # 4. Retrieve Schema Context
            docs = retriever.invoke(message)

            context_text = (
                "\n\n".join(doc.page_content for doc in docs)
                if docs else "No relevant schema found."
            )

            _logger.info(
                f"\n=== [AskOdoo][DB MODE] Schema Context ===\n{context_text}\n========================================="
            )

            # 5. Database-Specific Prompt
            system_template = (
                "You are an Odoo Database Assistant capable of performing CRUD operations.\n\n"
                "CRITICAL RULES:\n"
                "- Use ONLY the schema information provided below.\n"
                "- To READ data: Generate Odoo Python code using `self.env`.\n"
                "- To CREATE/UPDATE/DELETE: Generate Odoo Python code using `self.env`.\n"
                "- WRAP ALL PYTHON CODE in a code block like:\n"
                "```python\n"
                "# Your code here\n"
                "records = self.env['...'].search([...])\n"
                "result = records.mapped('name') # Assign final output to result\n"
                "```\n"
                "- DO NOT use `print()`. It is not available. The value of `result` will be returned to the user.\n"
                "- If reading data, assign the output to a variable named `result`.\n"
                "- If the user asks for specific values (e.g. 'names'), ensure your code extracts them (e.g. `.mapped('name')`).\n"
                "- REMEMBER: Every Odoo model AUTOMATICALLY has these fields: `id`, `create_date`, `create_uid`, `write_date`, `write_uid`. You can ALWAYS sort by `create_date`.\n"
                "- Explain your action briefly before the code block.\n"
                "- If something is not in the schema, say you don't know.\n\n"
                "DATABASE SCHEMA DOCUMENTS:\n{context}"
            )

            prompt = ChatPromptTemplate.from_messages([
                ("system", system_template),
                MessagesPlaceholder(variable_name="history"),
                ("human", "{question}"),
            ])

            prompt_messages = prompt.format_messages(
                context=context_text,
                history=history,
                question=message
            )

            _logger.info(
                "\n=== [AskOdoo][DB MODE] Full Prompt ===\n" +
                "\n".join(f"[{m.type.upper()}]: {m.content}" for m in prompt_messages) +
                "\n====================================="
            )

            # 6. Invoke LLM
            ai_message = llm.invoke(prompt_messages)
            response_text = ai_message.content

            _logger.info(
                f"\n=== [AskOdoo][DB MODE] LLM Response ===\n{response_text}\n====================================="
            )

        except Exception as e:
            _logger.exception("Database Mode Failed")
            response_text = f"Error: {str(e)}"

        # 7. Save AI Message
        self.env['ask.odoo.message'].create({
            'conversation_id': conversation.id,
            'type': 'ai',
            'content': response_text,
        })

        # 8. Update Conversation
        conversation.write({'last_activity': fields.Datetime.now()})

        if len(conversation.message_ids) <= 2:
            conversation.write({
                'name': message[:30] + "..." if len(message) > 30 else message
            })

        # 9. Extract Code if present
        action_code = None
        code_match = re.search(r"```python\n(.*?)\n```", response_text, re.DOTALL)
        if code_match:
            action_code = code_match.group(1).strip()

        return {
            'response': response_text,
            'conversation_id': conversation.id,
            'title': conversation.name,
            'action_code': action_code,
        }

    @api.model
    def execute_confirmed_code(self, code, conversation_id):
        """
        Executes the confirmed Python code safely.
        """
        try:
            local_context = {
                'self': self,
                'env': self.env,
                'result': None,
                'time': fields.Datetime.now,
                'datetime': fields.Datetime,
                'date': fields.Date,
            }
            # We use safe_eval but with full ORM access for the assistant
            safe_eval(code, local_context, mode="exec", nocopy=True)
            
            result = local_context.get('result')
            
            
            # Post-processing: If result is a RecordSet, make it readable
            if result is not None and hasattr(result, '_name') and hasattr(result, 'mapped'):
                # It's an Odoo RecordSet
                if not result:
                    result = "No records found."
                else:
                    try:
                        # Prefer display_name or name
                        result = result.mapped('display_name')
                    except:
                        result = str(result)
            
            # Convert dict_keys, dict_values, sets, and other iterables to lists for better formatting
            if result is not None:
                # Check for dict_keys, dict_values, or similar view objects
                result_type = type(result).__name__
                if result_type in ('dict_keys', 'dict_values', 'dict_items', 'set', 'frozenset'):
                    result = list(result)
                # Also handle generators and other iterables (but not strings!)
                elif hasattr(result, '__iter__') and not isinstance(result, (str, bytes)):
                    # Check if it's not already a list or tuple
                    if not isinstance(result, (list, tuple)):
                        try:
                            result = list(result)
                        except:
                            pass  # Keep as-is if conversion fails

            return {
                'status': 'success',
                'result': str(result) if result is not None else "Action Executed Successfully"
            }
        except Exception as e:
            return {
                'status': 'error',
                'message': str(e)
            }


    def process_message_rag(self, message, conversation_id=None, chat_mode='conversation'):
        """
        Main orchestrator using LangChain.
        """
        _logger.info(f"AskOdoo: Processing message in mode: {chat_mode}")

        # 1. Handle Conversation
        if conversation_id:
            conversation = self.env['ask.odoo.conversation'].browse(conversation_id)
        else:
            conversation = self.env['ask.odoo.conversation'].create({
                'name': 'New Chat',  # Will serve as temporary title
                'user_id': self.env.user.id
            })

        # 2. Save User Message
        user_msg = self.env['ask.odoo.message'].create({
            'conversation_id': conversation.id,
            'type': 'user',
            'content': message,
        })

        # 3. Build & Run Chain
        try:
            # Prepare Components
            llm = self._get_llm()
            retriever = self._get_retriever()
            
            # Prepare History (excluding the message we just added to avoid duplication in prompt)
            history = self._get_history(conversation.id, exclude_id=user_msg.id)
            
            # 1. Retrieve Context explicitly for logging
            docs = retriever.invoke(message)
            def format_docs(docs):
                return "\n\n".join(doc.page_content for doc in docs) if docs else "No documents found."
            context_text = format_docs(docs)
            
            _logger.info(f"\n=== [AskOdoo] DEBUG: Context Retrieved ===\n{context_text}\n===========================================")

            # 2. Prepare Prompt
            system_template = (
                f"You are a helpful Odoo AI Assistant running in {chat_mode.upper()} mode.\n\n"

                "CRITICAL INSTRUCTIONS:\n"
                "- Answer ONLY the current question. Do NOT repeat previous answers.\n"
                "- Make your answers short, to the point and accurate.\n"
                "- Use the knowledge base documents below to answer the question.\n"
                "- The conversation history is ONLY for understanding context - do NOT copy or repeat previous responses.\n"
                "- Each question deserves a fresh, direct answer based on the knowledge base.\n"
                "- Do NOT use assumptions.\n\n"

                "KNOWLEDGE BASE DOCUMENTS:\n{context}"
            )
            
            prompt_template = ChatPromptTemplate.from_messages([
                ("system", system_template),
                MessagesPlaceholder(variable_name="history"),
                ("human", "{question}"),
            ])
            
            # 3. Format complete prompt messages to see exactly what is sent to LLM
            prompt_messages = prompt_template.format_messages(
                context=context_text,
                history=history,
                question=message
            )
            
            # Log the full prompt payload (as text representation)
            prompt_debug_str = "\n".join([f"[{m.type.upper()}]: {m.content}" for m in prompt_messages])
            _logger.info(f"\n=== [AskOdoo] DEBUG: Full Prompt Payload ===\n{prompt_debug_str}\n============================================")

            # 4. Invoke LLM directly
            ai_message = llm.invoke(prompt_messages)
            response_text = ai_message.content
            
            _logger.info(f"\n=== [AskOdoo] DEBUG: LLM Response ===\n{response_text}\n=====================================")


        except Exception as e:
            _logger.exception("LangChain Execution Failed")
            response_text = f"Error: {str(e)}"

        # 4. Save AI Response
        self.env['ask.odoo.message'].create({
            'conversation_id': conversation.id,
            'type': 'ai',
            'content': response_text,
        })

        # 5. Update Last Activity & Title (if new)
        conversation.write({'last_activity': fields.Datetime.now()})

        if len(conversation.message_ids) <= 2:
            # Generate a short title based on the first user message
            title = message[:30] + "..." if len(message) > 30 else message
            conversation.write({'name': title})

        return {
            'response': response_text,
            'conversation_id': conversation.id,
            'title': conversation.name
        }

    @api.model
    def get_sidebar_conversations(self):
        """Fetch chats for the sidebar."""
        conversations = self.env['ask.odoo.conversation'].search(
            [('user_id', '=', self.env.user.id)],
            order='last_activity desc',
            limit=50
        )
        return [{'id': c.id, 'title': c.name} for c in conversations]

    @api.model
    def get_messages(self, conversation_id):
        """Fetch full history for a chat."""
        messages = self.env['ask.odoo.message'].search(
            [('conversation_id', '=', conversation_id)],
            order='create_date asc'
        )
        return [{
            'id': m.id,
            'text': m.content,
            'type': m.type
        } for m in messages]

    # ==========================
    # HELPERS
    # ==========================
    
    def _get_connection_string(self):
        db_name = self.env.cr.dbname
        user = config.get('db_user')
        password = config.get('db_password')
        host = config.get('db_host') or 'localhost'
        port = config.get('db_port') or 5432
        return f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{db_name}"

    def _get_retriever(self):
        if not AskOdooModel._vector_store:
            # Initialize Embeddings
            if not AskOdooModel._embeddings:
                # Use HuggingFaceEmbeddings as replacement for locally managed SentenceTransformer
                # Ensure 'sentence_transformers' is installed or 'langchain_huggingface'
                AskOdooModel._embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
            
            # Initialize PGVector
            # Note: PGVector expects specific extension and tables. 
            # We use the standard LangChain implementation.
            connection = self._get_connection_string()
            AskOdooModel._vector_store = PGVector(
                embeddings=AskOdooModel._embeddings,
                collection_name="ask_odoo_knowledge_chunk",
                connection=connection,
                use_jsonb=True,
            )
        
        return AskOdooModel._vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 3}
        )

    def _get_llm(self):
        # Groq Implementation
        GROQ_API_KEY = os.getenv("GROQ_API_KEY")
        print(f"Groq API Key: {GROQ_API_KEY}")
        return ChatGroq(
            model="llama-3.1-8b-instant",
            groq_api_key=GROQ_API_KEY,
            temperature=0.1
        )
        
        # # Google Gemini
        # print(f"Google API Key: {API_KEY}")
        # return ChatGoogleGenerativeAI(
        #     model="gemma-3-4b-it",
        #     google_api_key=API_KEY,
        #     temperature=0.1
        # )

        # # Ollama Implementation
        # return ChatOllama(
        #     model="llama3.2:latest",
        #     temperature=0,
        #     base_url="http://localhost:11434"
        # )

    def _get_history(self, conversation_id, exclude_id=None):
        """Fetch history and convert to LangChain Messages."""
        domain = [('conversation_id', '=', conversation_id)]
        if exclude_id:
            domain.append(('id', '!=', exclude_id))
            
        # Fetch latest N messages
        messages = self.env['ask.odoo.message'].search(
            domain,
            order='id desc',
            limit=10
        )
        
        # Sort by ID ascending explicitly to ensure chronological order [Oldest -> Newest]
        history_records = messages.sorted(lambda m: m.id)
        
        history = []
        for msg in history_records:
            if msg.type == 'user':
                history.append(HumanMessage(content=msg.content or ""))
            else:
                history.append(AIMessage(content=msg.content or ""))
                
        _logger.info(f"Retrieved {len(history)} messages for history. IDs: {history_records.ids}")
        return history

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

            # Build metadata dictionary
            model_metadata = {
                'model': model_name,
                'name': record.name,          # Human-readable description
                'module': record.modules,     # Comma-separated list of modules defining this model
                'fields': model_fields
            }
            
            schema_data.append(model_metadata)

        return schema_data

    def _schema_to_text(self, model_metadata):
        """
        Converts a model's schema dictionary into a natural language text summary
        suitable for embedding vectors.
        Output format:
        "Model: [Name] ([Technical Name])
         Module: [Module]
         Description: [Description]
         Key Fields:
         - [Field String] ([Field Name]): [Type] related to [Relation]"
        """
        # 1. Extract Basic Info
        model_tech_name = model_metadata.get('model', 'Unknown')
        model_name = model_metadata.get('name', model_tech_name)
        module_name = model_metadata.get('module', 'Unknown')
        
        # 2. Filter & Prioritize Fields
        # We want to focus on business logic fields, skipping standard audit fields for the summary
        ignored_fields = {'id', 'create_uid', 'create_date', 'write_uid', 'write_date', '__last_update'}
        
        all_fields = model_metadata.get('fields', [])
        
        # Separate interesting fields from the rest
        business_fields = [f for f in all_fields if f['name'] not in ignored_fields]
        
        # Sort heuristics: 
        # 1. 'name' field is usually most important
        # 2. 'state' or 'selection' fields (process status)
        # 3. Relational fields (connections to other data)
        # 4. Others
        def field_importance(f):
            name = f['name']
            ftype = f['type']
            
            if name == 'name': return 0
            if name == 'state': return 1
            if ftype == 'selection': return 2
            if ftype in ['many2one', 'one2many', 'many2many']: return 3
            return 4

        business_fields.sort(key=field_importance)
        
        # 3. Select Top 30
        top_fields = business_fields[:30]
        
        if not top_fields:
            return None

        
        # 4. Generate Field Strings
        field_lines = []
        for f in top_fields:
            line = f"- {f['string']} ({f['name']}): {f['type']}"
            if f.get('relation'):
                line += f" -> {f['relation']}"
            field_lines.append(line)
            
        # 5. Construct Final Text
        # Note: We repeat the Model Name slightly to ensure the embedding captures it strongly
        text = (
            f"Odoo Model: {model_name}\n"
            f"Technical Name: {model_tech_name}\n"
            f"Module: {module_name}\n"
            f"Key Fields:\n" + 
            "\n".join(field_lines)
        )
        
        return text

    # ==========================
    # SCHEMA VECTOR STORE
    # ==========================

    def _get_schema_vector_store(self):
        """
        Returns the PGVector store specifically for Database Schema.
        Uses a separate collection 'ask_odoo_schema'.
        """
        if not AskOdooModel._schema_vector_store:
            # Ensure Embeddings are ready
            if not AskOdooModel._embeddings:
                AskOdooModel._embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

            connection = self._get_connection_string()
            
            AskOdooModel._schema_vector_store = PGVector(
                embeddings=AskOdooModel._embeddings,
                collection_name="ask_odoo_schema",
                connection=connection,
                use_jsonb=True,
            )
        return AskOdooModel._schema_vector_store

    def _get_schema_retriever(self):
        """
        Returns a retriever for the schema vector store.
        """
        vector_store = self._get_schema_vector_store()
        return vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 10} # Retrieve top 10 to ensure we get related models
        )

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
        # We use a text splitter on the full corpus
        # Increased chunk size to 4000 to try and keep entire model definitions (like res.partner) intact
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=4000, 
            chunk_overlap=200,
            # Prioritize keeping models together usually, but follow size limits
            separators=["\n\nOdoo Model:", "\n\n", "\n", " "] 
        )
        
        texts = text_splitter.split_text(full_corpus)
        
        # Create Documents (metadata is lost in this massive chunking approach, 
        # but content contains the model info)
        documents = [Document(page_content=t, metadata={'source': 'schema_snapshot'}) for t in texts]
            
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
                AskOdooModel._schema_vector_store = None
                v_store = self._get_schema_vector_store()
                
            except Exception as e:
                # If collection didn't exist, that's fine.
                # But we should ensure we have a valid store for adding.
                _logger.warning(f"AskOdoo: Warning during collection cleanup: {e}")
                # Optional: Force re-init if unsure, but usually only needed on success
                if "not found" in str(e).lower():
                     AskOdooModel._schema_vector_store = None
                     v_store = self._get_schema_vector_store()
            
            v_store.add_documents(documents)
            _logger.info(f"AskOdoo: Indexed {len(documents)} schema chunks from snapshot.")
            
        return True

