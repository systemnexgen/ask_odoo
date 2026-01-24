from odoo import models, fields, api
from odoo.tools import config
import logging
import json
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
    _embeddings = None


    # ==========================
    # PUBLIC ENTRY POINT (RPC)
    # ==========================
    @api.model
    def process_message(self, message, conversation_id=None, chat_mode='conversation'):
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
        # API_KEY = "AIzaSyDY2sLovBB-Mw9vkg0FoIE9yM8xXTgTGm4" 
        # print(f"Google API Key: {API_KEY}")
        # return ChatGoogleGenerativeAI(
        #     model="gemma-3-4b-it",
        #     google_api_key=API_KEY,
        #     temperature=0.1
        # )

        # # Ollama Implementation (Commented out)
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

