from odoo import models, fields, api
from odoo.tools import config
import logging
import json

try:
    from langchain_postgres.vectorstores import PGVector
    from langchain_google_genai import ChatGoogleGenerativeAI
    from langchain_community.chat_models import ChatOllama
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
    from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.runnables import RunnablePassthrough
except ImportError:
    # Handle missing dependencies gracefully or log warning
    pass

_logger = logging.getLogger(__name__)

class AskOdooModel(models.Model):
    _name = 'ask.odoo.model'
    _description = 'Ask Odoo Model'

    name = fields.Char(required=True)
    description = fields.Text()
    user_id = fields.Many2one('res.users', default=lambda self: self.env.user)
    active = fields.Boolean(default=True)

    # Embedding Model Cache
    # Caches
    _vector_store = None
    _embeddings = None


    # ==========================
    # PUBLIC ENTRY POINT (RPC)
    # ==========================
    # ==========================
    # PUBLIC ENTRY POINT (RPC)
    # ==========================
    @api.model
    def process_message(self, message, conversation_id=None):
        """
        Main orchestrator using LangChain.
        """
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
            
            # Prepare Prompt
            system_template = (
                "You are a helpful Odoo AI Assistant. "
                "Answer the user's question using the provided KNOWLEDGE BASE DOCUMENTS and the CONVERSATION HISTORY. "
                "1. If the answer is in the documents, use that information explicitly.\n"
                "2. If the user refers to previous messages (e.g., 'my name', 'what did we discuss'), check the CONVERSATION HISTORY.\n"
                "3. If the answer is NOT in the documents or history, answer based on your general Odoo knowledge, but mention if you didn't find specific info in the uploaded files.\n\n"
                "KNOWLEDGE BASE DOCUMENTS:\n{context}"
            )
            
            prompt = ChatPromptTemplate.from_messages([
                ("system", system_template),
                MessagesPlaceholder(variable_name="history"),
                ("human", "{question}"),
            ])

            # Helper to format docs
            def format_docs(docs):
                return "\n\n".join(doc.page_content for doc in docs) if docs else ""

            # Define Chain
            rag_chain = (
                {
                    "context": retriever | format_docs,
                    "history": lambda x: history,
                    "question": lambda x: x
                }
                | prompt
                | llm
                | StrOutputParser()
            )
            
            # Execute
            response_text = rag_chain.invoke(message)

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
        # Google Gemini
        # Hardcoded key as per previous implementation (Recommend moving to System Parameters)
        API_KEY = "AIzaSyC3OdoD-Njt2SOpTBVPM4Z28JHiIhRfyTs"
        
        return ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            google_api_key=API_KEY,
            temperature=0.7,
            convert_system_message_to_human=True # Helps with some Gemini versions
        )

        # Ollama Implementation (Commented out)
        # return ChatOllama(
        #     model="llama3.2:latest",
        #     temperature=0.7,
        #     base_url="http://localhost:11434"
        # )

    def _get_history(self, conversation_id, exclude_id=None):
        """Fetch history and convert to LangChain Messages."""
        domain = [('conversation_id', '=', conversation_id)]
        if exclude_id:
            domain.append(('id', '!=', exclude_id))
            
        # CORRECT LOGIC: Fetch the *latest* N messages (desc), then reverse them to chronological order (asc)
        messages = self.env['ask.odoo.message'].search(
            domain,
            order='create_date desc',
            limit=10
        )
        
        # Odoo Recordset to list, reversed to be chronological [Oldest -> Newest] for the LLM
        # Note: messages is a recordset ordered by DESC (Newest -> Oldest). 
        # We need Oldest -> Newest for the chat history.
        history_records = list(messages)[::-1] 
        
        history = []
        for msg in history_records:
            if msg.type == 'user':
                history.append(HumanMessage(content=msg.content or ""))
            else:
                history.append(AIMessage(content=msg.content or ""))
                
        _logger.info(f"Retrieved {len(history)} messages for history context.")
        return history

