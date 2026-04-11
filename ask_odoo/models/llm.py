from odoo import models
from odoo.tools import config
import os
from dotenv import load_dotenv
from langchain_postgres.vectorstores import PGVector
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()

class AskOdooModel(models.Model):
    _inherit = 'ask.odoo.model'

    def _get_connection_string(self):
        db_name = self.env.cr.dbname
        user = config.get('db_user')
        password = config.get('db_password')
        host = config.get('db_host') or 'localhost'
        port = config.get('db_port') or 5432
        return f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{db_name}"

    def _get_llm(self):
        # Groq Implementation
        GROQ_API_KEY = os.getenv("GROQ_API_KEY")
        # print(f"Groq API Key: {GROQ_API_KEY}")
        return ChatGroq(
            model="llama-3.1-8b-instant",
            # model="openai/gpt-oss-120b",
            groq_api_key=GROQ_API_KEY,
            temperature=0
        )

    def _get_embeddings(self):
        """Returns the shared HuggingFace embeddings model."""
        if not type(self)._embeddings:
            type(self)._embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        return type(self)._embeddings

    def _get_retriever(self):
        if not type(self)._vector_store:
            # Initialize Embeddings
            self._get_embeddings()
            
            # Initialize PGVector
            # Note: PGVector expects specific extension and tables. 
            # We use the standard LangChain implementation.
            connection = self._get_connection_string()
            type(self)._vector_store = PGVector(
                embeddings=type(self)._embeddings,
                collection_name="ask_odoo_knowledge_chunk",
                connection=connection,
                use_jsonb=True,
            )
        
        return type(self)._vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 3}
        )

    def _get_schema_vector_store(self):
        """
        Returns the PGVector store specifically for Database Schema.
        Uses a separate collection 'ask_odoo_schema'.
        """
        if not type(self)._schema_vector_store:
            # Ensure Embeddings are ready
            self._get_embeddings()

            connection = self._get_connection_string()
            
            type(self)._schema_vector_store = PGVector(
                embeddings=type(self)._embeddings,
                collection_name="ask_odoo_schema",
                connection=connection,
                use_jsonb=True,
            )
        return type(self)._schema_vector_store

    def _get_schema_retriever(self):
        """
        Returns a retriever for the schema vector store.
        k=5: Phase 1 fetches 5 model candidates; Phase 2 (_get_relevant_schema)
        will trim each model's fields, so the extra candidates cost very little.
        """
        vector_store = self._get_schema_vector_store()
        return vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 3}
        )


