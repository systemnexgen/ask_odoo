from odoo import models
from .utils import get_pg_connection_string
import os
from dotenv import load_dotenv
from langchain_postgres.vectorstores import PGVector
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEndpointEmbeddings

load_dotenv()

_embeddings_instance = None
_vector_store_instance = None
_schema_vector_store_instance = None
_llm_instance = None

class AskOdooModel(models.Model):
    _inherit = 'ask.odoo.model'

    def _get_connection_string(self):
        return get_pg_connection_string(self.env)

    def _get_llm(self):
        """Returns the shared ChatGroq LLM client (singleton)."""
        global _llm_instance
        if _llm_instance is None:
            GROQ_API_KEY = os.getenv("GROQ_API_KEY")
            # print(f"Groq API Key: {GROQ_API_KEY}")
            _llm_instance = ChatGroq(
                # model="llama-3.1-8b-instant",
                model="openai/gpt-oss-120b",
                # model = "llama-3.3-70b-versatile",
                groq_api_key=GROQ_API_KEY,
                temperature=0
            )
        return _llm_instance

    def _get_embeddings(self):
        """Returns the shared HuggingFace embeddings model."""
        global _embeddings_instance
        if _embeddings_instance is None:
            _embeddings_instance = HuggingFaceEndpointEmbeddings(
                model="sentence-transformers/all-MiniLM-L6-v2",
                huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN")
            )
        return _embeddings_instance

    def _get_retriever(self):
        global _vector_store_instance
        if _vector_store_instance is None:
            # Initialize Embeddings
            emb = self._get_embeddings()
            
            # Initialize PGVector
            # Note: PGVector expects specific extension and tables. 
            # We use the standard LangChain implementation.
            connection = self._get_connection_string()
            _vector_store_instance = PGVector(
                embeddings=emb,
                collection_name="ask_odoo_knowledge_chunk",
                connection=connection,
                use_jsonb=True,
            )
        
        return _vector_store_instance.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 3}
        )

    def _get_schema_vector_store(self):
        """
        Returns the PGVector store specifically for Database Schema.
        Uses a separate collection 'ask_odoo_schema'.
        """
        global _schema_vector_store_instance
        if _schema_vector_store_instance is None:
            # Ensure Embeddings are ready
            emb = self._get_embeddings()

            connection = self._get_connection_string()
            
            _schema_vector_store_instance = PGVector(
                embeddings=emb,
                collection_name="ask_odoo_schema",
                connection=connection,
                use_jsonb=True,
            )
        return _schema_vector_store_instance

    def _get_schema_retriever(self):
        """
        Returns a retriever for the schema vector store.
        k=5: Phase 1 fetches 5 model candidates; Phase 2 (_get_relevant_schema)
        will trim each model's fields, so the extra candidates cost very little.
        """
        vector_store = self._get_schema_vector_store()
        return vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 5}
        )


