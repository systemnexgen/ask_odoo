import base64
import tempfile
import os
import logging
from odoo import models, fields, api
from odoo.tools import config

from langchain_postgres.vectorstores import PGVector
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

_logger = logging.getLogger(__name__)

class KnowledgeBaseDocument(models.Model):
    _name = 'ask.odoo.knowledge.document'
    _description = 'Knowledge Base Document'
    _order = 'last_updated desc'

    name = fields.Char(required=True)
    file_data = fields.Binary(string='File Content', required=True)
    file_name = fields.Char(string='File Name')
    description = fields.Text()
    last_updated = fields.Datetime(default=fields.Datetime.now)
    processed_content = fields.Text(string='Processed Text')
    vector_id = fields.Char(string='Vector DB ID') # Legacy field
    
    # Caches
    _vector_store = None
    _embeddings = None

    @api.model
    def create_document(self, name, file_content, file_name):
        """Creates a document and triggers processing."""
        doc = self.create({
            'name': name,
            'file_data': file_content,
            'file_name': file_name,
            'description': f"Uploaded PDF: {file_name}",
            'last_updated': fields.Datetime.now()
        })
        # Automatically process after upload
        doc.process_document()
        return doc.id

    def _get_connection_string(self):
        db_name = self.env.cr.dbname
        user = config.get('db_user')
        password = config.get('db_password')
        host = config.get('db_host') or 'localhost'
        port = config.get('db_port') or 5432
        return f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{db_name}"

    def process_document(self):
        """Processes the PDF content, chunks it, and generates embeddings using LangChain PGVector."""
        
        # 1. Initialize Components
        connection = self._get_connection_string()
        
        # Singleton-ish pattern for efficiency if processing multiple docs in one transaction
        if not KnowledgeBaseDocument._embeddings:
             KnowledgeBaseDocument._embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        
        vector_store = PGVector(
            embeddings=KnowledgeBaseDocument._embeddings,
            collection_name="ask_odoo_knowledge_chunk",
            connection=connection,
            use_jsonb=True,
        )

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=100,
            length_function=len,
        )

        for doc in self:
            if not doc.file_data:
                continue
                
            try:
                # 2. Decode generic Base64
                file_content = base64.b64decode(doc.file_data)
                
                # 3. Write to temp file for PyPDFLoader
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                    tmp_file.write(file_content)
                    tmp_path = tmp_file.name

                _logger.info(f"Processing PDF: {tmp_path}")

                try:
                    # 4. Load & Split
                    loader = PyPDFLoader(tmp_path)
                    raw_docs = loader.load()
                    
                    # Split into chunks
                    chunks = text_splitter.split_documents(raw_docs)
                    _logger.info(f"Extracted {len(chunks)} chunks from {doc.name}")

                    # 5. Enrich Metadata
                    # We store Odoo's document_id in metadata so we can filter/delete later if needed
                    for chunk in chunks:
                        chunk.metadata['document_id'] = doc.id
                        chunk.metadata['source_file'] = doc.file_name

                    # 6. Bulk Add to Vector Store
                    if chunks:
                        vector_store.add_documents(chunks)
                        _logger.info(f"Successfully processed {doc.name}: stored {len(chunks)} embeddings.")

                    doc.write({
                        'processed_content': "Processed via LangChain PGVector", 
                        'description': f"Processed {len(raw_docs)} pages, {len(chunks)} chunks. {doc.description or ''}"
                    })
                    
                finally:
                    # Cleanup temp file
                    if os.path.exists(tmp_path):
                        os.remove(tmp_path)

            except Exception as e:
                _logger.error(f"Error processing document {doc.id}: {e}")
                doc.description = f"Error processing: {str(e)}"
    
    @api.model
    def delete_document(self, doc_id):
        """Deletes a document by ID and cleans up its vector embeddings."""
        doc = self.browse(doc_id)
        if not doc.exists():
            return False

        try:
            # 1. Clean up Vectors via Raw SQL
            query = """
                DELETE FROM langchain_pg_embedding e
                USING langchain_pg_collection c
                WHERE e.collection_id = c.uuid
                  AND c.name = 'ask_odoo_knowledge_chunk'
                  AND e.cmetadata->>'document_id' = %s
            """
            self.env.cr.execute(query, (str(doc_id),))
            _logger.info(f"Deleted vectors for document ID {doc_id}")

        except Exception as e:
            _logger.error(f"Failed to delete vectors for doc {doc_id}: {e}")

        doc.unlink()
        return True

    @api.model
    def get_all_documents(self):
        """Returns list of documents for the UI."""
        docs = self.search([])
        return [{
            'id': d.id,
            'name': d.name,
            'description': d.description or 'No description',
            'lastUpdated': d.last_updated.strftime('%Y-%m-%d %H:%M:%S') if d.last_updated else '',
        } for d in docs]
