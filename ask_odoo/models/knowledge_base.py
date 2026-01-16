from odoo import models, fields, api

class KnowledgeBaseDocument(models.Model):
    """
    Model for storing knowledge base documents for RAG.
    """
    _name = 'ask.odoo.knowledge.document'
    _description = 'Knowledge Base Document'

    name = fields.Char(required=True)
    content = fields.Text()
    data = fields.Binary(attachment=True)
    filename = fields.Char()
    
    @api.model
    def get_all_documents(self):
        """Returns simplified list of documents for frontend."""
        docs = self.search([], order='create_date desc')
        return [{
            'id': d.id,
            'name': d.name,
            'description': f"Size: {len(d.content or '')} chars",
            'lastUpdated': d.write_date
        } for d in docs]

    @api.model
    def create_document(self, name, data, filename):
        """Creates a document from upload."""
        # In a real implementation, this would process the file content
        # and generate embeddings.
        self.create({
            'name': name,
            'data': data,
            'filename': filename,
            'content': "Processed content placeholder" 
        })
        return True

    @api.model
    def delete_document(self, doc_id):
        self.browse(doc_id).unlink()
        return True
