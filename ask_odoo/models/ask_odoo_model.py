from odoo import models, fields, api
import logging

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
