from odoo import models, fields, api

class AskOdooConversation(models.Model):
    # Stores the chat sessions
    _name = 'ask.odoo.conversation'
    _description = 'AI Chat Conversation'
    _order = 'write_date desc'

    name = fields.Char(string='Title', required=True, default='New Chat')
    user_id = fields.Many2one('res.users', string='User', default=lambda self: self.env.user, required=True)
    message_ids = fields.One2many('ask.odoo.message', 'conversation_id', string='Messages')
    # Track when the chat was last active for sorting in the sidebar
    last_activity = fields.Datetime(string='Last Activity', default=fields.Datetime.now)

class AskOdooMessage(models.Model):
    # Stores the messages within a chat session
    _name = 'ask.odoo.message'
    _description = 'AI Chat Message'
    _order = 'create_date asc'

    conversation_id = fields.Many2one('ask.odoo.conversation', string='Conversation', required=True, ondelete='cascade')
    type = fields.Selection([('user', 'User'), ('ai', 'AI')], string='Type', required=True)
    content = fields.Text(string='Content', required=True)
