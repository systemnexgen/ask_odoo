from odoo import models, fields, api
import logging
import json
from langchain_core.messages import HumanMessage, AIMessage

_logger = logging.getLogger(__name__)

class AskOdooModel(models.Model):
    _inherit = 'ask.odoo.model'

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
        result = []
        for m in messages:
            msg = {
                'id': m.id,
                'text': m.content,
                'type': m.type,
            }
            if m.result_html:
                msg['result_html'] = m.result_html
            if m.chart_data_json:
                try:
                    msg['chart_data'] = json.loads(m.chart_data_json)
                except Exception:
                    pass
            if m.action_code:
                msg['action_code'] = m.action_code
        
            result.append(msg)
        return result

    @api.model
    def delete_conversation(self, conversation_id):
        """Delete a conversation and all its messages (cascade)."""
        conversation = self.env['ask.odoo.conversation'].browse(conversation_id)
        if conversation.exists():
            conversation.unlink()
        return True

