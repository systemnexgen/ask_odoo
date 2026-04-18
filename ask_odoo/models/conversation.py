from odoo import models, fields, api
import logging
import json
from langchain_core.messages import HumanMessage, AIMessage

_logger = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────────────────────
HISTORY_WINDOW = 10             # Max number of past messages sent to the LLM
CONVERSATION_TITLE_LENGTH = 30  # Truncation length for auto-generated titles

class AskOdooModel(models.Model):
    _inherit = 'ask.odoo.model'

    # ── Shared Conversation Helpers ──────────────────────────────────────────

    def _ensure_conversation(self, message, conversation_id=None, default_name='New Chat'):
        """Get or create a conversation and save the user's message.

        Returns (conversation_record, user_message_record).
        Used by both db_mode and rag_mode to avoid duplicating the same
        setup logic.
        """
        if conversation_id:
            conversation = self.env['ask.odoo.conversation'].browse(conversation_id)
        else:
            conversation = self.env['ask.odoo.conversation'].create({
                'name': default_name,
                'user_id': self.env.user.id,
            })

        user_msg = self.env['ask.odoo.message'].create({
            'conversation_id': conversation.id,
            'type': 'user',
            'content': message,
        })
        return conversation, user_msg

    def _update_conversation_metadata(self, conversation, message):
        """Update last_activity and auto-generate a title for new conversations."""
        conversation.write({'last_activity': fields.Datetime.now()})
        if len(conversation.message_ids) <= 2:
            title = (message[:CONVERSATION_TITLE_LENGTH] + "..."
                     if len(message) > CONVERSATION_TITLE_LENGTH else message)
            conversation.write({'name': title})

    def _get_history(self, conversation_id, exclude_id=None):
        """Fetch history and convert to LangChain Messages."""
        domain = [('conversation_id', '=', conversation_id)]
        if exclude_id:
            domain.append(('id', '!=', exclude_id))
            
        # Fetch latest N messages
        messages = self.env['ask.odoo.message'].search(
            domain,
            order='id desc',
            limit=HISTORY_WINDOW,
        )
        
        # Sort by ID ascending explicitly to ensure chronological order [Oldest -> Newest]
        history_records = messages.sorted(lambda m: m.id)
        
        history = []
        for msg in history_records:
            if msg.type == 'user':
                history.append(HumanMessage(content=msg.content or ""))
            else:
                content = msg.content or ""
                if content.startswith("[HIDDEN]"):
                    content = content[8:]
                history.append(AIMessage(content=content))
                
        _logger.info("Retrieved %d messages for history. IDs: %s", len(history), history_records.ids)
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
            # Skip hidden messages for frontend rendering
            if m.content and m.content.startswith("[HIDDEN]"):
                continue
                
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

