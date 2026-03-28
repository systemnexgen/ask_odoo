from odoo import models, fields, api
import logging
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

_logger = logging.getLogger(__name__)

class AskOdooModel(models.Model):
    _inherit = 'ask.odoo.model'

    def process_message_rag(self, message, conversation_id=None, chat_mode='conversation'):
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
                "Odoo AI Assistant. Answer the current question only, using the knowledge base below. "
                "Be short and accurate. Do not repeat previous answers or assume.\n\n"
                "KNOWLEDGE BASE:\n{context}"
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

