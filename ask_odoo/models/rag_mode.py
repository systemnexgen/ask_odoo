from odoo import models
import logging
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
_logger = logging.getLogger(__name__)

def _extract_text_content(content):
    """
    Extract string content from LangChain message content,
    handling list of dict/string blocks (e.g. Gemini/Gemma models).
    """
    if not content:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for p in content:
            if isinstance(p, dict):
                if p.get("type") == "text":
                    parts.append(p.get("text", ""))
            else:
                parts.append(str(p))
        return "".join(parts)
    return str(content)


class AskOdooModel(models.Model):
    _inherit = 'ask.odoo.model'

    def process_message_rag(self, message, conversation_id=None, chat_mode='conversation'):
        """
        Main orchestrator using LangChain.
        """
        _logger.info("AskOdoo: Processing message in mode: %s", chat_mode)

        # 1. Handle Conversation
        conversation, user_msg = self._ensure_conversation(message, conversation_id)

        # 2. Build & Run Chain
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
            
            _logger.info(
                "\n=== [AskOdoo] DEBUG: Context Retrieved ===\n%s\n===========================================",
                context_text,
            )

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
            _logger.info(
                "\n=== [AskOdoo] DEBUG: Full Prompt Payload ===\n%s\n============================================",
                prompt_debug_str,
            )

            # 4. Invoke LLM directly
            ai_message = llm.invoke(prompt_messages)
            response_text = _extract_text_content(ai_message.content)
            
            _logger.info(
                "\n=== [AskOdoo] DEBUG: LLM Response ===\n%s\n=====================================",
                response_text,
            )


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
        self._update_conversation_metadata(conversation, message)

        return {
            'response': response_text,
            'conversation_id': conversation.id,
            'title': conversation.name
        }
