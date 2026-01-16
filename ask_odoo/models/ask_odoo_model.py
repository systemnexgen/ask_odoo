from odoo import models, fields, api
import google.generativeai as genai
import logging
import requests
import json

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None

_logger = logging.getLogger(__name__)

class AskOdooModel(models.Model):
    _name = 'ask.odoo.model'
    _description = 'Ask Odoo Model'

    name = fields.Char(required=True)
    description = fields.Text()
    user_id = fields.Many2one('res.users', default=lambda self: self.env.user)
    active = fields.Boolean(default=True)

    # Embedding Model Cache
    _embedding_model = None

    # ==========================
    # PUBLIC ENTRY POINT (RPC)
    # ==========================
    @api.model
    def process_message(self, message, conversation_id=None):
        """
        Main orchestrator with Database Persistence.
        """
        # 1. Handle Conversation
        if conversation_id:
            conversation = self.env['ask.odoo.conversation'].browse(conversation_id)
        else:
            conversation = self.env['ask.odoo.conversation'].create({
                'name': 'New Chat',  # Will serve as temporary title
                'user_id': self.env.user.id
            })

        # 2. Save User Message
        self.env['ask.odoo.message'].create({
            'conversation_id': conversation.id,
            'type': 'user',
            'content': message,
        })

        # 3. Build Context (History)
        # Fetch last 10 messages for context (in ascending order for the prompt)
        history_msgs = self.env['ask.odoo.message'].search(
            [('conversation_id', '=', conversation.id)],
            order='create_date desc',
            limit=10
        )
        # Reverse to get chronological order for the AI
        context_history = history_msgs.sorted(lambda m: m.create_date)

        # 4. RAG & Prompt Construction
        context = self._build_context(message, conversation.id)
        documents = self._retrieve_documents(context)  # Placeholder RAG
        prompt = self._build_prompt(context, documents, context_history)

        # 5. Call LLM
        response_text = self._call_llm(prompt)

        # 6. Save AI Response
        self.env['ask.odoo.message'].create({
            'conversation_id': conversation.id,
            'type': 'ai',
            'content': response_text,
        })

        # 7. Update Last Activity & Title (if new)
        conversation.write({'last_activity': fields.Datetime.now()})

        if len(conversation.message_ids) <= 2:
            # Generate a short title based on the first user message
            title = message[:30] + "..." if len(message) > 30 else message
            conversation.write({'name': title})

        return {
            'response': self._post_process(response_text),
            'conversation_id': conversation.id,
            'title': conversation.name
        }

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
        return [{
            'id': m.id,
            'text': m.content,
            'type': m.type
        } for m in messages]

    # ==========================
    # CONTEXT BUILDING
    # ==========================
    def _build_context(self, message, conversation_id=None):
        return {
            "user_message": message,
            "conversation_id": conversation_id,
            "user_name": self.env.user.name,
        }

    # ==========================
    # RAG RETRIEVAL (PGVECTOR)
    # ==========================
    def _retrieve_documents(self, context):
        """
        Retrieves relevant documents using Vector Search (pgvector).
        """
        user_message = context.get('user_message', '')
        if not user_message:
            return []

        # 1. Load Embedding Model (Lazy)
        if not AskOdooModel._embedding_model:
            if not SentenceTransformer:
                _logger.error("sentence_transformers library not installed. Cannot perform vector search.")
                return ["System Error: Vector Search unavailable."]
            
            _logger.info("Loading SentenceTransformer model 'all-MiniLM-L6-v2' for Retrieval...")
            AskOdooModel._embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

        # 2. Generate Query Embedding
        try:
            # encode returns a numpy array, we need a list for SQL
            query_embedding = AskOdooModel._embedding_model.encode([user_message])[0].tolist()
        except Exception as e:
            _logger.error(f"Failed to generate embedding: {e}")
            return []

        # 3. Vector Search via SQL
        # Using cosine distance operator <=> from pgvector
        # Results are ordered by distance ASC (closest first)
        limit = 3
        sql = """
            SELECT c.content, d.name, (c.embedding <=> %s::vector) as distance
            FROM ask_odoo_knowledge_chunk c
            JOIN ask_odoo_knowledge_document d ON c.document_id = d.id
            ORDER BY distance ASC
            LIMIT %s
        """
        try:
            self.env.cr.execute(sql, (str(query_embedding), limit))
            results = self.env.cr.fetchall()
        except Exception as e:
            _logger.error(f"Vector search SQL failed: {e}")
            return []

        _logger.info(f"RAG Vector Search found {len(results)} chunks.")
        docs = []
        for content, doc_name, distance in results:
            # distance is cosine distance (0-2).
            # We present it as relevant context.
            docs.append(f"Content from {doc_name} (Distance: {distance:.4f}):\n{content}")

        return docs

    # ==========================
    # PROMPT ASSEMBLY
    # ==========================
    def _build_prompt(self, context, documents, history_records):
        # Build Chat History Block
        history_block = ""
        for msg in history_records:
            role = "User" if msg.type == 'user' else "Assistant"
            history_block += f"{role}: {msg.content}\n"

        # Build Documents Block
        docs_block = ""
        if documents:
            docs_block = "RELAVENT KNOWLEDGE BASE DOCUMENTS:\n" + "\n\n".join(documents) + "\n\n"

        system_prompt = (
            "Use the provided KNOWLEDGE BASE DOCUMENTS to answer the user's question. "
            "If the answer is found in the documents, Use the exact wordings from the document. "
            "If the answer is NOT in the documents, answer based on your general Odoo knowledge, but you MUST mention that you didn't find specific info in the uploaded files.\n\n"
        )
        return f"{system_prompt}{docs_block}CONVERSATION HISTORY:\n{history_block}\nUser: {context['user_message']}\nAssistant:"

    # ==========================
    # LLM CALL (SWAPPABLE)
    # ==========================
    def _call_llm(self, prompt):
        # """
        # LLM abstraction layer.
        # Switched to Ollama (gemma3:1b).
        # """
        # OLLAMA_URL = "http://localhost:11434/api/generate"
        # MODEL_NAME = "llama3.2:latest"
        
        # try:
        #     payload = {
        #         "model": MODEL_NAME,
        #         "prompt": prompt,
        #         "stream": False,
        #         "options": {
        #             "num_ctx": 4096,  # Increase context window
        #             "num_predict": 1024, # Allow for longer answers
        #             "temperature": 0.7 
        #         }
        #     }
        #     response = requests.post(OLLAMA_URL, json=payload, timeout=120)
        #     response.raise_for_status()
        #     result = response.json()
        #     return result.get("response", "No response from Ollama.")
        
        # except Exception as e:
        #     _logger.exception("LLM call failed")
        #     return f"AI Error (Ollama): {str(e)}"

        # You can swap this for 'gemini-1.5-pro' if you need more reasoning power
        MODEL_NAME = "gemini-3-flash-preview"
        # Ideally, fetch this from Odoo System Parameters
        API_KEY = "AIzaSyCaB4B_Q2dnz8k7fxpyZ-Zrub8v9txB1u8"
        GEMINI_URL = f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL_NAME}:generateContent?key={API_KEY}"

        try:
            # Gemini requires a specific 'contents' > 'parts' structure
            payload = {
                "contents": [{
                    "parts": [{"text": prompt}]
                }],
                "generationConfig": {
                    "maxOutputTokens": 1024,
                    "temperature": 0.7
                }
            }

            # Standard Odoo headers usually include content-type, but requests handles json= well
            headers = {'Content-Type': 'application/json'}
            
            response = requests.post(GEMINI_URL, json=payload, headers=headers, timeout=120)
            response.raise_for_status()
            result = response.json()
            
            # Parse Gemini's nested response structure
            # candidates[0] -> content -> parts[0] -> text
            try:
                return result['candidates'][0]['content']['parts'][0]['text']
            except (KeyError, IndexError):
                # Handle cases where safety filters block the response
                return "No response generated. (Check safety settings or API quota)."
        
        except Exception as e:
            _logger.exception("LLM call failed")
            return f"AI Error (Gemini): {str(e)}"

    # ==========================
    # POST PROCESSING
    # ==========================
    def _post_process(self, response_text):
        """
        Final formatting layer.
        Later:
        - Markdown cleanup
        - Safety filtering
        - Code highlighting
        - Logging
        """
        return (
            f"{response_text}\n\n"
            f"(Server Time: {fields.Datetime.now()})"
        )
