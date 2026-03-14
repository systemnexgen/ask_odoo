from odoo import models, fields, api
from odoo.tools import config
from odoo.tools.safe_eval import safe_eval
import re
import logging
import json
import base64
import os
import pandas as pd
from dotenv import load_dotenv
load_dotenv()

from langchain_postgres.vectorstores import PGVector
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_community.chat_models import ChatOllama
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

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

    def process_message_db(self, message, conversation_id=None):
        """
        Handles database / schema related queries using Schema RAG.
        This does NOT execute CRUD yet — only explains models, fields, relations.
        """
        chat_mode = "database"
        _logger.info("AskOdoo: Processing message in DATABASE mode")

        # 1. Handle Conversation
        if conversation_id:
            conversation = self.env['ask.odoo.conversation'].browse(conversation_id)
        else:
            conversation = self.env['ask.odoo.conversation'].create({
                'name': 'New DB Chat',
                'user_id': self.env.user.id
            })

        # 2. Save User Message
        user_msg = self.env['ask.odoo.message'].create({
            'conversation_id': conversation.id,
            'type': 'user',
            'content': message,
        })

        try:
            # 3. Components
            llm = self._get_llm()
            retriever = self._get_schema_retriever()

            history = self._get_history(conversation.id, exclude_id=user_msg.id)

            # 4. Retrieve Schema Context
            docs = retriever.invoke(message)

            context_text = (
                "\n\n".join(doc.page_content for doc in docs)
                if docs else "No relevant schema found."
            )

            _logger.info(
                f"\n=== [AskOdoo][DB MODE] Schema Context ===\n{context_text}\n========================================="
            )

            # 5. Database-Specific Prompt
            system_template = (
                "Odoo DB Assistant. READ operations only. Use ONLY the schema below.\n\n"
                "RULES:\n"
                "- Generate Odoo ORM code using `self.env`. Wrap in ```python ... ``` block.\n"
                "- Assign final output to `result`. No `print()`.\n"
                "- For tables: return list of dicts or list of lists (first row = headers).\n"
                "- Use `limit=1` when searching for a single record ID.\n"
                "- Use HTML tags (not Markdown) for explanation text.\n"
                "- If not in schema, say you don't know.\n\n"
                "PRE-INJECTED (no imports allowed):\n"
                "`self`, `env`, `datetime` (.now()), `date` (.today()), `timedelta`, `time`\n\n"
                "- If history shows a previous ❌ error, do NOT repeat the same mistake.\n\n"
                "SCHEMA:\n{context}"
            )


            prompt = ChatPromptTemplate.from_messages([
                ("system", system_template),
                MessagesPlaceholder(variable_name="history"),
                ("human", "{question}"),
            ])

            prompt_messages = prompt.format_messages(
                context=context_text,
                history=history,
                question=message
            )

            _logger.info(
                "\n=== [AskOdoo][DB MODE] Full Prompt ===\n" +
                "\n".join(f"[{m.type.upper()}]: {m.content}" for m in prompt_messages) +
                "\n====================================="
            )

            # 6. Invoke LLM with a silent server-side retry loop.
            # Before the user ever sees a confirmation card, we validate that
            # every self.env['model.name'] reference in the generated code
            # actually exists in the Odoo registry. If not, we feed the error
            # back to the LLM as additional messages and try again—silently.
            MAX_CODE_RETRIES = 3
            messages_for_llm = prompt_messages  # start with the original prompt
            response_text = ""

            for attempt in range(MAX_CODE_RETRIES + 1):
                ai_message = llm.invoke(messages_for_llm)
                response_text = ai_message.content

                _logger.info(
                    f"\n=== [AskOdoo][DB MODE] LLM Response (attempt {attempt + 1}) ===\n"
                    f"{response_text}\n====================================="
                )

                # Extract candidate code
                code_match = re.search(r"```python\n(.*?)\n```", response_text, re.DOTALL)
                if not code_match:
                    # No code block — nothing to validate, return as-is
                    break

                candidate_code = code_match.group(1).strip()
                is_valid, val_error = self._validate_code_models(candidate_code)

                if is_valid:
                    _logger.info(f"AskOdoo: Code passed model validation on attempt {attempt + 1}.")
                    break

                if attempt < MAX_CODE_RETRIES:
                    _logger.warning(
                        f"AskOdoo: Code validation failed (attempt {attempt + 1}): {val_error}. "
                        f"Silently retrying..."
                    )
                    # Append the AI's bad response and a correction request
                    # as new messages so the LLM sees its mistake
                    messages_for_llm = messages_for_llm + [
                        AIMessage(content=response_text),
                        HumanMessage(content=(
                            f"The code you just generated is invalid: {val_error}\n"
                            f"This model does NOT exist in this Odoo instance. "
                            f"Use only models listed in the DATABASE SCHEMA DOCUMENTS above. "
                            f"Output the corrected code block only."
                        ))
                    ]
                else:
                    _logger.error(
                        f"AskOdoo: Code still invalid after {MAX_CODE_RETRIES} retries. "
                        f"Returning last response to user."
                    )

        except Exception as e:
            _logger.exception("Database Mode Failed")
            response_text = f"Error: {str(e)}"

        # 7. Save AI Message
        self.env['ask.odoo.message'].create({
            'conversation_id': conversation.id,
            'type': 'ai',
            'content': response_text,
        })

        # 8. Update Conversation
        conversation.write({'last_activity': fields.Datetime.now()})

        if len(conversation.message_ids) <= 2:
            conversation.write({
                'name': message[:30] + "..." if len(message) > 30 else message
            })

        # 9. Extract Code if present
        action_code = None
        code_match = re.search(r"```python\n(.*?)\n```", response_text, re.DOTALL)
        if code_match:
            action_code = code_match.group(1).strip()
            # Save confirmation code as a separate message for history reload
            try:
                self.env['ask.odoo.message'].create({
                    'conversation_id': conversation.id,
                    'type': 'confirmation',
                    'content': 'Proposed Action',
                    'action_code': action_code,
                })
            except Exception:
                _logger.warning("AskOdoo: Could not save confirmation to history")

        return {
            'response': response_text,
            'conversation_id': conversation.id,
            'title': conversation.name,
            'action_code': action_code,
        }

    @api.model
    def _explain_error_to_user(self, code, error_message, conversation_id=None):
        """
        Uses the LLM to produce a structured 3-part error response.
        Now includes conversation history so the LLM can see previous
        failed attempts and avoid repeating the same mistake.
        """
        try:
            llm = self._get_llm()

            system_template = (
                "Code execution failed. Respond in EXACTLY this format:\n\n"
                "**❌ Error:** [what went wrong]\n"
                "**Reason:** [why]\n\n"
                "**💡 Proposed Fix:** [ORM change description]\n"
                "```python\nresult = ...\n```\n\n"
                "**Would you like me to try again with this fix?**\n\n"
                "Only fix ORM code. No imports. Assign to `result`.\n"
            )

            human_template = (
                "Code that was executed:\n"
                "```python\n{code}\n```\n\n"
                "Error:\n{error}\n\n"
                "Respond using the exact 3-part format described."
            )

            # Include conversation history so LLM sees previous failures
            history = []
            if conversation_id:
                history = self._get_history(conversation_id)

            prompt = ChatPromptTemplate.from_messages([
                ("system", system_template),
                MessagesPlaceholder(variable_name="history"),
                ("human", human_template),
            ])

            messages = prompt.format_messages(
                code=code,
                error=error_message,
                history=history,
            )

            response = llm.invoke(messages)
            return response.content
        except Exception as e:
            _logger.error(f"Failed to explain error with LLM: {e}")
            return f"**❌ Error:** {error_message}\n\n**Would you like me to try again?**"

    @api.model
    def execute_confirmed_code(self, code, conversation_id):
        """
        Executes the confirmed Python code safely and saves the result
        back to the conversation history so the LLM can learn from
        execution errors on the next user turn.
        """
        try:
            from datetime import timedelta
            local_context = {
                'self': self,
                'env': self.env,
                'result': None,
                'time': fields.Datetime.now,
                'datetime': fields.Datetime,
                'date': fields.Date,
                'timedelta': timedelta,   # pre-inject so LLM code can use it without import
            }
            # Use a savepoint so that if the generated code causes a DB error,
            # the transaction rolls back to here instead of poisoning everything.
            with self.env.cr.savepoint():
                safe_eval(code, local_context, mode="exec", nocopy=True)
            
            result = local_context.get('result')
            
            
            # Post-processing: If result is a RecordSet, make it readable
            if result is not None and hasattr(result, '_name') and hasattr(result, 'mapped'):
                # It's an Odoo RecordSet
                if not result:
                    result = "No records found."
                else:
                    try:
                        # Prefer display_name or name
                        result = result.mapped('display_name')
                    except:
                        result = str(result)
            
            # Convert dict_keys, dict_values, sets, and other iterables to lists for better formatting
            if result is not None:
                # Check for dict_keys, dict_values, or similar view objects
                result_type = type(result).__name__
                if result_type in ('dict_keys', 'dict_values', 'dict_items', 'set', 'frozenset'):
                    result = list(result)
                # Also handle generators and other iterables (but not strings!)
                elif hasattr(result, '__iter__') and not isinstance(result, (str, bytes)):
                    # Check if it's not already a list or tuple
                    if not isinstance(result, (list, tuple)):
                        try:
                            result = list(result)
                        except:
                            pass  # Keep as-is if conversion fails

            # Try to convert lists to HTML tables using Pandas
            chart_data = None
            if isinstance(result, (list, tuple)):
                try:
                    if not result:
                        result = "No records found."
                    else:
                        df = pd.DataFrame(result)

                        # Check if result is a List of Lists (and not empty)
                        # If the first row looks like headers (all strings), promote it
                        if isinstance(result, list) and len(result) > 0 and isinstance(result[0], list):
                            first_row = result[0]
                            if first_row and all(isinstance(x, str) for x in first_row):
                                # Treat first row as headers
                                df = pd.DataFrame(result[1:], columns=first_row)

                        # Fix for simple lists becoming column "0" - rename to "Result" if single column
                        if len(df.columns) == 1 and str(df.columns[0]) == '0':
                            df.columns = ['Result']

                        # ── Chart Data Detection ──────────────────────────────
                        # Analyze the DataFrame to determine if it's chartable
                        chart_data = self._detect_chart_data(df)

                        # border=0 to let CSS handle it
                        # classes='dataframe' to ensure our CSS target hits it
                        result = df.to_html(index=False, border=0, classes='dataframe')
                except Exception:
                    # Fallback to string representation if conversion fails
                    result = str(result)
            else:
                result = str(result) if result is not None else "Action Executed Successfully"

            execution_result = {
                'status': 'success',
                'result': result,
                'chart_data': chart_data,
            }

            # ── Save success back to conversation history ─────────────────
            if conversation_id:
                try:
                    preview = result if isinstance(result, str) else str(result)
                    history_content = (
                        f"✅ Code executed successfully.\n"
                        f"Result preview: {preview[:300]}{'...' if len(preview) > 300 else ''}"
                    )
                    # Determine if result is HTML
                    is_html = isinstance(result, str) and (
                        result.strip().startswith('<table') or result.strip().startswith('<div')
                    )
                    msg_vals = {
                        'conversation_id': conversation_id,
                        'type': 'ai',
                        'content': history_content,
                    }
                    if is_html:
                        msg_vals['result_html'] = result
                    if chart_data:
                        msg_vals['chart_data_json'] = json.dumps(chart_data)
                    self.env['ask.odoo.message'].create(msg_vals)
                except Exception as hist_e:
                    _logger.warning(f"AskOdoo: Could not save success result to history: {hist_e}")

            return execution_result

        except Exception as e:
            _logger.exception("Execution Error")

            # ── Guard 1: Retry counter bail-out ───────────────────────────
            # Count consecutive ❌ errors in conversation. After 3, stop retrying.
            MAX_CONSECUTIVE_ERRORS = 3
            if conversation_id:
                error_msgs = self.env['ask.odoo.message'].search([
                    ('conversation_id', '=', conversation_id),
                    ('type', '=', 'ai'),
                    ('content', 'like', '❌ Execution Error'),
                ], order='id desc', limit=MAX_CONSECUTIVE_ERRORS)

                if len(error_msgs) >= MAX_CONSECUTIVE_ERRORS:
                    _logger.warning("AskOdoo: Max consecutive errors reached. Bailing out.")
                    bail_message = (
                        "**❌ Multiple attempts have failed.**\n\n"
                        "I was unable to fix this automatically. "
                        "Please try rephrasing your question or providing more details."
                    )
                    # Save bail-out to history
                    try:
                        self.env['ask.odoo.message'].create({
                            'conversation_id': conversation_id,
                            'type': 'ai',
                            'content': bail_message,
                        })
                    except Exception:
                        pass
                    return {
                        'status': 'error',
                        'message': bail_message,
                        'debug_message': str(e),
                        'retry_code': None,  # No retry offered
                    }

            # Generate structured error response (now with history awareness)
            friendly_error = self._explain_error_to_user(code, str(e), conversation_id)

            # Extract the corrected code block from the LLM's structured response
            retry_code = None
            code_match = re.search(r"```python\n(.*?)\n```", friendly_error, re.DOTALL)
            if code_match:
                retry_code = code_match.group(1).strip()

            # ── Guard 2: Duplicate detection ──────────────────────────────
            # If proposed fix is identical to the code that just failed, reject it.
            if retry_code and retry_code.strip() == code.strip():
                _logger.warning("AskOdoo: Proposed fix is identical to failed code. Rejecting.")
                retry_code = None
                friendly_error = (
                    f"**❌ Error:** {str(e)}\n\n"
                    "I was unable to generate a different fix. "
                    "Please try rephrasing your question or providing more details."
                )

            # ── Save error + proposed fix to conversation history ─────────
            if conversation_id:
                try:
                    history_content = (
                        f"❌ Execution Error for the previously proposed code:\n"
                        f"Code attempted:\n```python\n{code}\n```\n"
                        f"Error: {str(e)}\n"
                        + (f"Proposed fix code:\n```python\n{retry_code}\n```" if retry_code else "")
                    )
                    self.env['ask.odoo.message'].create({
                        'conversation_id': conversation_id,
                        'type': 'ai',
                        'content': history_content,
                    })
                except Exception as hist_e:
                    _logger.warning(f"AskOdoo: Could not save error to history: {hist_e}")

            return {
                'status': 'error',
                'message': friendly_error,
                'debug_message': str(e),
                'retry_code': retry_code,  # None = no retry button shown
            }


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

    # ==========================
    # HELPERS
    # ==========================
    
    def _validate_code_models(self, code):
        """
        Lightweight pre-execution guard: scans the generated code for all
        self.env['model.name'] references and checks each one exists in the
        Odoo registry. Returns (True, None) if all models are valid, or
        (False, error_message) for the first invalid model found.

        This is intentionally NOT a full execution — it is a fast name-check
        to catch the most common hallucination (wrong model technical name)
        before the user sees a broken confirmation card.
        """
        # Match both single and double-quoted model references
        model_refs = re.findall(
            r"self\.env\[['\"]([a-z][a-z0-9._]+)['\"]\]", code
        )
        for model_name in model_refs:
            if model_name not in self.env:
                # Build a helpful hint: find the closest real model name
                # (same prefix family) to give the LLM a nudge
                prefix = model_name.split('.')[0]
                suggestions = [
                    m for m in self.env.registry
                    if m.startswith(prefix) and not self.env[m]._abstract
                ][:5]  # top 5 matches
                hint = (
                    f" Did you mean one of: {suggestions}?" if suggestions else ""
                )
                return False, (
                    f"Model '{model_name}' does not exist in this Odoo instance.{hint}"
                )
        return True, None

    def _detect_chart_data(self, df):
        """
        Analyzes a Pandas DataFrame and returns Chart.js-compatible JSON
        if the data is suitable for visualization. Returns None otherwise.

        Auto-selects chart type:
          - Line chart: if the label column looks like dates/time
          - Pie/Doughnut: if ≤7 categories and only 1 numeric column
          - Bar chart: default for everything else
        """
        try:
            if df is None or df.empty or len(df) < 2:
                return None

            # Classify columns into labels (text/object) and values (numeric)
            label_cols = []
            value_cols = []

            for col in df.columns:
                col_str = str(col)
                # Skip columns named 'id' or ending in '_id' — not useful for charts
                if col_str.lower() == 'id' or col_str.lower().endswith('_id'):
                    continue

                if pd.api.types.is_numeric_dtype(df[col]):
                    value_cols.append(col)
                elif pd.api.types.is_object_dtype(df[col]) or pd.api.types.is_string_dtype(df[col]):
                    label_cols.append(col)

            # Need at least 1 label column and 1 numeric column
            if not label_cols or not value_cols:
                return None

            # Use the first label column as the X-axis
            label_col = label_cols[0]
            labels = df[label_col].astype(str).tolist()

            # Cap at 50 data points for readability
            if len(labels) > 50:
                return None

            # ── Chart Type Selection ──────────────────────────────────
            chart_type = 'bar'  # default

            # Check if labels look like dates/time → line chart
            is_time_series = False
            date_keywords = ['date', 'month', 'year', 'week', 'day', 'time', 'period', 'quarter']
            if any(kw in str(label_col).lower() for kw in date_keywords):
                is_time_series = True
                chart_type = 'line'

            # Small categories with single metric → pie chart
            if len(labels) <= 7 and len(value_cols) == 1 and not is_time_series:
                chart_type = 'doughnut'

            # ── Color Palette ─────────────────────────────────────────
            colors = [
                'rgba(99, 102, 241, 0.8)',    # Indigo
                'rgba(16, 185, 129, 0.8)',    # Emerald
                'rgba(245, 158, 11, 0.8)',    # Amber
                'rgba(239, 68, 68, 0.8)',     # Red
                'rgba(139, 92, 246, 0.8)',    # Violet
                'rgba(6, 182, 212, 0.8)',     # Cyan
                'rgba(236, 72, 153, 0.8)',    # Pink
                'rgba(34, 197, 94, 0.8)',     # Green
                'rgba(251, 146, 60, 0.8)',    # Orange
                'rgba(59, 130, 246, 0.8)',    # Blue
            ]
            border_colors = [c.replace('0.8)', '1)') for c in colors]

            # ── Build Datasets ────────────────────────────────────────
            datasets = []
            for i, vcol in enumerate(value_cols):
                # Convert values, coercing errors to 0
                values = pd.to_numeric(df[vcol], errors='coerce').fillna(0).tolist()

                dataset = {
                    'label': str(vcol),
                    'data': values,
                }

                if chart_type == 'doughnut':
                    # Pie/doughnut: each slice gets a different color
                    dataset['backgroundColor'] = colors[:len(values)]
                    dataset['borderColor'] = border_colors[:len(values)]
                    dataset['borderWidth'] = 2
                else:
                    # Bar/Line: each dataset gets one color
                    color_idx = i % len(colors)
                    dataset['backgroundColor'] = colors[color_idx]
                    dataset['borderColor'] = border_colors[color_idx]
                    dataset['borderWidth'] = 2

                    if chart_type == 'line':
                        dataset['fill'] = False
                        dataset['tension'] = 0.3

                datasets.append(dataset)

            chart_data = {
                'type': chart_type,
                'labels': labels,
                'datasets': datasets,
                'title': f"{', '.join(str(v) for v in value_cols)} by {label_col}",
            }

            _logger.info(f"AskOdoo: Chart detected — type={chart_type}, labels={len(labels)}, datasets={len(datasets)}")
            return chart_data

        except Exception as e:
            _logger.warning(f"AskOdoo: Chart detection failed (non-critical): {e}")
            return None

    def _get_connection_string(self):
        db_name = self.env.cr.dbname
        user = config.get('db_user')
        password = config.get('db_password')
        host = config.get('db_host') or 'localhost'
        port = config.get('db_port') or 5432
        return f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{db_name}"

    def _get_retriever(self):
        if not AskOdooModel._vector_store:
            # Initialize Embeddings
            if not AskOdooModel._embeddings:
                # Use HuggingFaceEmbeddings as replacement for locally managed SentenceTransformer
                # Ensure 'sentence_transformers' is installed or 'langchain_huggingface'
                AskOdooModel._embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
            
            # Initialize PGVector
            # Note: PGVector expects specific extension and tables. 
            # We use the standard LangChain implementation.
            connection = self._get_connection_string()
            AskOdooModel._vector_store = PGVector(
                embeddings=AskOdooModel._embeddings,
                collection_name="ask_odoo_knowledge_chunk",
                connection=connection,
                use_jsonb=True,
            )
        
        return AskOdooModel._vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 3}
        )

    def _get_llm(self):
        # Groq Implementation
        GROQ_API_KEY = os.getenv("GROQ_API_KEY")
        # print(f"Groq API Key: {GROQ_API_KEY}")
        return ChatGroq(
            model="llama-3.1-8b-instant",
            groq_api_key=GROQ_API_KEY,
            temperature=0.1
        )
        
        # # Google Gemini
        # print(f"Google API Key: {API_KEY}")
        # return ChatGoogleGenerativeAI(
        #     model="gemma-3-4b-it",
        #     google_api_key=API_KEY,
        #     temperature=0.1
        # )

        # # Ollama Implementation
        # return ChatOllama(
        #     model="llama3.2:latest",
        #     temperature=0,
        #     base_url="http://localhost:11434"
        # )

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

    def _schema_to_text(self, model_metadata):
        """
        Converts model schema to text with ALL fields included.
        No filters applied — every field is preserved.
        """
        # 1. Basic Info
        model_name = model_metadata.get('name', model_metadata.get('model'))
        model_tech_name = model_metadata.get('model')

        # 2. Include ALL Fields (no filtering)
        all_fields = []
        for f in model_metadata.get('fields', []):
            # Formatting: "Name (tech_name): type"
            # If relation exists, add "-> RelatedModel"
            desc = f"- {f['string']} ({f['name']}): {f['type']}"
            if f.get('relation'):
                desc += f" -> {f['relation']}"
            all_fields.append(desc)

        # 3. Construct Text
        text = f"Model: {model_name} ({model_tech_name})\nFields:\n" + "\n".join(all_fields)

        return text

    def _get_db_schema(self):
        """
        Dynamically extracts schema metadata for all available models.
        Returns a list of dictionaries describing models and their fields.
        """
        schema_data = []

        # Iterate over all models registered in the database (ir.model)
        # This ensures we get descriptions and module info from the database records
        model_records = self.env['ir.model'].search([])

        for record in model_records:
            model_name = record.model
            
            # Ensure the model is currently accessible in the environment registry
            if model_name not in self.env:
                continue

            current_model = self.env[model_name]
            
            # Skip Abstract and Transient models (Schema RAG should focus on persistent data)
            # This filters out technical mixins like ir.websocket, base_import.mapping, etc.
            if current_model._abstract or current_model._transient:
                continue
            
            # Extract fields dynamically from the model class
            # This includes custom fields (x_) and fields added by Studio
            model_fields = []
            for field_name, field_obj in current_model._fields.items():
                field_data = {
                    'name': field_name,
                    'type': field_obj.type,
                    'string': field_obj.string,
                    'relation': getattr(field_obj, 'comodel_name', None),
                }
                model_fields.append(field_data)

            # Extract methods dynamically
            # We look for methods starting with 'action_' or 'button_' which usually denote business logic
            model_methods = []
            # Use dir() to get all attributes, but we need to be careful about what we access
            # We only check attributes that exist on the class/recordset
            for attr_name in dir(current_model):
                if attr_name.startswith(('action_', 'button_')) and not attr_name.startswith('_'):
                    try:
                        attr = getattr(current_model, attr_name)
                        if callable(attr):
                            # Get the first line of the docstring as description
                            doc = (attr.__doc__ or "").strip().split('\n')[0]
                            model_methods.append({
                                'name': attr_name,
                                'description': doc or "No description available."
                            })
                    except:
                        continue

            # Build metadata dictionary
            model_metadata = {
                'model': model_name,
                'name': record.name,          # Human-readable description
                'module': record.modules,     # Comma-separated list of modules defining this model
                'fields': model_fields,
                'methods': model_methods      # Include the extracted methods
            }
            
            schema_data.append(model_metadata)

        return schema_data

    # ==========================
    # SCHEMA VECTOR STORE
    # ==========================

    def _get_schema_vector_store(self):
        """
        Returns the PGVector store specifically for Database Schema.
        Uses a separate collection 'ask_odoo_schema'.
        """
        if not AskOdooModel._schema_vector_store:
            # Ensure Embeddings are ready
            if not AskOdooModel._embeddings:
                AskOdooModel._embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

            connection = self._get_connection_string()
            
            AskOdooModel._schema_vector_store = PGVector(
                embeddings=AskOdooModel._embeddings,
                collection_name="ask_odoo_schema",
                connection=connection,
                use_jsonb=True,
            )
        return AskOdooModel._schema_vector_store

    def _get_schema_retriever(self):
        """
        Returns a retriever for the schema vector store.
        """
        vector_store = self._get_schema_vector_store()
        return vector_store.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": 5, 
                "fetch_k": 20,
                "lambda_mult": 0.7 # 0 = max diversity, 1 = pure similarity
            } 
        )

    @api.model
    def refresh_schema_index(self):
        """
        Re-indexes the entire Odoo database schema into the vector store.
        Generates a single canonical text corpus (snapshot), saves it as an attachment,
        then chunks and embeds from that corpus.
        """
        _logger.info("AskOdoo: Starting Schema Indexing...")
        
        # 1. Get Metadata
        schemas = self._get_db_schema()
        _logger.info(f"AskOdoo: Found {len(schemas)} models.")
        
        # 2. Generate Canonical Text Corpus
        # We join all model descriptions into one large text
        corpus_parts = []
        for s in schemas:
            text_content = self._schema_to_text(s)
            if text_content:
                corpus_parts.append(text_content)
            
        # Join with a separator that helps visualization handling
        full_corpus = ("\n\n" + "="*50 + "\n\n").join(corpus_parts)
        full_corpus += "\n\n" + "="*50 + "\n=== END OF SCHEMA SNAPSHOT ===\n"
        
        # 3. Save as Snapshot (ir.attachment) for Audit/Debug
        # This serves as the 'source of truth' for the embedding
        try:
            # Cleanup old snapshots to avoid clutter
            self.env['ir.attachment'].search([
                ('description', '=', 'Canonical schema snapshot for AskOdoo RAG')
            ]).unlink()
            _logger.info("AskOdoo: Removed old schema snapshots.")

            attachment_name = f"odoo_schema_snapshot_{fields.Datetime.now().isoformat().replace(':','-')}.txt"
            b64_data = base64.b64encode(full_corpus.encode('utf-8'))
            
            self.env['ir.attachment'].create({
                'name': attachment_name,
                'type': 'binary',
                'datas': b64_data,
                'mimetype': 'text/plain',
                'description': 'Canonical schema snapshot for AskOdoo RAG',
            })
            _logger.info(f"AskOdoo: Saved schema snapshot: {attachment_name}")
        except Exception as e:
            _logger.warning(f"AskOdoo: Failed to save schema snapshot attachment: {e}")

        # 4. Chunk and Embed FROM THE CORPUS
        # Split strictly by the 50-equals separator so that each chunk corresponds to exactly ONE model.
        # This prevents models from being broken up by text length.
        separator = "\n\n" + "="*50 + "\n\n"
        
        # Remove the EOF marker before splitting to keep the list clean
        eof_marker = "\n\n" + "="*50 + "\n=== END OF SCHEMA SNAPSHOT ===\n"
        clean_corpus = full_corpus.replace(eof_marker, "")
        
        # Split into exact model chunks
        texts = [t.strip() for t in clean_corpus.split(separator) if t.strip()]
        
        # Create Documents (each document is now guaranteed to be one full model definition)
        documents = [Document(page_content=t, metadata={'source': 'schema_snapshot'}) for t in texts]
            
        # 5. Add to Vector Store
        # 5. Add to Vector Store
        if documents:
            v_store = self._get_schema_vector_store()
            
            # Clear existing collection to prevent duplicates
            try:
                v_store.delete_collection()
                _logger.info("AskOdoo: Cleared existing schema collection.")
                
                # IMPORTANT: The collection is now gone from the DB.
                # We MUST re-initialize the PGVector store so it recreates the collection.
                # If we use the old 'v_store' object, it errors with "Collection not found".
                AskOdooModel._schema_vector_store = None
                v_store = self._get_schema_vector_store()
                
            except Exception as e:
                # If collection didn't exist, that's fine.
                # But we should ensure we have a valid store for adding.
                _logger.warning(f"AskOdoo: Warning during collection cleanup: {e}")
                # Optional: Force re-init if unsure, but usually only needed on success
                if "not found" in str(e).lower():
                     AskOdooModel._schema_vector_store = None
                     v_store = self._get_schema_vector_store()
            
            v_store.add_documents(documents)
            _logger.info(f"AskOdoo: Indexed {len(documents)} schema chunks from snapshot.")
            
        return True

