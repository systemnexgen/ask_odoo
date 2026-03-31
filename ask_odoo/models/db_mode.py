from odoo import models, fields, api
from odoo.tools.safe_eval import safe_eval
import re
import logging
import json
import pandas as pd
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

_logger = logging.getLogger(__name__)

class AskOdooModel(models.Model):
    _inherit = 'ask.odoo.model'

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
                "Odoo DB Assistant. READ operations ONLY. Create, Update, and Delete operations are STRICTLY PROHIBITED. Use ONLY the schema below.\n\n"
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

        # 7. Extract Code if present
        execution_result = None
        has_code = False
        code_match = re.search(r"```python\n(.*?)\n```", response_text, re.DOTALL)
        if code_match:
            has_code = True
            action_code = code_match.group(1).strip()

        # 8. Save AI Message (Hidden if code is executed so UI doesn't show explanation)
        self.env['ask.odoo.message'].create({
            'conversation_id': conversation.id,
            'type': 'ai',
            'content': ("[HIDDEN]" if has_code else "") + response_text,
        })

        # 9. Update Conversation
        conversation.write({'last_activity': fields.Datetime.now()})

        if len(conversation.message_ids) <= 2:
            conversation.write({
                'name': message[:30] + "..." if len(message) > 30 else message
            })

        if has_code:
            execution_result = self.execute_confirmed_code(action_code, conversation.id)

        return {
            'response': response_text,
            'conversation_id': conversation.id,
            'title': conversation.name,
            'execution_result': execution_result,
            'has_code': has_code,
        }

    @api.model
    def execute_confirmed_code(self, code, conversation_id, retry_depth=0):
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

            # ── Guard 1: Retry Depth ───────────────────────────
            MAX_AUTO_RETRIES = 3
            if retry_depth >= MAX_AUTO_RETRIES:
                _logger.warning("AskOdoo: Max auto-retries reached. Bailing out.")
                bail_message = (
                    "**❌ Unable to automatically retrieve data.**\n\n"
                    "I tried fixing the query multiple times but could not find the correct fields. "
                    "Please try rephrasing your question or providing more details."
                )
                if conversation_id:
                    self.env['ask.odoo.message'].create({
                        'conversation_id': conversation_id,
                        'type': 'ai',
                        'content': "[HIDDEN]" + bail_message,
                    })
                return {
                    'status': 'error',
                    'message': bail_message,
                    'debug_message': str(e),
                    'retry_code': None,
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

            # Auto-execute retry code if valid
            if retry_code:
                if conversation_id:
                    try:
                        history_content = (
                            f"❌ Execution Error:\n```python\n{code}\n```\nError: {str(e)}\n"
                            f"Proposed fix code:\n```python\n{retry_code}\n```"
                        )
                        self.env['ask.odoo.message'].create({
                            'conversation_id': conversation_id,
                            'type': 'ai',
                            'content': "[HIDDEN]" + history_content,
                        })
                    except Exception as hist_e:
                        _logger.warning(f"AskOdoo: Could not save error to history: {hist_e}")
                
                _logger.info(f"AskOdoo: Auto-retrying fixed code. Depth: {retry_depth + 1}")
                return self.execute_confirmed_code(retry_code, conversation_id, retry_depth=retry_depth + 1)

            # If no retry code was found, save final error and return
            if conversation_id:
                try:
                    history_content = (
                        f"❌ Execution Error for the previously proposed code:\n"
                        f"Code attempted:\n```python\n{code}\n```\n"
                        f"Error: {str(e)}\n"
                    )
                    self.env['ask.odoo.message'].create({
                        'conversation_id': conversation_id,
                        'type': 'ai',
                        'content': "[HIDDEN]" + history_content,
                    })
                except Exception as hist_e:
                    _logger.warning(f"AskOdoo: Could not save error to history: {hist_e}")

            return {
                'status': 'error',
                'message': friendly_error,
                'debug_message': str(e),
                'retry_code': None,
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
            return f"**❌ Error:** {error_message}\n\n**Please rephrase your query or provide more details.**"

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

