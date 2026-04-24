from odoo import models, fields, api
from odoo.tools.safe_eval import safe_eval
import re
import logging
import json
from datetime import timedelta
import pandas as pd
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from .prompts import get_db_mode_prompt

_logger = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────────────────────
MAX_CODE_RETRIES = 3            # Silent LLM retries for model validation
MAX_EXECUTION_RETRIES = 3       # Auto-retries for code execution errors
RESULT_PREVIEW_LENGTH = 300     # Max chars for result preview in history
_CODE_BLOCK_RE = re.compile(r"```python\n(.*?)\n```", re.DOTALL)


# ── Read-Only Proxies for Code Execution ─────────────────────────────────────
# These prevent LLM-generated code from calling write/unlink/sudo etc.

class _ReadOnlyModel:
    """Proxy that only exposes read operations on an Odoo model."""
    _ALLOWED_METHODS = frozenset({
        'search', 'search_read', 'search_count', 'read_group',
        'browse', 'read', 'mapped', 'filtered', 'sorted',
        'ids', 'name_get', 'fields_get',
    })
    _ALLOWED_ATTRS = frozenset({
        '_name', '_description', 'id', 'ids', 'display_name',
        'env',  # needed for traversing relations like record.partner_id.name
    })

    def __init__(self, model):
        object.__setattr__(self, '_model', model)

    def __getattr__(self, name):
        if name in self._ALLOWED_METHODS or name in self._ALLOWED_ATTRS:
            attr = getattr(self._model, name)
            # Wrap returned recordsets so chained access is also read-only
            if hasattr(attr, '_name') and hasattr(attr, 'mapped'):
                return _ReadOnlyModel(attr)
            return attr
        raise PermissionError(
            f"Operation '{name}' is not allowed in read-only mode. "
            f"Only these operations are permitted: {', '.join(sorted(self._ALLOWED_METHODS))}"
        )

    def __iter__(self):
        return iter(self._model)

    def __len__(self):
        return len(self._model)

    def __bool__(self):
        return bool(self._model)

    def __repr__(self):
        return f"ReadOnly({self._model!r})"


class _ReadOnlyEnv:
    """Proxy around ``self.env`` that returns ReadOnlyModel wrappers."""

    def __init__(self, env):
        object.__setattr__(self, '_env', env)

    def __getitem__(self, model_name):
        return _ReadOnlyModel(self._env[model_name])

    def __contains__(self, item):
        return item in self._env

    @property
    def user(self):
        return self._env.user

    @property
    def company(self):
        return self._env.company


class _ReadOnlySelf:
    """Proxy around ``self`` that only exposes env (read-only)."""

    def __init__(self, model_instance):
        object.__setattr__(self, '_instance', model_instance)

    @property
    def env(self):
        return _ReadOnlyEnv(self._instance.env)



class AskOdooModel(models.Model):
    _inherit = 'ask.odoo.model'

    # ── Helpers ──────────────────────────────────────────────────────────────

    @staticmethod
    def _extract_code_block(text):
        """Extract a Python code block from LLM response text.

        Returns (has_code: bool, code: str | None).
        """
        match = _CODE_BLOCK_RE.search(text)
        if match:
            return True, match.group(1).strip()
        return False, None

    def process_message_db(self, message, conversation_id=None):
        """
        Handles database / schema related queries using Schema RAG.
        This does NOT execute CRUD yet — only explains models, fields, relations.
        """
        chat_mode = "database"
        _logger.info("AskOdoo: Processing message in DATABASE mode")

        # 1. Handle Conversation
        conversation, user_msg = self._ensure_conversation(
            message, conversation_id, default_name='New DB Chat'
        )

        context_text = "No context available."
        try:
            # 3. Components
            llm = self._get_llm()
            history = self._get_history(conversation.id, exclude_id=user_msg.id)

            # 4. Retrieve & Trim Schema Context (Two-Phase)
            # Phase 1: vector similarity → top-5 models
            # Phase 2: field trimming per model (keyword + relational + generic rules)
            context_text = self._get_relevant_schema(message)

            _logger.info(
                "\n=== [AskOdoo][DB MODE] Schema Context ===\n%s\n=========================================",
                context_text,
            )

            # 5. Database-Specific Prompt
            embeddings = self._get_embeddings()
            prompt = get_db_mode_prompt(message, embeddings, top_k=2)

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
            MAX_LLM_RETRIES = MAX_CODE_RETRIES
            messages_for_llm = prompt_messages  # start with the original prompt
            response_text = ""

            for attempt in range(MAX_LLM_RETRIES + 1):
                ai_message = llm.invoke(messages_for_llm)
                response_text = ai_message.content

                _logger.info(
                    f"\n=== [AskOdoo][DB MODE] LLM Response (attempt {attempt + 1}) ===\n"
                    f"{response_text}\n====================================="
                )

                # Extract candidate code
                has_candidate, candidate_code = self._extract_code_block(response_text)
                if not has_candidate:
                    # No code block — nothing to validate, return as-is
                    break

                is_valid, val_error = self._validate_code_models(candidate_code)

                if is_valid:
                    _logger.info(f"AskOdoo: Code passed model validation on attempt {attempt + 1}.")
                    break

                if attempt < MAX_LLM_RETRIES:
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
                        "AskOdoo: Code still invalid after %d retries. "
                        "Returning last response to user.", MAX_CODE_RETRIES,
                    )

        except Exception as e:
            _logger.exception("Database Mode Failed")
            response_text = f"Error: {str(e)}"

        # 7. Extract Code if present
        execution_result = None
        has_code, action_code = self._extract_code_block(response_text)

        # 8. Save AI Message (Hidden if code is executed so UI doesn't show explanation)
        self.env['ask.odoo.message'].create({
            'conversation_id': conversation.id,
            'type': 'ai',
            'content': ("[HIDDEN]" if has_code else "") + response_text,
        })

        # 9. Update Conversation
        self._update_conversation_metadata(conversation, message)

        if has_code:
            execution_result = self.execute_confirmed_code(action_code, conversation.id, schema_context=context_text)

        return {
            'response': response_text,
            'conversation_id': conversation.id,
            'title': conversation.name,
            'execution_result': execution_result,
            'has_code': has_code,
        }

    # ── Code Execution Pipeline ──────────────────────────────────────────────

    def _execute_code_safely(self, code):
        """Run code inside a savepoint via safe_eval.

        The execution context exposes a read-only proxy for ``self.env``
        so that LLM-generated code cannot accidentally (or maliciously)
        call write/unlink/sudo.

        Returns (result, chart_config) from the execution context.
        Raises on any execution error.
        """
        readonly_env = _ReadOnlyEnv(self.env)
        local_context = {
            'self': _ReadOnlySelf(self),
            'env': readonly_env,
            'result': None,
            'chart_config': None,
            'time': fields.Datetime.now,
            'datetime': fields.Datetime,
            'date': fields.Date,
            'timedelta': timedelta,
        }
        with self.env.cr.savepoint():
            safe_eval(code, local_context, mode="exec", nocopy=True)
        return local_context.get('result'), local_context.get('chart_config')

    @staticmethod
    def _postprocess_result(result):
        """Convert RecordSets, view objects, and generators to serialisable types."""
        # RecordSet → list of display names
        if result is not None and hasattr(result, '_name') and hasattr(result, 'mapped'):
            if not result:
                return "No records found."
            try:
                return result.mapped('display_name')
            except (KeyError, AttributeError) as e:
                _logger.debug("Could not map display_name: %s", e)
                return str(result)

        if result is None:
            return result

        # dict_keys / dict_values / sets → list
        result_type = type(result).__name__
        if result_type in ('dict_keys', 'dict_values', 'dict_items', 'set', 'frozenset'):
            return list(result)

        # Other non-string iterables → list
        if hasattr(result, '__iter__') and not isinstance(result, (str, bytes, list, tuple)):
            try:
                return list(result)
            except (TypeError, ValueError) as e:
                _logger.debug("Could not convert iterable to list: %s", e)

        return result

    def _result_to_html(self, result, chart_config):
        """Convert list/tuple results to an HTML table + optional chart data.

        Returns (html_or_string, chart_data).
        """
        chart_data = None
        if not isinstance(result, (list, tuple)):
            return (str(result) if result is not None else "Action Executed Successfully"), chart_data

        try:
            if not result:
                return "No records found.", chart_data

            # Cleanup Odoo-specific dictionary fields before Pandas conversion
            if isinstance(result, list) and result and isinstance(result[0], dict):
                clean_result = []
                for row in result:
                    clean_row = {}
                    for k, v in row.items():
                        if k == '__domain':
                            continue
                        # If value is a M2O tuple (id, name), extract the name
                        if isinstance(v, (list, tuple)) and len(v) == 2 and isinstance(v[0], int) and isinstance(v[1], str):
                            clean_row[k] = v[1]
                        else:
                            clean_row[k] = v
                    clean_result.append(clean_row)
                result = clean_result

            df = pd.DataFrame(result)

            # List-of-lists: promote first row to header if all-strings
            if isinstance(result, list) and result and isinstance(result[0], list):
                first_row = result[0]
                if first_row and all(isinstance(x, str) for x in first_row):
                    df = pd.DataFrame(result[1:], columns=first_row)

            # Rename default "0" column for simple lists
            if len(df.columns) == 1 and str(df.columns[0]) == '0':
                df.columns = ['Result']

            # Chart generation (only when explicitly requested)
            if chart_config:
                chart_data = self._generate_explicit_chart(df, chart_config)

            html = df.to_html(index=False, border=0, classes='dataframe')
            return html, chart_data

        except Exception:
            return str(result), chart_data

    def _save_execution_to_history(self, conversation_id, result, chart_data):
        """Persist a success message (with optional HTML and chart) to conversation history."""
        if not conversation_id:
            return
        try:
            preview = result if isinstance(result, str) else str(result)
            history_content = (
                f" {preview[:RESULT_PREVIEW_LENGTH]}"
                f"{'...' if len(preview) > RESULT_PREVIEW_LENGTH else ''}"
            )
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
            _logger.warning("AskOdoo: Could not save success result to history: %s", hist_e)

    def _save_error_to_history(self, conversation_id, content):
        """Persist a hidden error/retry message to conversation history."""
        if not conversation_id:
            return
        try:
            self.env['ask.odoo.message'].create({
                'conversation_id': conversation_id,
                'type': 'ai',
                'content': "[HIDDEN]" + content,
            })
        except Exception as hist_e:
            _logger.warning("AskOdoo: Could not save error to history: %s", hist_e)

    @api.model
    def execute_confirmed_code(self, code, conversation_id, schema_context=None):
        """Execute confirmed Python code with automated retry on failure.

        On success: post-processes the result, converts to HTML, saves to
        conversation history.  On failure: asks the LLM to fix the code and
        retries up to MAX_EXECUTION_RETRIES times.
        """
        for attempt in range(MAX_EXECUTION_RETRIES):
            try:
                raw_result, chart_config = self._execute_code_safely(code)
                result = self._postprocess_result(raw_result)
                result, chart_data = self._result_to_html(result, chart_config)

                self._save_execution_to_history(conversation_id, result, chart_data)

                return {
                    'status': 'success',
                    'result': result,
                    'chart_data': chart_data,
                }

            except Exception as e:
                _logger.exception("Execution Error")

                friendly_error = self._explain_error_to_user(
                    code, str(e), conversation_id, schema_context,
                )
                _, retry_code = self._extract_code_block(friendly_error)

                # Reject identical fixes
                if retry_code and retry_code.strip() == code.strip():
                    _logger.warning("AskOdoo: Proposed fix is identical to failed code. Rejecting.")
                    retry_code = None
                    friendly_error = (
                        f"**❌ Error:** {str(e)}\n\n"
                        "I was unable to generate a different fix. "
                        "Please try rephrasing your question or providing more details."
                    )

                if retry_code:
                    self._save_error_to_history(conversation_id, (
                        f"❌ Execution Error:\n```python\n{code}\n```\nError: {str(e)}\n"
                        f"Proposed fix code:\n```python\n{retry_code}\n```"
                    ))
                    _logger.info("AskOdoo: Auto-retrying fixed code. Depth: %d", attempt + 1)
                    code = retry_code
                else:
                    self._save_error_to_history(conversation_id, (
                        f"❌ Execution Error for the previously proposed code:\n"
                        f"Code attempted:\n```python\n{code}\n```\n"
                        f"Error: {str(e)}\n"
                    ))
                    return {
                        'status': 'error',
                        'message': friendly_error,
                        'debug_message': str(e),
                        'retry_code': None,
                    }

        # Guard: Max retries reached
        _logger.warning("AskOdoo: Max auto-retries reached. Bailing out.")
        bail_message = (
            "**❌ Unable to automatically retrieve data.**\n\n"
            "I tried fixing the query multiple times but could not find the correct fields. "
            "Please try rephrasing your question or providing more details."
        )
        self._save_error_to_history(conversation_id, bail_message)
        return {
            'status': 'error',
            'message': bail_message,
            'debug_message': "Max retries reached without success.",
            'retry_code': None,
        }

    @api.model
    def _explain_error_to_user(self, code, error_message, conversation_id=None, schema_context=None):
        """
        Uses the LLM to produce a structured 3-part error response.
        Now includes conversation history and schema context so the LLM can see previous
        failed attempts and avoid hallucinating fields on retries.
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
                "You MUST use ONLY the exact models and fields explicitly defined in the schema below. "
                "Do not guess or assume fields exist if they are not listed.\n\n"
                "SCHEMA:\n{context}\n"
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
                context=schema_context or "No schema context provided."
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

