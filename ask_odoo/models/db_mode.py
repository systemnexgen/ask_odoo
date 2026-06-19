from odoo import models, api
from langchain_core.messages import SystemMessage, HumanMessage
import logging, json, ast, re

_logger = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────────────────────
MAX_RESULT_LIMIT = 50           # Hard cap regardless of what LLM requests
RESULT_PREVIEW_LENGTH = 300     # Max chars for result preview in history

# ── Intent Map ───────────────────────────────────────────────────────────────
# Each entry defines a supported data area with its Odoo model, trigger
# keywords, and the exact fields the LLM is allowed to use in queries.
# No vector search needed — the LLM picks the right intent from this list.

INTENT_MAP = [
    {
        "name": "Contacts",
        "model": "res.partner",
        "keywords": "contact, customer, supplier, vendor, partner",
        "fields": ["name", "email", "phone", "city", "country_id"],
    },
    {
        "name": "CRM Leads",
        "model": "crm.lead",
        "keywords": "lead, opportunity, crm, pipeline",
        "fields": ["name", "partner_id", "stage_id", "expected_revenue", "probability"],
    },
    {
        "name": "Sales Orders",
        "model": "sale.order",
        "keywords": "sale, order, quotation",
        "fields": ["name", "partner_id", "amount_total", "state", "date_order"],
    },
    {
        "name": "Employees",
        "model": "hr.employee",
        "keywords": "employee, staff, worker, hr",
        "fields": ["name", "department_id", "job_id", "work_email", "parent_id"],
    },
    {
        "name": "Products",
        "model": "product.template",
        "keywords": "product, item, goods",
        "fields": ["name", "list_price", "categ_id", "qty_available", "type"],
    },
    {
        "name": "Invoices",
        "model": "account.move",
        "keywords": "invoice, bill, payment due, receivable",
        "fields": ["name", "partner_id", "amount_total", "state", "invoice_date"],
    },
    {
        "name": "Payments",
        "model": "account.payment",
        "keywords": "payment, paid, received",
        "fields": ["partner_id", "amount", "date", "state"],
    },
    {
        "name": "Inventory/Stock",
        "model": "stock.quant",
        "keywords": "stock, inventory, warehouse, quantity",
        "fields": ["product_id", "location_id", "quantity"],
    },
    {
        "name": "Transfers",
        "model": "stock.picking",
        "keywords": "transfer, delivery, receipt, shipment",
        "fields": ["name", "partner_id", "state", "scheduled_date"],
    },
    {
        "name": "Departments",
        "model": "hr.department",
        "keywords": "department, team, division",
        "fields": ["name", "manager_id"],
    },
    {
        "name": "Calendar",
        "model": "calendar.event",
        "keywords": "meeting, event, appointment, calendar",
        "fields": ["name", "start", "stop", "partner_ids"],
    },
    {
        "name": "Companies",
        "model": "res.company",
        "keywords": "company, branch, entity",
        "fields": ["name", "email", "phone"],
    },
]


def _build_intent_context():
    """Format the intent map into a string the LLM can use as context."""
    lines = []
    for intent in INTENT_MAP:
        lines.append(
            f"- {intent['name']} | model: {intent['model']} | "
            f"keywords: {intent['keywords']} | "
            f"fields: {', '.join(intent['fields'])}"
        )
    return "\n".join(lines)


# Pre-build the context string (module-level, constant)
_INTENT_CONTEXT = _build_intent_context()


class AskOdooModel(models.Model):
    _inherit = 'ask.odoo.model'

    # ── Step 1: Prompt Refiner ───────────────────────────────────────────────

    def _refine_question(self, question, history=None):
        """Clean and standardize the user's Odoo ERP question.

        Fixes typos, expands abbreviations, and returns only the
        refined question string.
        """
        llm = self._get_llm()

        try:
            system_msg = SystemMessage(content=(
                "You are an Odoo ERP question refiner. "
                "Your task is to take the current user question and the conversation history, "
                "and produce a single, self-contained, and refined Odoo ERP query question. "
                "Make sure to resolve all pronouns and relative terms (e.g. 'them', 'it', 'show those', "
                "'filter by...') based on the previous messages. "
                "Standardize terms, fix typos, and expand abbreviations (e.g. SO → Sales Order, PO → Purchase Order, "
                "inv → invoice, emp → employee, dept → department). "
                "Return ONLY the refined question, nothing else."
            ))
            
            messages = [system_msg]
            if history:
                messages.extend(history)
            
            messages.append(HumanMessage(content=question))
            
            response = llm.invoke(messages)
            refined = response.content.strip()
            _logger.info(
                "\n=== [AskOdoo][DB MODE] Step 1 — Refined Question ===\n"
                "Original : %s\nRefined  : %s\n"
                "=====================================================",
                question, refined,
            )
            return refined
        except Exception as e:
            _logger.warning("AskOdoo: Prompt refiner failed, using original: %s", e)
            return question

    # ── Step 2: Query Generator ──────────────────────────────────────────────

    def _generate_query(self, refined_question):
        """Generate a structured ORM query specification from the refined question.

        Returns a dict with keys: intent, model, domain, fields, limit, order.
        Returns None if the LLM cannot map the question to an intent.
        """
        llm = self._get_llm()

        system_prompt = (
            "You are an Odoo ORM query generator. "
            "Given the question and available data areas below, respond with JSON only:\n"
            "{\n"
            '  "intent": "intent_name",\n'
            '  "model": "odoo.model.name",\n'
            '  "domain": [["field", "operator", "value"]],\n'
            '  "fields": ["field1", "field2"],\n'
            '  "limit": 20,\n'
            '  "order": "field desc"\n'
            "}\n\n"
            "Available data areas:\n"
            f"{_INTENT_CONTEXT}\n\n"
            "Rules:\n"
            "- Only use fields listed for that intent\n"
            "- Use valid Odoo domain syntax with proper operators "
            "(=, !=, >, <, >=, <=, like, ilike, in, not in)\n"
            "- IMPORTANT: The domain MUST be valid JSON (use lists of lists, NOT Python tuples). "
            "Example: [[\"state\", \"=\", \"draft\"]]\n"
            "- For Many2one fields (ending in _id), use the field name directly "
            "for domain filters (e.g., ('partner_id', 'ilike', 'name'))\n"
            "- Return JSON only, no explanation\n"
            "- If the question does not match any available data area, "
            'respond with: {"intent": "unsupported"}'
        )
        try:
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=refined_question),
            ]
            response = llm.invoke(messages)
            raw_text = response.content.strip()

            _logger.info(
                "\n=== [AskOdoo][DB MODE] Step 2 — Query Generator Raw ===\n"
                "%s\n"
                "=======================================================",
                raw_text,
            )

            # Extract JSON block using regex or basic string manipulation
            json_match = re.search(r'```(?:json)?\s*(.*?)\s*```', raw_text, re.DOTALL)
            if json_match:
                raw_text = json_match.group(1).strip()
            else:
                raw_text = raw_text.strip("`").strip()
                if raw_text.startswith("json"):
                    raw_text = raw_text[4:].strip()

           
            try:
                query_spec = json.loads(raw_text)
            except json.JSONDecodeError:
                # Fallback: Sometimes LLMs output Python dicts/tuples instead of pure JSON
                query_spec = ast.literal_eval(raw_text)

            # Check for unsupported intent
            if query_spec.get("intent") == "unsupported":
                _logger.info("AskOdoo: Question mapped to unsupported intent.")
                return None

            # Clamp limit to MAX_RESULT_LIMIT
            raw_limit = query_spec.get("limit", 20)
            if isinstance(raw_limit, int):
                query_spec["limit"] = min(raw_limit, MAX_RESULT_LIMIT)
            else:
                query_spec["limit"] = 20

            # Ensure domain is a list of lists/tuples
            domain = query_spec.get("domain", [])
            if not isinstance(domain, list):
                query_spec["domain"] = []
            else:
                # Convert any inner lists to tuples for Odoo ORM (Odoo domain format)
                formatted_domain = []
                for term in domain:
                    if isinstance(term, list):
                        formatted_domain.append(tuple(term))
                    else:
                        formatted_domain.append(term)
                query_spec["domain"] = formatted_domain

            _logger.info(
                "\n=== [AskOdoo][DB MODE] Step 2 — Parsed Query ===\n"
                "Intent: %s | Model: %s | Limit: %d\n"
                "Domain: %s\nFields: %s\nOrder: %s\n"
                "=================================================",
                query_spec.get("intent"), query_spec.get("model"),
                query_spec.get("limit", 20),
                query_spec.get("domain"), query_spec.get("fields"),
                query_spec.get("order"),
            )

            return query_spec

        except (json.JSONDecodeError, ValueError, SyntaxError) as e:
            _logger.error("AskOdoo: Failed to parse query generator output: %s", e)
            return None
        except Exception as e:
            _logger.exception("AskOdoo: Query generator failed: %s", e)
            return None

    # ── ORM Execution ────────────────────────────────────────────────────────

    def _execute_orm_query(self, query_spec):
        """Execute the ORM query from the spec and return results.

        Returns (results_list, error_string).
        On success: (list_of_dicts, None)
        On failure: (None, friendly_error_message)
        """
        model_name = query_spec.get("model", "")
        domain = query_spec.get("domain", [])
        field_list = query_spec.get("fields", [])
        limit = query_spec.get("limit", 20)
        order = query_spec.get("order", "")

        # Validate model exists
        if model_name not in self.env:
            available = [
                i["name"] for i in INTENT_MAP
                if i["model"] in self.env
            ]
            return None, (
                f"**❌ Model not found:** `{model_name}` does not exist in this Odoo instance.\n\n"
                f"Available data areas: {', '.join(available)}"
            )

        try:
            # Execute search_read — ORM enforces user permissions automatically
            results = self.env[model_name].search_read(
                domain,
                field_list,
                limit=limit,
                order=order or None,
            )

            # Clean up Many2one tuples: (id, name) → name
            clean_results = []
            for row in results:
                clean_row = {}
                for k, v in row.items():
                    if k == 'id' and len(row) > 1:
                        continue  # Skip internal ID if other fields are present
                    if isinstance(v, (list, tuple)) and len(v) == 2 and isinstance(v[0], int) and isinstance(v[1], str):
                        clean_row[k] = v[1]  # Extract display name
                    elif isinstance(v, (list, tuple)) and len(v) > 2:
                        clean_row[k] = f"{len(v)} records"
                    else:
                        clean_row[k] = v
                clean_results.append(clean_row)

            _logger.info(
                "AskOdoo: ORM query returned %d results from %s",
                len(clean_results), model_name,
            )

            return clean_results, None

        except Exception as e:
            _logger.exception("AskOdoo: ORM execution failed for %s", model_name)
            return None, (
                f"**❌ Query Error:** Could not retrieve data from `{model_name}`.\n\n"
                f"**Reason:** {str(e)}\n\n"
                "Please try rephrasing your question."
            )

    # ── Step 3: Response Formatter ───────────────────────────────────────────

    def _format_response(self, question, results, intent_name=""):
        """Format ORM results into a clear, concise answer using the LLM.

        Uses markdown tables for lists. Max 200 words.
        """
        llm = self._get_llm()

        # Truncate results for the LLM context if too large
        results_text = json.dumps(results[:MAX_RESULT_LIMIT], default=str, indent=2)
        if len(results_text) > 4000:
            results_text = results_text[:4000] + "\n... (truncated)"

        system_prompt = (
            "You are an Odoo data presentation assistant. "
            "Format the following database results into a clear, concise answer. "
            "Rules:\n"
            "- Use markdown tables for list data (more than 1 record)\n"
            "- For single values or counts, present as a brief sentence\n"
            "- Max 200 words\n"
            "- Be direct and professional\n"
            "- Do NOT add disclaimers or explain what you did\n"
            "- If results are empty, say 'No records found' clearly"
        )

        human_prompt = (
            f"Question: {question}\n"
            f"Data Area: {intent_name}\n"
            f"Number of results: {len(results)}\n"
            f"Results:\n{results_text}"
        )
        try:
            
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=human_prompt),
            ]
            response = llm.invoke(messages)
            formatted = response.content.strip()

            _logger.info(
                "\n=== [AskOdoo][DB MODE] Step 3 — Formatted Response ===\n"
                "%s\n"
                "======================================================",
                formatted[:500],
            )

            return formatted

        except Exception as e:
            _logger.warning("AskOdoo: Response formatter failed: %s", e)
            # Fallback: return raw results as a simple string
            if not results:
                return "No records found."
            return f"Found {len(results)} records:\n```json\n{results_text}\n```"

    # ── Main Entry Point ─────────────────────────────────────────────────────

    def process_message_db(self, message, conversation_id=None):
        """Handle database queries using the 3-step intent pipeline.

        This replaces the old RAG-based code generation approach with:
          Step 1 — Prompt Refiner (clean the question)
          Step 2 — Query Generator (map to ORM query via intent)
          Step 3 — Response Formatter (present results in markdown)

        Maintains the same function signature and return shape that the
        frontend (ai_chat.js) expects.
        """
        _logger.info("AskOdoo: Processing message in DATABASE mode (intent pipeline)")

        # 1. Handle Conversation
        conversation, user_msg = self._ensure_conversation(
            message, conversation_id, default_name='New DB Chat'
        )

        response_text = ""
        try:
            # Fetch conversation history excluding the current user message
            history = self._get_history(conversation.id, exclude_id=user_msg.id)

            # Step 1 — Refine the question using history context
            refined = self._refine_question(message, history=history)

            # Step 2 — Generate query specification
            query_spec = self._generate_query(refined)

            if query_spec is None:
                response_text = (
                    "**⚠️ Unsupported data area.**\n\n"
                    "I couldn't map your question to a supported data area. "
                    "Currently supported areas:\n\n"
                    + "\n".join(
                        f"- **{i['name']}** ({i['keywords']})"
                        for i in INTENT_MAP
                    )
                    + "\n\nPlease rephrase your question using one of these areas, "
                    "or contact your administrator to add support for more data areas."
                )
            else:
                # Execute ORM query
                results, error = self._execute_orm_query(query_spec)

                if error:
                    response_text = error
                elif not results:
                    response_text = (
                        f"**No records found** for your query in "
                        f"**{query_spec.get('intent', 'Unknown')}**.\n\n"
                        "Try broadening your search or adjusting the criteria."
                    )
                else:
                    # Step 3 — Format the response
                    response_text = self._format_response(
                        refined, results,
                        intent_name=query_spec.get("intent", ""),
                    )

        except Exception as e:
            _logger.exception("AskOdoo: Database Mode Pipeline Failed")
            response_text = (
                f"**❌ Error:** {str(e)}\n\n"
                "An unexpected error occurred. Please try again."
            )

        # Save AI message to conversation history
        self.env['ask.odoo.message'].create({
            'conversation_id': conversation.id,
            'type': 'ai',
            'content': response_text,
        })

        # Update conversation metadata (title, last_activity)
        self._update_conversation_metadata(conversation, message)

        return {
            'response': response_text,
            'conversation_id': conversation.id,
            'title': conversation.name,
            'execution_result': None,
            'has_code': False,
        }
