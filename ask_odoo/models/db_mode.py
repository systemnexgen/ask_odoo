"""
Database Mode — MCP Client Agent.

Rewrites the old rigid intent-map pipeline with an LLM agent loop
that uses the existing mcp_server as its data backend via XML-RPC.
This is the same protocol Claude Desktop / OpenCode use.

Each tool call is tracked with name, args, result preview, duration,
and timestamp — sent to the frontend for a "deep research" UI.
"""

from odoo import models, api
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage
import logging
import json
import time
from datetime import datetime

from .mcp_client import get_mcp_client

_logger = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────────────────────
MAX_AGENT_ITERATIONS = 30
MAX_RESULT_PREVIEW = 500  # chars in tool result preview for LLM context


# ── Tool Definitions (JSON Schema for LLM tool-calling) ──────────────────────
# These are passed to ChatGroq.bind_tools() so the LLM can invoke them.

TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "list_models",
            "description": (
                "List all Odoo models that are available for querying via MCP. "
                "Call this FIRST to discover which models you can access. "
                "Returns a list of {model, name} objects."
            ),
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_model_fields",
            "description": (
                "Get the field definitions of an Odoo model. "
                "Returns field names with their types, labels, and relations. "
                "Use this to understand the schema before building search queries."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "model_name": {
                        "type": "string",
                        "description": "Technical model name, e.g. 'res.partner', 'sale.order'",
                    },
                },
                "required": ["model_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_records",
            "description": (
                "Search and retrieve records from an Odoo model using domain filters. "
                "Domain is a JSON-serialized list of conditions like '[[\"field\", \"operator\", \"value\"]]'. "
                "Operators: =, !=, >, <, >=, <=, like, ilike, in, not in, child_of. "
                "Use ilike for case-insensitive text search. "
                "For Many2one fields (ending in _id), filter by name using "
                "('field_id.name', 'ilike', 'value') or ('field_id', '=', id). "
                "Returns a list of record dictionaries."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "model_name": {
                        "type": "string",
                        "description": "Technical model name, e.g. 'sale.order'",
                    },
                    "domain": {
                        "type": "string",
                        "description": "JSON-serialized Odoo domain filter list, e.g. '[[\"state\", \"=\", \"sale\"]]. Use \"|\" or \"&\" as logical operators if needed.'",
                    },
                    "fields": {
                        "type": "array",
                        "description": "List of field names to return, e.g. ['name', 'amount_total']",
                        "items": {"type": "string"},
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of records to return (default 20, max 50)",
                    },
                    "order": {
                        "type": "string",
                        "description": "Sort order, e.g. 'amount_total desc' or 'name asc'",
                    },
                },
                "required": ["model_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "count_records",
            "description": (
                "Count how many records match a domain filter in an Odoo model. "
                "Use this for aggregation questions like 'how many invoices are overdue?'"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "model_name": {
                        "type": "string",
                        "description": "Technical model name",
                    },
                    "domain": {
                        "type": "string",
                        "description": "JSON-serialized Odoo domain filter list, e.g. '[[\"state\", \"=\", \"sale\"]]'",
                    },
                },
                "required": ["model_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_group",
            "description": (
                "Perform aggregated/grouped queries on an Odoo model. "
                "Similar to SQL GROUP BY. Use for summaries like 'total sales by customer' "
                "or 'average order value by month'. "
                "Fields should include the measure (e.g. 'amount_total') and groupby field. "
                "Supported aggregates: field_name:sum, field_name:avg, field_name:max, etc."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "model_name": {
                        "type": "string",
                        "description": "Technical model name",
                    },
                    "domain": {
                        "type": "string",
                        "description": "JSON-serialized Odoo domain filter list, e.g. '[[\"state\", \"=\", \"sale\"]]'",
                    },
                    "fields": {
                        "type": "array",
                        "description": "Fields to aggregate, e.g. ['amount_total', 'partner_id']",
                        "items": {"type": "string"},
                    },
                    "groupby": {
                        "type": "array",
                        "description": "Fields to group by, e.g. ['partner_id', 'state']",
                        "items": {"type": "string"},
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Max groups to return",
                    },
                    "orderby": {
                        "type": "string",
                        "description": "Sort order for groups, e.g. 'amount_total desc'",
                    },
                },
                "required": ["model_name", "fields", "groupby"],
            },
        },
    },
]


# ── Tool Execution Engine ────────────────────────────────────────────────────

def _parse_domain(domain_arg):
    """
    Parse domain argument which can be a JSON string or a python list/tuple.
    """
    if not domain_arg:
        return []
    if isinstance(domain_arg, str):
        try:
            return json.loads(domain_arg)
        except Exception as e:
            _logger.warning("Failed to parse JSON domain string: %s. Using empty list.", e)
            return []
    return domain_arg


def _execute_tool(tool_name, tool_args, allowed_company_ids=None):
    """
    Execute a single MCP tool call and return (result, duration_ms, error).

    All calls go through the MCP XML-RPC client, which means they pass
    through the mcp_server's access control, rate limiting, and logging.
    """
    client = get_mcp_client()

    try:
        if tool_name == "list_models":
            result, duration = client.list_models()
            return result, duration, None

        elif tool_name == "get_model_fields":
            model_name = tool_args.get("model_name", "")
            raw_fields, duration = client.get_model_fields(model_name)
            # Simplify the fields dict for LLM context (reduce token usage)
            simplified = {}
            for fname, fdata in raw_fields.items():
                entry = {"type": fdata.get("type", ""), "label": fdata.get("string", "")}
                if fdata.get("relation"):
                    entry["relation"] = fdata["relation"]
                simplified[fname] = entry
            return simplified, duration, None

        elif tool_name == "search_records":
            model_name = tool_args.get("model_name", "")
            domain = _parse_domain(tool_args.get("domain"))
            fields = tool_args.get("fields", [])
            limit = tool_args.get("limit", 20)
            order = tool_args.get("order", "")
            result, duration = client.search_records(
                model_name, domain, fields, limit, order,
                allowed_company_ids=allowed_company_ids,
            )
            # Clean up Many2one tuples: [id, "name"] → "name"
            clean = []
            for row in (result or []):
                clean_row = {}
                for k, v in row.items():
                    if k == "id":
                        clean_row[k] = v
                    elif isinstance(v, list) and len(v) == 2 and isinstance(v[0], int):
                        clean_row[k] = v[1]  # Display name
                    elif isinstance(v, list) and len(v) > 2:
                        clean_row[k] = f"{len(v)} records"
                    elif v is False:
                        clean_row[k] = None
                    else:
                        clean_row[k] = v
                clean.append(clean_row)
            return clean, duration, None

        elif tool_name == "count_records":
            model_name = tool_args.get("model_name", "")
            domain = _parse_domain(tool_args.get("domain"))
            result, duration = client.count_records(
                model_name, domain, allowed_company_ids=allowed_company_ids
            )
            return {"count": result}, duration, None

        elif tool_name == "read_group":
            model_name = tool_args.get("model_name", "")
            domain = _parse_domain(tool_args.get("domain"))
            fields = tool_args.get("fields", [])
            groupby = tool_args.get("groupby", [])
            limit = tool_args.get("limit")
            orderby = tool_args.get("orderby")
            result, duration = client.read_group(
                model_name, domain, fields, groupby, limit, orderby,
                allowed_company_ids=allowed_company_ids,
            )
            # Clean up Many2one tuples in grouped results
            clean = []
            for row in (result or []):
                clean_row = {}
                for k, v in row.items():
                    if k == "__domain":
                        continue  # skip internal domain
                    elif isinstance(v, list) and len(v) == 2 and isinstance(v[0], int):
                        clean_row[k] = v[1]
                    elif v is False:
                        clean_row[k] = None
                    else:
                        clean_row[k] = v
                clean.append(clean_row)
            return clean, duration, None

        else:
            return None, 0, f"Unknown tool: {tool_name}"

    except Exception as e:
        return None, 0, str(e)


# ── System Prompt ────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """\
You are an Odoo ERP data assistant connected to a live Odoo database via MCP tools.

## Your Workflow
1. **Discover**: Call `list_models` to see what models are available.
2. **Inspect**: Call `get_model_fields` to understand the schema of relevant models.
3. **Query**: Use `search_records`, `count_records`, or `read_group` to fetch data.
4. **Synthesize**: Format the results into a clear, professional answer.

## Rules
- ALWAYS call `list_models` first if you don't know what's available.
- ALWAYS call `get_model_fields` before `search_records` to verify field names exist.
- Use `ilike` for case-insensitive text matching.
- For Many2one fields (e.g. partner_id, product_id), filter by subfield:
  `['partner_id.name', 'ilike', 'Azure']` or `['partner_id', '=', 42]`.
- For date filters, use ISO format: `['date_order', '>=', '2025-01-01']`.
- Use `read_group` for aggregate queries (sums, counts, averages by group).
- Limit results to 50 max. Default to 20.
- If a tool call fails, read the error and adjust your query.
- Do NOT invent data. Only present what the tools return.
- Format lists as markdown tables. Format single values as brief sentences.
- Be concise and professional. Max 300 words in your final answer.
"""


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

    # ── Main Entry Point ─────────────────────────────────────────────────

    def process_message_db(self, message, conversation_id=None):
        """
        Handle database queries using the MCP Client Agent loop.

        The LLM decides which tools to call, executes them via the
        mcp_server XML-RPC endpoints, and synthesizes a final answer.
        Each tool call is tracked for the frontend's deep-research UI.
        """
        allowed_company_ids = self.env.context.get('allowed_company_ids') or self.env.companies.ids
        _logger.info("AskOdoo: Processing message in DATABASE mode (MCP Agent). Allowed companies: %s", allowed_company_ids)

        # 1. Handle Conversation
        conversation, user_msg = self._ensure_conversation(
            message, conversation_id, default_name='New DB Chat'
        )

        response_text = ""
        tool_steps = []

        try:
            # Fetch conversation history
            history = self._get_history(conversation.id, exclude_id=user_msg.id)

            # Build the LLM with tools bound
            llm = self._get_llm()
            llm_with_tools = llm.bind_tools(TOOL_SCHEMAS)

            # Build initial message list
            messages = [SystemMessage(content=SYSTEM_PROMPT)]
            if history:
                messages.extend(history)
            messages.append(HumanMessage(content=message))

            # ── Agent Loop ────────────────────────────────────────────
            for iteration in range(MAX_AGENT_ITERATIONS):
                _logger.info(
                    "AskOdoo MCP Agent: Iteration %d/%d",
                    iteration + 1, MAX_AGENT_ITERATIONS,
                )

                # Call LLM
                ai_response = llm_with_tools.invoke(messages)
                messages.append(ai_response)

                # Check if LLM wants to call tools
                tool_calls = ai_response.tool_calls
                if not tool_calls:
                    # No more tool calls — LLM has produced a final answer
                    response_text = _extract_text_content(ai_response.content)
                    _logger.info(
                        "AskOdoo MCP Agent: Final answer at iteration %d",
                        iteration + 1,
                    )
                    break

                # Execute each tool call
                for tc in tool_calls:
                    tool_name = tc["name"]
                    tool_args = tc.get("args", {})
                    tool_id = tc.get("id", f"call_{iteration}_{tool_name}")
                    timestamp = datetime.now().strftime("%H:%M:%S")

                    _logger.info(
                        "AskOdoo MCP Agent: Calling tool '%s' with args: %s",
                        tool_name, json.dumps(tool_args, default=str)[:300],
                    )

                    # Execute through MCP XML-RPC
                    result, duration_ms, error = _execute_tool(
                        tool_name, tool_args, allowed_company_ids=allowed_company_ids
                    )

                    if error:
                        tool_result_str = json.dumps({"error": error}, default=str)
                        step_status = "error"
                    else:
                        tool_result_str = json.dumps(result, default=str)
                        step_status = "success"

                    # Truncate for LLM context to avoid token overflow
                    llm_result = tool_result_str
                    if len(llm_result) > 4000:
                        llm_result = llm_result[:4000] + "\n... (truncated)"

                    # Build result preview for the UI (shorter)
                    result_preview = tool_result_str[:MAX_RESULT_PREVIEW]
                    if len(tool_result_str) > MAX_RESULT_PREVIEW:
                        result_preview += "..."

                    # Record step for frontend tracking
                    step = {
                        "tool": tool_name,
                        "args": tool_args,
                        "result_preview": result_preview,
                        "duration_ms": duration_ms,
                        "timestamp": timestamp,
                        "status": step_status,
                        "iteration": iteration + 1,
                    }
                    tool_steps.append(step)

                    _logger.info(
                        "AskOdoo MCP Agent: Tool '%s' → %s (%dms)",
                        tool_name, step_status, duration_ms,
                    )

                    # Feed result back to the LLM as a ToolMessage
                    messages.append(
                        ToolMessage(content=llm_result, tool_call_id=tool_id)
                    )

            else:
                # Exhausted iterations
                response_text = (
                    "I reached the maximum number of steps while analyzing your query. "
                    "Here's what I found so far:\n\n" + _extract_text_content(ai_response.content)
                )

        except Exception as e:
            _logger.exception("AskOdoo: MCP Agent Pipeline Failed")
            error_msg = str(e)
            # Provide a user-friendly error for common issues
            if "authenticate" in error_msg.lower() or "api_key" in error_msg.lower():
                response_text = (
                    "**❌ MCP Connection Error**\n\n"
                    "Could not connect to the MCP server. Please verify:\n"
                    "- `MCP_URL`, `MCP_DB`, and `MCP_API_KEY` are set in the `.env` file\n"
                    "- The MCP Server module is installed and enabled\n"
                    "- An API key has been generated for your user"
                )
            else:
                response_text = (
                    f"**❌ Error:** {error_msg}\n\n"
                    "An unexpected error occurred. Please try again."
                )

        # Save AI message
        self.env['ask.odoo.message'].create({
            'conversation_id': conversation.id,
            'type': 'ai',
            'content': response_text,
        })

        # Update conversation metadata
        self._update_conversation_metadata(conversation, message)

        return {
            'response': response_text,
            'conversation_id': conversation.id,
            'title': conversation.name,
            'execution_result': None,
            'has_code': False,
            'tool_steps': tool_steps,
        }
