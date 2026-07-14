"""
MCP XML-RPC Client for ask_odoo Database Mode.

Connects to the existing mcp_server module's XML-RPC endpoints,
mirroring how external MCP clients like Claude Desktop connect.
"""

import os
import time
import logging
import xmlrpc.client

from dotenv import load_dotenv

load_dotenv()
_logger = logging.getLogger(__name__)

# ── Module-level singleton ────────────────────────────────────────────────────
_mcp_client_instance = None


def get_mcp_client():
    """Return a shared MCPClient singleton (lazy-initialized)."""
    global _mcp_client_instance
    if _mcp_client_instance is None:
        _mcp_client_instance = MCPClient()
    return _mcp_client_instance


class MCPClient:
    """
    XML-RPC client that authenticates against the mcp_server module
    and executes ORM operations through the MCP access-control layer.

    This is the same protocol used by Claude Desktop / OpenCode.
    """

    def __init__(self):
        self.url = os.getenv("MCP_URL", "http://localhost:8069")
        self.db = os.getenv("MCP_DB", "")
        self.username = os.getenv("MCP_USER", "admin")
        self.api_key = os.getenv("MCP_API_KEY", "")

        if not self.db:
            raise ValueError(
                "MCP_DB is not configured in .env. "
                "Please set MCP_DB to your Odoo database name."
            )
        if not self.api_key:
            raise ValueError(
                "MCP_API_KEY is not configured in .env. "
                "Please generate an API key in Odoo Settings > Users > API Keys."
            )

        # XML-RPC endpoints — standard Odoo endpoints
        # (same as what mcp-server-odoo package uses for Claude Desktop)
        self._common = xmlrpc.client.ServerProxy(
            f"{self.url}/xmlrpc/2/common", allow_none=True
        )
        self._object = xmlrpc.client.ServerProxy(
            f"{self.url}/xmlrpc/2/object", allow_none=True
        )

        # Authenticate once and cache the uid
        self._uid = None
        self._authenticate()

    # ── Authentication ────────────────────────────────────────────────────

    def _authenticate(self):
        """Authenticate via XML-RPC common endpoint (same as Claude Desktop)."""
        try:
            self._uid = self._common.authenticate(
                self.db, self.username, self.api_key, {}
            )
            if not self._uid:
                raise ConnectionError(
                    "MCP authentication failed. Check MCP_API_KEY and MCP_DB."
                )
            _logger.info(
                "MCP Client: Authenticated successfully (uid=%s, db=%s)",
                self._uid, self.db,
            )
        except Exception as e:
            _logger.error("MCP Client: Authentication failed: %s", e)
            raise

    # ── Low-level execute_kw ──────────────────────────────────────────────

    def _execute(self, model, method, args=None, kwargs=None, allowed_company_ids=None):
        """
        Call execute_kw through the MCP XML-RPC object endpoint.

        Returns (result, duration_ms).
        Raises on XML-RPC faults (access denied, rate limit, etc).
        """
        args = args or []
        kwargs = kwargs or {}

        if allowed_company_ids:
            context = kwargs.setdefault('context', {})
            context['allowed_company_ids'] = allowed_company_ids

        start = time.perf_counter()
        try:
            result = self._object.execute_kw(
                self.db, self._uid, self.api_key,
                model, method, args, kwargs,
            )
            duration_ms = int((time.perf_counter() - start) * 1000)
            return result, duration_ms
        except xmlrpc.client.Fault as fault:
            duration_ms = int((time.perf_counter() - start) * 1000)
            _logger.warning(
                "MCP Client: XML-RPC fault on %s.%s — [%s] %s (%dms)",
                model, method, fault.faultCode, fault.faultString, duration_ms,
            )
            raise
        except Exception as e:
            duration_ms = int((time.perf_counter() - start) * 1000)
            _logger.error(
                "MCP Client: Error on %s.%s — %s (%dms)",
                model, method, e, duration_ms,
            )
            raise

    # ── High-level tool methods (used by LangChain tools) ─────────────────

    def list_models(self):
        """
        List all MCP-enabled models by querying mcp.enabled.model.

        Returns: list of dicts [{model, name}, ...]
        """
        records, duration = self._execute(
            "mcp.enabled.model", "search_read",
            [[("active", "=", True)]],
            {"fields": ["model_name"], "limit": 100},
        )
        # Resolve display names from ir.model
        model_names = [r["model_name"] for r in records if r.get("model_name")]
        if not model_names:
            return [], duration

        ir_models, dur2 = self._execute(
            "ir.model", "search_read",
            [[("model", "in", model_names)]],
            {"fields": ["model", "name"]},
        )
        total_duration = duration + dur2

        result = [{"model": m["model"], "name": m["name"]} for m in ir_models]
        return result, total_duration

    def get_model_fields(self, model_name):
        """
        Get field definitions for a model via fields_get.

        Returns: dict of field_name -> {type, string, relation, ...}
        """
        result, duration = self._execute(
            model_name, "fields_get",
            [],
            {"attributes": ["string", "type", "relation", "required", "readonly"]},
        )
        return result, duration

    def search_records(self, model_name, domain=None, fields=None,
                       limit=20, order=None, allowed_company_ids=None):
        """
        Search and read records from an MCP-enabled model.

        Returns: list of record dicts
        """
        domain = domain or []
        kwargs = {"fields": fields or [], "limit": min(limit, 50)}
        if order:
            kwargs["order"] = order

        result, duration = self._execute(
            model_name, "search_read", [domain], kwargs,
            allowed_company_ids=allowed_company_ids
        )
        return result, duration

    def count_records(self, model_name, domain=None, allowed_company_ids=None):
        """Count records matching a domain."""
        domain = domain or []
        result, duration = self._execute(
            model_name, "search_count", [domain],
            allowed_company_ids=allowed_company_ids
        )
        return result, duration

    def read_records(self, model_name, record_ids, fields=None, allowed_company_ids=None):
        """Read specific records by ID."""
        kwargs = {}
        if fields:
            kwargs["fields"] = fields
        result, duration = self._execute(
            model_name, "read", [record_ids], kwargs,
            allowed_company_ids=allowed_company_ids
        )
        return result, duration

    def read_group(self, model_name, domain=None, fields=None,
                   groupby=None, limit=None, orderby=None, allowed_company_ids=None):
        """
        Perform a read_group (aggregate) query.

        Returns: list of grouped result dicts
        """
        domain = domain or []
        fields = fields or []
        groupby = groupby or []
        kwargs = {}
        if limit:
            kwargs["limit"] = limit
        if orderby:
            kwargs["orderby"] = orderby

        result, duration = self._execute(
            model_name, "read_group",
            [domain, fields, groupby],
            kwargs,
            allowed_company_ids=allowed_company_ids
        )
        return result, duration
