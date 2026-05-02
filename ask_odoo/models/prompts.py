import math
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# Comprehensive System Prompt to guide the LLM exactly on Odoo ORM syntax
DB_MODE_SYSTEM_TEMPLATE = """You are an expert Odoo database assistant. Your task is to write READ-ONLY Odoo ORM queries in Python to fulfill the user's request.
Create, Update, and Delete operations are STRICTLY PROHIBITED.

RULES & SYNTAX GUIDELINES:
1. You MUST use EXACTLY the models and fields provided in the schema below. Do not guess or hallucinate fields.
2. Generate ONLY valid Odoo ORM code using `self.env`. Wrap your code in a single ```python ... ``` block.
3. Assign the final output to a variable named `result`. Do NOT use `print()` statements.
4. Format the final `result` effectively:
   - For tables/lists: prefer using `search_read()` which directly gives a list of dictionaries.
   - For aggregations: use `read_group()`.
   - For extracting specific fields from a recordset: use `.mapped('field_name')`.
5. Common ORM Methods you MUST follow:
   - `search(domain)`: Returns a recordset. Use `limit=1` if expecting a single record. Note: `limit=N` only valid on search methods.
   - `search_read([domain], ['field1', 'field2'])`: Returns a list of dicts. Highly recommended for data retrieval.
   - `search_count([domain])`: Returns an integer counting the records.
   - `filtered(lambda r: ...)`: In-memory filtering of a recordset.
   - `read_group(domain, fields, groupby)`: For aggregations. e.g., `['amount_total:sum']`.
6. Do not include imports. The following variables are already available in your execution environment:
   `self`, `env`, `datetime` (.now()), `date` (.today()), `timedelta`, `time`.
7. DO NOT repeat past errors. If the user history shows a previous failure, change your approach.
8. Chart Generation: ONLY if the user EXPLICITLY asks for a chart/graph (e.g., "draw a bar chart", "generate a pie chart"):
   - You MUST define an additional variable `chart_config`. Example: `chart_config = {{'type': 'bar', 'x': 'Product Name', 'y': ['Quantity']}}`.
   - The allowed types are 'bar', 'line', 'pie', 'doughnut'.
   - `chart_config` keys ('x' and elements in 'y') must EXACTLY match the column names or keys in `result`. If output has `{{'product_id': 'Product A', 'amount_total': 100}}`, then `x` must be 'product_id' and `y` must be `['amount_total']`. 
   - If the request is for a chart but vaguely specified (e.g., "draw a graph for users", "draw a pie chart of sales" without specifying X/Y axes), DO NOT generate any Python code. Instead, generate a text response asking the user to clarify the chart type and which specific properties/fields to use for the axes.

If the schema does not contain the fields needed or if the question is unrelated to the database schema below, clearly state that you cannot execute it instead of hallucinating code. However, if the user asks for a chart and the request is vague as described in Rule 8, you MUST ask for clarification FIRST, before rejecting based on the schema.

SCHEMA:
{context}"""

# Hand-crafted examples of Odoo ORM expressions mapped to natural language queries.
FEW_SHOT_EXAMPLES = [

    # ══════════════════════════════════════════════════════════════════════════
    # MODULE 1 · SALES  (sale.order / sale.order.line)
    # ══════════════════════════════════════════════════════════════════════════

    # --- basic counts & filters ---
    {
        "question": "How many confirmed sale orders are there in total?",
        "code": "result = self.env['sale.order'].search_count([('state', 'in', ['sale', 'done'])])"
    },
    {
        "question": "How many sale orders are currently in draft state?",
        "code": "result = self.env['sale.order'].search_count([('state', '=', 'draft')])"
    },
    {
        "question": "List all sale orders created today.",
        "code": (
            "today = date.today().strftime('%Y-%m-%d')\n"
            "result = self.env['sale.order'].search_read(\n"
            "    [('create_date', '>=', today)],\n"
            "    ['name', 'partner_id', 'amount_total', 'state']\n"
            ")"
        )
    },
    {
        "question": "List all sale orders for a specific customer named 'Azure Interior'.",
        "code": (
            "partner = self.env['res.partner'].search([('name', 'ilike', 'Azure Interior')], limit=1)\n"
            "result = self.env['sale.order'].search_read(\n"
            "    [('partner_id', '=', partner.id)],\n"
            "    ['name', 'date_order', 'amount_total', 'state']\n"
            ") if partner else 'Customer not found'"
        )
    },
    {
        "question": "Show all sale orders with a total amount greater than 10,000.",
        "code": (
            "result = self.env['sale.order'].search_read(\n"
            "    [('amount_total', '>', 10000), ('state', 'in', ['sale', 'done'])],\n"
            "    ['name', 'partner_id', 'amount_total', 'date_order']\n"
            ")"
        )
    },
    {
        "question": "What is the total confirmed sales revenue this month?",
        "code": (
            "today = date.today()\n"
            "first_day = today.replace(day=1).strftime('%Y-%m-%d')\n"
            "result = self.env['sale.order'].read_group(\n"
            "    [('state', 'in', ['sale', 'done']), ('date_order', '>=', first_day)],\n"
            "    ['amount_total:sum'], []\n"
            ")"
        )
    },
    {
        "question": "What is the total confirmed sales revenue this year?",
        "code": (
            "year = date.today().year\n"
            "result = self.env['sale.order'].read_group(\n"
            "    [('state', 'in', ['sale', 'done']), ('date_order', '>=', f'{year}-01-01')],\n"
            "    ['amount_total:sum'], []\n"
            ")"
        )
    },
    {
        "question": "List all sale orders that are overdue and still in draft or sent state.",
        "code": (
            "today = date.today().strftime('%Y-%m-%d')\n"
            "result = self.env['sale.order'].search_read(\n"
            "    [('state', 'in', ['draft', 'sent']), ('validity_date', '<', today)],\n"
            "    ['name', 'partner_id', 'validity_date', 'amount_total']\n"
            ")"
        )
    },
    {
        "question": "Which are the top 10 best-selling products by quantity sold?",
        "code": (
            "result = self.env['sale.order.line'].read_group(\n"
            "    [('order_id.state', 'in', ['sale', 'done'])],\n"
            "    ['product_id', 'product_uom_qty:sum'],\n"
            "    ['product_id'],\n"
            "    orderby='product_uom_qty DESC',\n"
            "    limit=10\n"
            ")"
        )
    },
    {
        "question": "Show monthly sales totals for the current year grouped by month.",
        "code": (
            "year = date.today().year\n"
            "result = self.env['sale.order'].read_group(\n"
            "    [('state', 'in', ['sale', 'done']),\n"
            "     ('date_order', '>=', f'{year}-01-01'),\n"
            "     ('date_order', '<=', f'{year}-12-31')],\n"
            "    ['date_order:month', 'amount_total:sum'],\n"
            "    ['date_order:month']\n"
            ")"
        )
    },
    {
        "question": "Which customers have placed more than 5 orders and what is their total spend?",
        "code": (
            "groups = self.env['sale.order'].read_group(\n"
            "    [('state', 'in', ['sale', 'done'])],\n"
            "    ['partner_id', 'amount_total:sum', 'id:count'],\n"
            "    ['partner_id']\n"
            ")\n"
            "result = [\n"
            "    {'customer': g['partner_id'][1], 'orders': g['partner_id_count'], 'total_spend': g['amount_total']}\n"
            "    for g in groups if g['partner_id_count'] > 5\n"
            "]"
        )
    },
    {
        "question": "What is the average order value per sales team?",
        "code": (
            "result = self.env['sale.order'].read_group(\n"
            "    [('state', 'in', ['sale', 'done'])],\n"
            "    ['team_id', 'amount_total:avg'],\n"
            "    ['team_id']\n"
            ")"
        )
    },
    {
        "question": "List all sale orders that have been cancelled this year.",
        "code": (
            "year = date.today().year\n"
            "result = self.env['sale.order'].search_read(\n"
            "    [('state', '=', 'cancel'), ('date_order', '>=', f'{year}-01-01')],\n"
            "    ['name', 'partner_id', 'date_order', 'amount_total']\n"
            ")"
        )
    },
    {
        "question": "Show all sale order lines that contain the product 'Laptop'.",
        "code": (
            "result = self.env['sale.order.line'].search_read(\n"
            "    [('product_id.name', 'ilike', 'Laptop'),\n"
            "     ('order_id.state', 'in', ['sale', 'done'])],\n"
            "    ['order_id', 'product_id', 'product_uom_qty', 'price_unit', 'price_subtotal']\n"
            ")"
        )
    },
    {
        "question": "What is the total discount given on confirmed sale orders this year?",
        "code": (
            "year = date.today().year\n"
            "lines = self.env['sale.order.line'].search(\n"
            "    [('order_id.state', 'in', ['sale', 'done']),\n"
            "     ('order_id.date_order', '>=', f'{year}-01-01'),\n"
            "     ('discount', '>', 0)]\n"
            ")\n"
            "result = {\n"
            "    'total_discount_amount': sum(\n"
            "        (l.price_unit * l.product_uom_qty * l.discount / 100) for l in lines\n"
            "    )\n"
            "}"
        )
    },
    {
        "question": "Show quarterly sales totals for the current year.",
        "code": (
            "year = date.today().year\n"
            "result = self.env['sale.order'].read_group(\n"
            "    [('state', 'in', ['sale', 'done']),\n"
            "     ('date_order', '>=', f'{year}-01-01')],\n"
            "    ['date_order:quarter', 'amount_total:sum'],\n"
            "    ['date_order:quarter']\n"
            ")"
        )
    },
    {
        "question": "Which sale orders have a delivery status of 'Nothing to send'?",
        "code": (
            "result = self.env['sale.order'].search_read(\n"
            "    [('delivery_status', '=', False), ('state', 'in', ['sale', 'done'])],\n"
            "    ['name', 'partner_id', 'amount_total', 'delivery_status']\n"
            ")"
        )
    },
    {
        "question": "List the top 5 sales representatives by total confirmed revenue this year.",
        "code": (
            "year = date.today().year\n"
            "result = self.env['sale.order'].read_group(\n"
            "    [('state', 'in', ['sale', 'done']), ('date_order', '>=', f'{year}-01-01')],\n"
            "    ['user_id', 'amount_total:sum'],\n"
            "    ['user_id'],\n"
            "    orderby='amount_total DESC',\n"
            "    limit=5\n"
            ")"
        )
    },
    {
        "question": "Show total sales by country for confirmed orders.",
        "code": (
            "result = self.env['sale.order'].read_group(\n"
            "    [('state', 'in', ['sale', 'done'])],\n"
            "    ['partner_id.country_id', 'amount_total:sum'],\n"
            "    ['partner_id.country_id'],\n"
            "    orderby='amount_total DESC'\n"
            ")"
        )
    },
    {
        "question": "How many sale order lines have a discount greater than 20%?",
        "code": (
            "result = self.env['sale.order.line'].search_count(\n"
            "    [('discount', '>', 20), ('order_id.state', 'in', ['sale', 'done'])]\n"
            ")"
        )
    },
    {
        "question": "Show all uninvoiced confirmed sale orders.",
        "code": (
            "result = self.env['sale.order'].search_read(\n"
            "    [('state', 'in', ['sale', 'done']), ('invoice_status', '=', 'to invoice')],\n"
            "    ['name', 'partner_id', 'date_order', 'amount_total']\n"
            ")"
        )
    },
    {
        "question": "What is the total value of sale orders placed last month?",
        "code": (
            "today = date.today()\n"
            "first_this_month = today.replace(day=1)\n"
            "last_last_month = (first_this_month - timedelta(days=1)).strftime('%Y-%m-%d')\n"
            "first_last_month = (first_this_month - timedelta(days=1)).replace(day=1).strftime('%Y-%m-%d')\n"
            "result = self.env['sale.order'].read_group(\n"
            "    [('state', 'in', ['sale', 'done']),\n"
            "     ('date_order', '>=', first_last_month),\n"
            "     ('date_order', '<=', last_last_month)],\n"
            "    ['amount_total:sum'], []\n"
            ")"
        )
    },

    # ══════════════════════════════════════════════════════════════════════════
    # MODULE 2 · ACCOUNTING  (account.move / account.move.line / account.payment)
    # ══════════════════════════════════════════════════════════════════════════

    {
        "question": "What is the total revenue from posted customer invoices this year?",
        "code": (
            "year = date.today().year\n"
            "result = self.env['account.move'].read_group(\n"
            "    [('move_type', '=', 'out_invoice'), ('state', '=', 'posted'),\n"
            "     ('invoice_date', '>=', f'{year}-01-01')],\n"
            "    ['amount_untaxed:sum', 'amount_tax:sum', 'amount_total:sum'], []\n"
            ")"
        )
    },
    {
        "question": "List all overdue customer invoices with the amount still owed.",
        "code": (
            "today = date.today().strftime('%Y-%m-%d')\n"
            "result = self.env['account.move'].search_read(\n"
            "    [('move_type', '=', 'out_invoice'), ('state', '=', 'posted'),\n"
            "     ('payment_state', 'in', ['not_paid', 'partial']),\n"
            "     ('invoice_date_due', '<', today)],\n"
            "    ['name', 'partner_id', 'invoice_date_due', 'amount_residual']\n"
            ")"
        )
    },
    {
        "question": "Show the top 5 customers by outstanding receivable balance.",
        "code": (
            "result = self.env['account.move'].read_group(\n"
            "    [('move_type', '=', 'out_invoice'), ('state', '=', 'posted'),\n"
            "     ('payment_state', 'in', ['not_paid', 'partial'])],\n"
            "    ['partner_id', 'amount_residual:sum'],\n"
            "    ['partner_id'],\n"
            "    orderby='amount_residual DESC',\n"
            "    limit=5\n"
            ")"
        )
    },
    {
        "question": "List all vendor bills that are due in the next 7 days and are unpaid.",
        "code": (
            "today = date.today()\n"
            "week_later = (today + timedelta(days=7)).strftime('%Y-%m-%d')\n"
            "today_str = today.strftime('%Y-%m-%d')\n"
            "result = self.env['account.move'].search_read(\n"
            "    [('move_type', '=', 'in_invoice'), ('state', '=', 'posted'),\n"
            "     ('payment_state', '=', 'not_paid'),\n"
            "     ('invoice_date_due', '>=', today_str),\n"
            "     ('invoice_date_due', '<=', week_later)],\n"
            "    ['name', 'partner_id', 'invoice_date_due', 'amount_total']\n"
            ")"
        )
    },
    {
        "question": "What is the total amount of credit notes issued this month?",
        "code": (
            "today = date.today()\n"
            "first_day = today.replace(day=1).strftime('%Y-%m-%d')\n"
            "result = self.env['account.move'].read_group(\n"
            "    [('move_type', '=', 'out_refund'), ('state', '=', 'posted'),\n"
            "     ('invoice_date', '>=', first_day)],\n"
            "    ['amount_total:sum'], []\n"
            ")"
        )
    },
    {
        "question": "How many unpaid customer invoices are there?",
        "code": (
            "result = self.env['account.move'].search_count(\n"
            "    [('move_type', '=', 'out_invoice'), ('state', '=', 'posted'),\n"
            "     ('payment_state', '=', 'not_paid')]\n"
            ")"
        )
    },
    {
        "question": "Show total invoiced amount grouped by currency for posted invoices.",
        "code": (
            "result = self.env['account.move'].read_group(\n"
            "    [('move_type', '=', 'out_invoice'), ('state', '=', 'posted')],\n"
            "    ['currency_id', 'amount_total:sum'],\n"
            "    ['currency_id']\n"
            ")"
        )
    },
    {
        "question": "What is the total tax collected on customer invoices this quarter?",
        "code": (
            "today = date.today()\n"
            "quarter_start = date(today.year, ((today.month - 1) // 3) * 3 + 1, 1).strftime('%Y-%m-%d')\n"
            "result = self.env['account.move.line'].read_group(\n"
            "    [('move_id.move_type', '=', 'out_invoice'), ('move_id.state', '=', 'posted'),\n"
            "     ('move_id.invoice_date', '>=', quarter_start),\n"
            "     ('tax_line_id', '!=', False)],\n"
            "    ['account_id', 'balance:sum'],\n"
            "    ['account_id']\n"
            ")"
        )
    },
    {
        "question": "List all invoices that are in draft state.",
        "code": (
            "result = self.env['account.move'].search_read(\n"
            "    [('move_type', 'in', ['out_invoice', 'in_invoice']), ('state', '=', 'draft')],\n"
            "    ['name', 'partner_id', 'move_type', 'amount_total', 'create_date']\n"
            ")"
        )
    },
    {
        "question": "Show monthly revenue from customer invoices for the current year.",
        "code": (
            "year = date.today().year\n"
            "result = self.env['account.move'].read_group(\n"
            "    [('move_type', '=', 'out_invoice'), ('state', '=', 'posted'),\n"
            "     ('invoice_date', '>=', f'{year}-01-01')],\n"
            "    ['invoice_date:month', 'amount_untaxed:sum', 'amount_total:sum'],\n"
            "    ['invoice_date:month']\n"
            ")"
        )
    },
    {
        "question": "What is the total amount paid by customers this month?",
        "code": (
            "today = date.today()\n"
            "first_day = today.replace(day=1).strftime('%Y-%m-%d')\n"
            "result = self.env['account.payment'].read_group(\n"
            "    [('payment_type', '=', 'inbound'), ('state', '=', 'posted'),\n"
            "     ('date', '>=', first_day)],\n"
            "    ['amount:sum'], []\n"
            ")"
        )
    },
    {
        "question": "List all customer invoices that were fully paid in the last 30 days.",
        "code": (
            "cutoff = (date.today() - timedelta(days=30)).strftime('%Y-%m-%d')\n"
            "result = self.env['account.move'].search_read(\n"
            "    [('move_type', '=', 'out_invoice'), ('state', '=', 'posted'),\n"
            "     ('payment_state', '=', 'paid'),\n"
            "     ('invoice_date', '>=', cutoff)],\n"
            "    ['name', 'partner_id', 'invoice_date', 'amount_total']\n"
            ")"
        )
    },
    {
        "question": "Show total vendor bills received per vendor this year.",
        "code": (
            "year = date.today().year\n"
            "result = self.env['account.move'].read_group(\n"
            "    [('move_type', '=', 'in_invoice'), ('state', '=', 'posted'),\n"
            "     ('invoice_date', '>=', f'{year}-01-01')],\n"
            "    ['partner_id', 'amount_total:sum'],\n"
            "    ['partner_id'],\n"
            "    orderby='amount_total DESC'\n"
            ")"
        )
    },
    {
        "question": "What is the total outstanding payable (unpaid vendor bills)?",
        "code": (
            "result = self.env['account.move'].read_group(\n"
            "    [('move_type', '=', 'in_invoice'), ('state', '=', 'posted'),\n"
            "     ('payment_state', 'in', ['not_paid', 'partial'])],\n"
            "    ['amount_residual:sum'], []\n"
            ")"
        )
    },
    {
        "question": "Show the ageing of overdue receivables bucketed by 0-30, 31-60, 61-90, and 90+ days.",
        "code": (
            "today = date.today()\n"
            "overdue = self.env['account.move'].search(\n"
            "    [('move_type', '=', 'out_invoice'), ('state', '=', 'posted'),\n"
            "     ('payment_state', 'in', ['not_paid', 'partial']),\n"
            "     ('invoice_date_due', '<', today.strftime('%Y-%m-%d'))]\n"
            ")\n"
            "buckets = {'0_30': 0, '31_60': 0, '61_90': 0, '90_plus': 0}\n"
            "for inv in overdue:\n"
            "    days = (today - inv.invoice_date_due).days\n"
            "    if days <= 30: buckets['0_30'] += inv.amount_residual\n"
            "    elif days <= 60: buckets['31_60'] += inv.amount_residual\n"
            "    elif days <= 90: buckets['61_90'] += inv.amount_residual\n"
            "    else: buckets['90_plus'] += inv.amount_residual\n"
            "result = buckets"
        )
    },
    {
        "question": "List all payments made to vendors in the last 7 days.",
        "code": (
            "cutoff = (date.today() - timedelta(days=7)).strftime('%Y-%m-%d')\n"
            "result = self.env['account.payment'].search_read(\n"
            "    [('payment_type', '=', 'outbound'), ('state', '=', 'posted'),\n"
            "     ('date', '>=', cutoff)],\n"
            "    ['name', 'partner_id', 'amount', 'date', 'journal_id']\n"
            ")"
        )
    },
    {
        "question": "What is the highest invoice amount issued to a single customer?",
        "code": (
            "invoices = self.env['account.move'].search_read(\n"
            "    [('move_type', '=', 'out_invoice'), ('state', '=', 'posted')],\n"
            "    ['name', 'partner_id', 'amount_total'],\n"
            "    order='amount_total DESC',\n"
            "    limit=1\n"
            ")\n"
            "result = invoices[0] if invoices else 'No invoices found'"
        )
    },
    {
        "question": "Show all journal entries posted in the last 30 days.",
        "code": (
            "cutoff = (date.today() - timedelta(days=30)).strftime('%Y-%m-%d')\n"
            "result = self.env['account.move'].search_read(\n"
            "    [('move_type', '=', 'entry'), ('state', '=', 'posted'),\n"
            "     ('date', '>=', cutoff)],\n"
            "    ['name', 'date', 'journal_id', 'amount_total', 'ref']\n"
            ")"
        )
    },
    {
        "question": "What is the total expense recorded in the Expenses journal this month?",
        "code": (
            "today = date.today()\n"
            "first_day = today.replace(day=1).strftime('%Y-%m-%d')\n"
            "journal = self.env['account.journal'].search([('type', '=', 'purchase'), ('name', 'ilike', 'expense')], limit=1)\n"
            "result = self.env['account.move'].read_group(\n"
            "    [('journal_id', '=', journal.id), ('state', '=', 'posted'),\n"
            "     ('date', '>=', first_day)],\n"
            "    ['amount_total:sum'], []\n"
            ") if journal else 'Expenses journal not found'"
        )
    },
    {
        "question": "How many invoices were issued per month this year?",
        "code": (
            "year = date.today().year\n"
            "result = self.env['account.move'].read_group(\n"
            "    [('move_type', '=', 'out_invoice'), ('state', '=', 'posted'),\n"
            "     ('invoice_date', '>=', f'{year}-01-01')],\n"
            "    ['invoice_date:month', 'id:count'],\n"
            "    ['invoice_date:month']\n"
            ")"
        )
    },
    {
        "question": "What is the average invoice value for confirmed customer invoices this year?",
        "code": (
            "year = date.today().year\n"
            "result = self.env['account.move'].read_group(\n"
            "    [('move_type', '=', 'out_invoice'), ('state', '=', 'posted'),\n"
            "     ('invoice_date', '>=', f'{year}-01-01')],\n"
            "    ['amount_total:avg'], []\n"
            ")"
        )
    },

    # ══════════════════════════════════════════════════════════════════════════
    # MODULE 3 · INVENTORY  (stock.picking / stock.move / stock.quant)
    # ══════════════════════════════════════════════════════════════════════════

    {
        "question": "How many delivery orders are still pending right now?",
        "code": (
            "result = self.env['stock.picking'].search_count(\n"
            "    [('picking_type_code', '=', 'outgoing'),\n"
            "     ('state', 'in', ['confirmed', 'assigned'])]\n"
            ")"
        )
    },
    {
        "question": "List all deliveries that were completed today.",
        "code": (
            "today = date.today().strftime('%Y-%m-%d')\n"
            "result = self.env['stock.picking'].search_read(\n"
            "    [('picking_type_code', '=', 'outgoing'), ('state', '=', 'done'),\n"
            "     ('date_done', '>=', today)],\n"
            "    ['name', 'partner_id', 'date_done', 'origin']\n"
            ")"
        )
    },
    {
        "question": "What is the current stock quantity of all products in all internal locations?",
        "code": (
            "result = self.env['stock.quant'].read_group(\n"
            "    [('location_id.usage', '=', 'internal')],\n"
            "    ['product_id', 'quantity:sum', 'reserved_quantity:sum'],\n"
            "    ['product_id']\n"
            ")"
        )
    },
    {
        "question": "List all products that have negative stock.",
        "code": (
            "result = self.env['stock.quant'].search_read(\n"
            "    [('location_id.usage', '=', 'internal'), ('quantity', '<', 0)],\n"
            "    ['product_id', 'location_id', 'quantity']\n"
            ")"
        )
    },
    {
        "question": "Which delivery orders are overdue and have not been shipped yet?",
        "code": (
            "today = date.today().strftime('%Y-%m-%d')\n"
            "result = self.env['stock.picking'].search_read(\n"
            "    [('picking_type_code', '=', 'outgoing'),\n"
            "     ('state', 'not in', ['done', 'cancel']),\n"
            "     ('scheduled_date', '<', today)],\n"
            "    ['name', 'partner_id', 'scheduled_date', 'state']\n"
            ")"
        )
    },
    {
        "question": "Show total quantity received per product this month.",
        "code": (
            "today = date.today()\n"
            "first_day = today.replace(day=1).strftime('%Y-%m-%d')\n"
            "result = self.env['stock.move'].read_group(\n"
            "    [('picking_id.picking_type_code', '=', 'incoming'), ('state', '=', 'done'),\n"
            "     ('date', '>=', first_day)],\n"
            "    ['product_id', 'quantity:sum'],\n"
            "    ['product_id']\n"
            ")"
        )
    },
    {
        "question": "What is the total inventory value across all internal locations?",
        "code": (
            "quants = self.env['stock.quant'].search([('location_id.usage', '=', 'internal')])\n"
            "result = {\n"
            "    'total_inventory_value': sum(\n"
            "        q.quantity * q.product_id.standard_price for q in quants\n"
            "    )\n"
            "}"
        )
    },
    {
        "question": "List all products below their reorder point.",
        "code": (
            "orderpoints = self.env['stock.warehouse.orderpoint'].search([])\n"
            "result = [\n"
            "    {'product': op.product_id.name, 'on_hand': op.qty_on_hand,\n"
            "     'min_qty': op.product_min_qty, 'warehouse': op.warehouse_id.name}\n"
            "    for op in orderpoints if op.qty_on_hand < op.product_min_qty\n"
            "]"
        )
    },
    {
        "question": "How many receipts (incoming shipments) have been validated this week?",
        "code": (
            "monday = (date.today() - timedelta(days=date.today().weekday())).strftime('%Y-%m-%d')\n"
            "result = self.env['stock.picking'].search_count(\n"
            "    [('picking_type_code', '=', 'incoming'), ('state', '=', 'done'),\n"
            "     ('date_done', '>=', monday)]\n"
            ")"
        )
    },
    {
        "question": "Which products have had no stock movements in the last 90 days (slow movers)?",
        "code": (
            "cutoff = (date.today() - timedelta(days=90)).strftime('%Y-%m-%d')\n"
            "moved_product_ids = self.env['stock.move'].search(\n"
            "    [('state', '=', 'done'), ('date', '>=', cutoff)]\n"
            ").mapped('product_id.id')\n"
            "result = self.env['product.product'].search_read(\n"
            "    [('id', 'not in', moved_product_ids), ('type', '=', 'consu')],\n"
            "    ['name', 'default_code', 'categ_id']\n"
            ")"
        )
    },
    {
        "question": "Show all internal transfer operations currently in progress.",
        "code": (
            "result = self.env['stock.picking'].search_read(\n"
            "    [('picking_type_code', '=', 'internal'),\n"
            "     ('state', 'in', ['confirmed', 'assigned', 'waiting'])],\n"
            "    ['name', 'location_id', 'location_dest_id', 'scheduled_date', 'state']\n"
            ")"
        )
    },
    {
        "question": "What is the total quantity shipped out per product this year?",
        "code": (
            "year = date.today().year\n"
            "result = self.env['stock.move'].read_group(\n"
            "    [('picking_id.picking_type_code', '=', 'outgoing'), ('state', '=', 'done'),\n"
            "     ('date', '>=', f'{year}-01-01')],\n"
            "    ['product_id', 'quantity:sum'],\n"
            "    ['product_id'],\n"
            "    orderby='quantity DESC'\n"
            ")"
        )
    },
    {
        "question": "List all stock pickings that were cancelled this month.",
        "code": (
            "today = date.today()\n"
            "first_day = today.replace(day=1).strftime('%Y-%m-%d')\n"
            "result = self.env['stock.picking'].search_read(\n"
            "    [('state', '=', 'cancel'), ('write_date', '>=', first_day)],\n"
            "    ['name', 'picking_type_code', 'partner_id', 'write_date']\n"
            ")"
        )
    },
    {
        "question": "Show all lots/serial numbers that are about to expire in the next 30 days.",
        "code": (
            "today = date.today().strftime('%Y-%m-%d')\n"
            "in_30 = (date.today() + timedelta(days=30)).strftime('%Y-%m-%d')\n"
            "result = self.env['stock.lot'].search_read(\n"
            "    [('expiration_date', '>=', today), ('expiration_date', '<=', in_30)],\n"
            "    ['name', 'product_id', 'expiration_date', 'product_qty']\n"
            ")"
        )
    },
    {
        "question": "How many products are currently fully out of stock?",
        "code": (
            "products_with_stock = self.env['stock.quant'].search(\n"
            "    [('location_id.usage', '=', 'internal'), ('quantity', '>', 0)]\n"
            ").mapped('product_id.id')\n"
            "result = self.env['product.product'].search_count(\n"
            "    [('type', '=', 'consu'), ('id', 'not in', products_with_stock)]\n"
            ")"
        )
    },
    {
        "question": "Show stock quantity grouped by warehouse for a specific product named 'Chair'.",
        "code": (
            "product = self.env['product.product'].search([('name', 'ilike', 'Chair')], limit=1)\n"
            "if product:\n"
            "    result = self.env['stock.quant'].read_group(\n"
            "        [('product_id', '=', product.id), ('location_id.usage', '=', 'internal')],\n"
            "        ['location_id', 'quantity:sum'],\n"
            "        ['location_id']\n"
            "    )\n"
            "else:\n"
            "    result = 'Product not found'"
        )
    },
    {
        "question": "List all receipt operations (incoming shipments) that are still waiting to be processed.",
        "code": (
            "result = self.env['stock.picking'].search_read(\n"
            "    [('picking_type_code', '=', 'incoming'),\n"
            "     ('state', 'in', ['confirmed', 'assigned', 'waiting'])],\n"
            "    ['name', 'partner_id', 'scheduled_date', 'origin']\n"
            ")"
        )
    },
    {
        "question": "What is the total number of stock moves done this month?",
        "code": (
            "today = date.today()\n"
            "first_day = today.replace(day=1).strftime('%Y-%m-%d')\n"
            "result = self.env['stock.move'].search_count(\n"
            "    [('state', '=', 'done'), ('date', '>=', first_day)]\n"
            ")"
        )
    },
    {
        "question": "Show products with the highest stock value (quantity × cost).",
        "code": (
            "quants = self.env['stock.quant'].search([('location_id.usage', '=', 'internal'), ('quantity', '>', 0)])\n"
            "product_values = {}\n"
            "for q in quants:\n"
            "    key = q.product_id.name\n"
            "    product_values[key] = product_values.get(key, 0) + q.quantity * q.product_id.standard_price\n"
            "result = sorted(product_values.items(), key=lambda x: x[1], reverse=True)[:10]"
        )
    },
    {
        "question": "List all products that have stock reserved but no confirmed deliveries.",
        "code": (
            "result = self.env['stock.quant'].search_read(\n"
            "    [('location_id.usage', '=', 'internal'), ('reserved_quantity', '>', 0)],\n"
            "    ['product_id', 'location_id', 'quantity', 'reserved_quantity']\n"
            ")"
        )
    },

    # ══════════════════════════════════════════════════════════════════════════
    # MODULE 4 · PURCHASE  (purchase.order / purchase.order.line)
    # ══════════════════════════════════════════════════════════════════════════

    {
        "question": "What is the total amount spent on purchases per vendor this year?",
        "code": (
            "year = date.today().year\n"
            "result = self.env['purchase.order'].read_group(\n"
            "    [('state', 'in', ['purchase', 'done']), ('date_order', '>=', f'{year}-01-01')],\n"
            "    ['partner_id', 'amount_total:sum'],\n"
            "    ['partner_id'],\n"
            "    orderby='amount_total DESC'\n"
            ")"
        )
    },
    {
        "question": "List all purchase orders confirmed but not yet fully received.",
        "code": (
            "result = self.env['purchase.order'].search_read(\n"
            "    [('state', '=', 'purchase'), ('receipt_status', '!=', 'full')],\n"
            "    ['name', 'partner_id', 'date_order', 'date_planned', 'amount_total', 'receipt_status']\n"
            ")"
        )
    },
    {
        "question": "Which are the top 10 most purchased products by quantity this year?",
        "code": (
            "year = date.today().year\n"
            "result = self.env['purchase.order.line'].read_group(\n"
            "    [('order_id.state', 'in', ['purchase', 'done']),\n"
            "     ('order_id.date_order', '>=', f'{year}-01-01')],\n"
            "    ['product_id', 'product_qty:sum'],\n"
            "    ['product_id'],\n"
            "    orderby='product_qty DESC',\n"
            "    limit=10\n"
            ")"
        )
    },
    {
        "question": "Show all draft purchase orders waiting for more than 3 days.",
        "code": (
            "cutoff = (date.today() - timedelta(days=3)).strftime('%Y-%m-%d')\n"
            "result = self.env['purchase.order'].search_read(\n"
            "    [('state', '=', 'draft'), ('create_date', '<=', cutoff)],\n"
            "    ['name', 'partner_id', 'create_date', 'amount_total']\n"
            ")"
        )
    },
    {
        "question": "What is the average lead time per vendor for received purchase orders?",
        "code": (
            "orders = self.env['purchase.order'].search([('state', '=', 'done')])\n"
            "vendor_data = {}\n"
            "for order in orders:\n"
            "    if order.date_approve and order.effective_date:\n"
            "        lead = (order.effective_date - order.date_approve).days\n"
            "        key = order.partner_id.name\n"
            "        vendor_data.setdefault(key, []).append(lead)\n"
            "result = [\n"
            "    {'vendor': v, 'avg_lead_time_days': round(sum(days) / len(days), 1)}\n"
            "    for v, days in vendor_data.items()\n"
            "]"
        )
    },
    {
        "question": "Show total purchase spending grouped by product category this year.",
        "code": (
            "year = date.today().year\n"
            "result = self.env['purchase.order.line'].read_group(\n"
            "    [('order_id.state', 'in', ['purchase', 'done']),\n"
            "     ('order_id.date_order', '>=', f'{year}-01-01')],\n"
            "    ['product_id.categ_id', 'price_subtotal:sum'],\n"
            "    ['product_id.categ_id']\n"
            ")"
        )
    },
    {
        "question": "How many purchase orders are currently awaiting approval?",
        "code": (
            "result = self.env['purchase.order'].search_count(\n"
            "    [('state', '=', 'to approve')]\n"
            ")"
        )
    },
    {
        "question": "List all purchase orders placed this week.",
        "code": (
            "monday = (date.today() - timedelta(days=date.today().weekday())).strftime('%Y-%m-%d')\n"
            "result = self.env['purchase.order'].search_read(\n"
            "    [('create_date', '>=', monday)],\n"
            "    ['name', 'partner_id', 'date_order', 'amount_total', 'state']\n"
            ")"
        )
    },
    {
        "question": "Show purchase orders with a delivery date overdue and not yet received.",
        "code": (
            "today = date.today().strftime('%Y-%m-%d')\n"
            "result = self.env['purchase.order'].search_read(\n"
            "    [('state', 'in', ['purchase']), ('date_planned', '<', today),\n"
            "     ('receipt_status', '!=', 'full')],\n"
            "    ['name', 'partner_id', 'date_planned', 'amount_total', 'receipt_status']\n"
            ")"
        )
    },
    {
        "question": "What is the total value of all purchase orders placed this month?",
        "code": (
            "today = date.today()\n"
            "first_day = today.replace(day=1).strftime('%Y-%m-%d')\n"
            "result = self.env['purchase.order'].read_group(\n"
            "    [('state', 'in', ['purchase', 'done']), ('date_order', '>=', first_day)],\n"
            "    ['amount_total:sum'], []\n"
            ")"
        )
    },
    {
        "question": "Which vendors have the highest number of purchase orders this year?",
        "code": (
            "year = date.today().year\n"
            "result = self.env['purchase.order'].read_group(\n"
            "    [('date_order', '>=', f'{year}-01-01')],\n"
            "    ['partner_id', 'id:count'],\n"
            "    ['partner_id'],\n"
            "    orderby='partner_id_count DESC',\n"
            "    limit=10\n"
            ")"
        )
    },
    {
        "question": "List all purchase order lines where the received quantity is less than ordered.",
        "code": (
            "lines = self.env['purchase.order.line'].search(\n"
            "    [('order_id.state', 'in', ['purchase', 'done'])]\n"
            ")\n"
            "result = [\n"
            "    {'order': l.order_id.name, 'product': l.product_id.name,\n"
            "     'ordered': l.product_qty, 'received': l.qty_received}\n"
            "    for l in lines if l.qty_received < l.product_qty\n"
            "]"
        )
    },
    {
        "question": "What is the total number of purchase orders by state?",
        "code": (
            "result = self.env['purchase.order'].read_group(\n"
            "    [],\n"
            "    ['state', 'id:count'],\n"
            "    ['state']\n"
            ")"
        )
    },
    {
        "question": "Show all purchase orders from a specific vendor named 'Wood Corner'.",
        "code": (
            "vendor = self.env['res.partner'].search([('name', 'ilike', 'Wood Corner')], limit=1)\n"
            "result = self.env['purchase.order'].search_read(\n"
            "    [('partner_id', '=', vendor.id)],\n"
            "    ['name', 'date_order', 'amount_total', 'state', 'receipt_status']\n"
            ") if vendor else 'Vendor not found'"
        )
    },
    {
        "question": "Show the monthly purchase spend trend for the current year.",
        "code": (
            "year = date.today().year\n"
            "result = self.env['purchase.order'].read_group(\n"
            "    [('state', 'in', ['purchase', 'done']), ('date_order', '>=', f'{year}-01-01')],\n"
            "    ['date_order:month', 'amount_total:sum'],\n"
            "    ['date_order:month']\n"
            ")"
        )
    },
    {
        "question": "List purchase orders whose total amount exceeds 50,000.",
        "code": (
            "result = self.env['purchase.order'].search_read(\n"
            "    [('amount_total', '>', 50000), ('state', 'in', ['purchase', 'done'])],\n"
            "    ['name', 'partner_id', 'date_order', 'amount_total']\n"
            ")"
        )
    },

    # ══════════════════════════════════════════════════════════════════════════
    # MODULE 5 · CRM  (crm.lead)
    # ══════════════════════════════════════════════════════════════════════════

    {
        "question": "How many leads are currently open grouped by stage?",
        "code": (
            "result = self.env['crm.lead'].read_group(\n"
            "    [('type', '=', 'lead'), ('active', '=', True)],\n"
            "    ['stage_id', 'id:count'],\n"
            "    ['stage_id']\n"
            ")"
        )
    },
    {
        "question": "What is the total expected revenue in the pipeline grouped by salesperson?",
        "code": (
            "result = self.env['crm.lead'].read_group(\n"
            "    [('type', '=', 'opportunity'), ('active', '=', True)],\n"
            "    ['user_id', 'expected_revenue:sum'],\n"
            "    ['user_id']\n"
            ")"
        )
    },
    {
        "question": "List all opportunities with no activity in the last 14 days.",
        "code": (
            "cutoff = (date.today() - timedelta(days=14)).strftime('%Y-%m-%d')\n"
            "result = self.env['crm.lead'].search_read(\n"
            "    [('type', '=', 'opportunity'), ('active', '=', True),\n"
            "     '|', ('activity_date_deadline', '<', cutoff),\n"
            "          ('activity_date_deadline', '=', False)],\n"
            "    ['name', 'partner_id', 'user_id', 'stage_id', 'activity_date_deadline']\n"
            ")"
        )
    },
    {
        "question": "What is the win rate percentage this year?",
        "code": (
            "year = date.today().year\n"
            "won = self.env['crm.lead'].search_count(\n"
            "    [('type', '=', 'opportunity'), ('stage_id.is_won', '=', True),\n"
            "     ('date_closed', '>=', f'{year}-01-01')]\n"
            ")\n"
            "lost = self.env['crm.lead'].with_context(active_test=False).search_count(\n"
            "    [('type', '=', 'opportunity'), ('active', '=', False),\n"
            "     ('probability', '=', 0), ('date_closed', '>=', f'{year}-01-01')]\n"
            ")\n"
            "total = won + lost\n"
            "result = {'won': won, 'lost': lost, 'win_rate_percent': round((won / total) * 100, 2) if total else 0}"
        )
    },
    {
        "question": "Show the total pipeline value grouped by sales team.",
        "code": (
            "result = self.env['crm.lead'].read_group(\n"
            "    [('type', '=', 'opportunity'), ('active', '=', True)],\n"
            "    ['team_id', 'expected_revenue:sum'],\n"
            "    ['team_id'],\n"
            "    orderby='expected_revenue DESC'\n"
            ")"
        )
    },
    {
        "question": "Which leads were created this week and not yet assigned to a salesperson?",
        "code": (
            "monday = (date.today() - timedelta(days=date.today().weekday())).strftime('%Y-%m-%d')\n"
            "result = self.env['crm.lead'].search_read(\n"
            "    [('type', '=', 'lead'), ('create_date', '>=', monday), ('user_id', '=', False)],\n"
            "    ['name', 'partner_name', 'email_from', 'create_date', 'source_id']\n"
            ")"
        )
    },
    {
        "question": "What is the average time to close a won opportunity per sales team?",
        "code": (
            "won_opps = self.env['crm.lead'].search(\n"
            "    [('type', '=', 'opportunity'), ('stage_id.is_won', '=', True),\n"
            "     ('date_closed', '!=', False), ('create_date', '!=', False)]\n"
            ")\n"
            "team_data = {}\n"
            "for opp in won_opps:\n"
            "    days = (opp.date_closed - opp.create_date).days\n"
            "    key = opp.team_id.name or 'No Team'\n"
            "    team_data.setdefault(key, []).append(days)\n"
            "result = [{'team': t, 'avg_days_to_close': round(sum(d)/len(d), 1)} for t, d in team_data.items()]"
        )
    },
    {
        "question": "How many new leads were created this month vs last month?",
        "code": (
            "today = date.today()\n"
            "first_this_month = today.replace(day=1)\n"
            "this_month_start = first_this_month.strftime('%Y-%m-%d')\n"
            "last_month_end = (first_this_month - timedelta(days=1)).strftime('%Y-%m-%d')\n"
            "last_month_start = (first_this_month - timedelta(days=1)).replace(day=1).strftime('%Y-%m-%d')\n"
            "this_month = self.env['crm.lead'].search_count(\n"
            "    [('type', '=', 'lead'), ('create_date', '>=', this_month_start)]\n"
            ")\n"
            "last_month = self.env['crm.lead'].search_count(\n"
            "    [('type', '=', 'lead'), ('create_date', '>=', last_month_start),\n"
            "     ('create_date', '<=', last_month_end)]\n"
            ")\n"
            "result = {'this_month': this_month, 'last_month': last_month, 'growth': this_month - last_month}"
        )
    },
    {
        "question": "List all opportunities with a probability greater than 70% and expected revenue over 5000.",
        "code": (
            "result = self.env['crm.lead'].search_read(\n"
            "    [('type', '=', 'opportunity'), ('active', '=', True),\n"
            "     ('probability', '>', 70), ('expected_revenue', '>', 5000)],\n"
            "    ['name', 'partner_id', 'user_id', 'probability', 'expected_revenue', 'stage_id']\n"
            ")"
        )
    },
    {
        "question": "Show leads grouped by source (campaign/medium) this year.",
        "code": (
            "year = date.today().year\n"
            "result = self.env['crm.lead'].read_group(\n"
            "    [('type', '=', 'lead'), ('create_date', '>=', f'{year}-01-01')],\n"
            "    ['source_id', 'id:count'],\n"
            "    ['source_id'],\n"
            "    orderby='source_id_count DESC'\n"
            ")"
        )
    },
    {
        "question": "What is the total expected revenue of opportunities closing this month?",
        "code": (
            "today = date.today()\n"
            "first_day = today.replace(day=1).strftime('%Y-%m-%d')\n"
            "last_day = today.replace(day=((today.replace(day=28) + timedelta(days=4)).replace(day=1) - timedelta(days=1)).day).strftime('%Y-%m-%d')\n"
            "result = self.env['crm.lead'].read_group(\n"
            "    [('type', '=', 'opportunity'), ('active', '=', True),\n"
            "     ('date_deadline', '>=', first_day), ('date_deadline', '<=', last_day)],\n"
            "    ['expected_revenue:sum', 'id:count'], []\n"
            ")"
        )
    },
    {
        "question": "Which salesperson has the most open opportunities right now?",
        "code": (
            "result = self.env['crm.lead'].read_group(\n"
            "    [('type', '=', 'opportunity'), ('active', '=', True)],\n"
            "    ['user_id', 'id:count'],\n"
            "    ['user_id'],\n"
            "    orderby='user_id_count DESC',\n"
            "    limit=1\n"
            ")"
        )
    },
    {
        "question": "List all lost opportunities and the reason they were lost, grouped by lost reason.",
        "code": (
            "result = self.env['crm.lead'].read_group(\n"
            "    [('type', '=', 'opportunity'), ('active', '=', False), ('probability', '=', 0)],\n"
            "    ['lost_reason_id', 'id:count'],\n"
            "    ['lost_reason_id'],\n"
            "    orderby='lost_reason_id_count DESC'\n"
            ")"
        )
    },
    {
        "question": "Show all opportunities that have a deadline within the next 7 days.",
        "code": (
            "today = date.today().strftime('%Y-%m-%d')\n"
            "in_7 = (date.today() + timedelta(days=7)).strftime('%Y-%m-%d')\n"
            "result = self.env['crm.lead'].search_read(\n"
            "    [('type', '=', 'opportunity'), ('active', '=', True),\n"
            "     ('date_deadline', '>=', today), ('date_deadline', '<=', in_7)],\n"
            "    ['name', 'partner_id', 'user_id', 'expected_revenue', 'date_deadline', 'stage_id']\n"
            ")"
        )
    },
    {
        "question": "What is the total number of activities scheduled per salesperson?",
        "code": (
            "result = self.env['mail.activity'].read_group(\n"
            "    [('res_model', '=', 'crm.lead')],\n"
            "    ['user_id', 'id:count'],\n"
            "    ['user_id'],\n"
            "    orderby='user_id_count DESC'\n"
            ")"
        )
    },

    # ══════════════════════════════════════════════════════════════════════════
    # MODULE 6 · HR  (hr.employee / hr.leave / hr.payslip / hr.attendance)
    # ══════════════════════════════════════════════════════════════════════════

    {
        "question": "How many active employees are there?",
        "code": "result = self.env['hr.employee'].search_count([('active', '=', True)])"
    },
    {
        "question": "List all employees grouped by department.",
        "code": (
            "result = self.env['hr.employee'].read_group(\n"
            "    [('active', '=', True)],\n"
            "    ['department_id', 'id:count'],\n"
            "    ['department_id']\n"
            ")"
        )
    },
    {
        "question": "How many employees joined the company this year?",
        "code": (
            "year = date.today().year\n"
            "result = self.env['hr.employee'].search_count(\n"
            "    [('active', '=', True), ('create_date', '>=', f'{year}-01-01')]\n"
            ")"
        )
    },
    {
        "question": "List all employees whose contract is expiring in the next 30 days.",
        "code": (
            "today = date.today().strftime('%Y-%m-%d')\n"
            "in_30 = (date.today() + timedelta(days=30)).strftime('%Y-%m-%d')\n"
            "result = self.env['hr.contract'].search_read(\n"
            "    [('state', '=', 'open'),\n"
            "     ('date_end', '>=', today), ('date_end', '<=', in_30)],\n"
            "    ['name', 'employee_id', 'date_end', 'wage']\n"
            ")"
        )
    },
    {
        "question": "Show all pending leave requests awaiting approval.",
        "code": (
            "result = self.env['hr.leave'].search_read(\n"
            "    [('state', '=', 'confirm')],\n"
            "    ['employee_id', 'holiday_status_id', 'date_from', 'date_to', 'number_of_days']\n"
            ")"
        )
    },
    {
        "question": "How many leave days were taken by department this month?",
        "code": (
            "today = date.today()\n"
            "first_day = today.replace(day=1).strftime('%Y-%m-%d')\n"
            "result = self.env['hr.leave'].read_group(\n"
            "    [('state', '=', 'validate'), ('date_from', '>=', first_day)],\n"
            "    ['department_id', 'number_of_days:sum'],\n"
            "    ['department_id']\n"
            ")"
        )
    },
    {
        "question": "List all employees on leave today.",
        "code": (
            "now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')\n"
            "today = datetime.now().strftime('%Y-%m-%d')\n"
            "result = self.env['hr.leave'].search_read(\n"
            "    [('state', '=', 'validate'),\n"
            "     ('date_from', '<=', now),\n"
            "     ('date_to', '>=', today)],\n"
            "    ['employee_id', 'holiday_status_id', 'date_from', 'date_to']\n"
            ")"
        )
    },
    {
        "question": "What is the total payroll cost this month across all employees?",
        "code": (
            "today = date.today()\n"
            "result = self.env['hr.payslip'].read_group(\n"
            "    [('state', '=', 'done'),\n"
            "     ('date_from', '>=', today.replace(day=1).strftime('%Y-%m-%d'))],\n"
            "    ['net_wage:sum'], []\n"
            ")"
        )
    },
    {
        "question": "Show the top 10 highest paid employees based on their latest payslip.",
        "code": (
            "payslips = self.env['hr.payslip'].search(\n"
            "    [('state', '=', 'done')],\n"
            "    order='date_to DESC'\n"
            ")\n"
            "seen = set()\n"
            "top = []\n"
            "for p in payslips:\n"
            "    if p.employee_id.id not in seen:\n"
            "        seen.add(p.employee_id.id)\n"
            "        top.append({'employee': p.employee_id.name, 'net_wage': p.net_wage})\n"
            "result = sorted(top, key=lambda x: x['net_wage'], reverse=True)[:10]"
        )
    },
    {
        "question": "How many employees have not taken any leave this year?",
        "code": (
            "year = date.today().year\n"
            "employees_with_leave = self.env['hr.leave'].search(\n"
            "    [('state', '=', 'validate'), ('date_from', '>=', f'{year}-01-01')]\n"
            ").mapped('employee_id.id')\n"
            "result = self.env['hr.employee'].search_count(\n"
            "    [('active', '=', True), ('id', 'not in', employees_with_leave)]\n"
            ")"
        )
    },
    {
        "question": "Show total overtime hours logged per employee this month.",
        "code": (
            "today = date.today()\n"
            "first_day = today.replace(day=1).strftime('%Y-%m-%d')\n"
            "result = self.env['hr.attendance'].read_group(\n"
            "    [('check_in', '>=', first_day), ('overtime_hours', '>', 0)],\n"
            "    ['employee_id', 'overtime_hours:sum'],\n"
            "    ['employee_id'],\n"
            "    orderby='overtime_hours DESC'\n"
            ")"
        )
    },
    {
        "question": "List employees who are late (check-in after 9 AM) most frequently this week.",
        "code": (
            "monday = date.today() - timedelta(days=date.today().weekday())\n"
            "monday_str = monday.strftime('%Y-%m-%d')\n"
            "attendances = self.env['hr.attendance'].search(\n"
            "    [('check_in', '>=', monday_str)]\n"
            ")\n"
            "late_counts = {}\n"
            "for att in attendances:\n"
            "    if att.check_in and att.check_in.hour >= 9:\n"
            "        key = att.employee_id.name\n"
            "        late_counts[key] = late_counts.get(key, 0) + 1\n"
            "result = sorted(late_counts.items(), key=lambda x: x[1], reverse=True)"
        )
    },
    {
        "question": "List all employees by job position.",
        "code": (
            "result = self.env['hr.employee'].read_group(\n"
            "    [('active', '=', True)],\n"
            "    ['job_id', 'id:count'],\n"
            "    ['job_id'],\n"
            "    orderby='job_id_count DESC'\n"
            ")"
        )
    },
    {
        "question": "How many employees are on probation (contract type = probation)?",
        "code": (
            "result = self.env['hr.contract'].search_count(\n"
            "    [('state', '=', 'open'), ('contract_type_id.name', 'ilike', 'probation')]\n"
            ")"
        )
    },
    {
        "question": "What is the total number of leave requests approved vs refused this year?",
        "code": (
            "year = date.today().year\n"
            "result = self.env['hr.leave'].read_group(\n"
            "    [('date_from', '>=', f'{year}-01-01'), ('state', 'in', ['validate', 'refuse'])],\n"
            "    ['state', 'id:count'],\n"
            "    ['state']\n"
            ")"
        )
    },
    {
        "question": "Show employees whose birthday is this month.",
        "code": (
            "month = date.today().month\n"
            "employees = self.env['hr.employee'].search([('active', '=', True), ('birthday', '!=', False)])\n"
            "result = [\n"
            "    {'name': e.name, 'birthday': str(e.birthday), 'department': e.department_id.name}\n"
            "    for e in employees if e.birthday and e.birthday.month == month\n"
            "]"
        )
    },
    {
        "question": "List all employees who have been with the company for more than 5 years.",
        "code": (
            "cutoff = date(date.today().year - 5, date.today().month, date.today().day).strftime('%Y-%m-%d')\n"
            "result = self.env['hr.employee'].search_read(\n"
            "    [('active', '=', True), ('create_date', '<=', cutoff)],\n"
            "    ['name', 'department_id', 'job_id', 'create_date']\n"
            ")"
        )
    },

    # ══════════════════════════════════════════════════════════════════════════
    # MODULE 7 · MANUFACTURING  (mrp.production / mrp.bom / mrp.workorder)
    # ══════════════════════════════════════════════════════════════════════════

    {
        "question": "How many manufacturing orders are currently in progress?",
        "code": (
            "result = self.env['mrp.production'].search_count(\n"
            "    [('state', '=', 'progress')]\n"
            ")"
        )
    },
    {
        "question": "List all manufacturing orders that are overdue and not yet finished.",
        "code": (
            "today = date.today().strftime('%Y-%m-%d')\n"
            "result = self.env['mrp.production'].search_read(\n"
            "    [('state', 'not in', ['done', 'cancel']),\n"
            "     ('date_deadline', '<', today)],\n"
            "    ['name', 'product_id', 'product_qty', 'date_deadline', 'state']\n"
            ")"
        )
    },
    {
        "question": "What is the total quantity produced per product this month?",
        "code": (
            "today = date.today()\n"
            "first_day = today.replace(day=1).strftime('%Y-%m-%d')\n"
            "result = self.env['mrp.production'].read_group(\n"
            "    [('state', '=', 'done'), ('date_finished', '>=', first_day)],\n"
            "    ['product_id', 'qty_produced:sum'],\n"
            "    ['product_id']\n"
            ")"
        )
    },
    {
        "question": "Show all manufacturing orders with a component shortage (availability not confirmed).",
        "code": (
            "result = self.env['mrp.production'].search_read(\n"
            "    [('state', 'in', ['confirmed', 'progress']),\n"
            "     ('reservation_state', 'not in', ['assigned'])],\n"
            "    ['name', 'product_id', 'product_qty', 'reservation_state', 'date_start']\n"
            ")"
        )
    },
    {
        "question": "List all bills of materials (BOMs) for a product named 'Table'.",
        "code": (
            "product = self.env['product.template'].search([('name', 'ilike', 'Table')], limit=1)\n"
            "result = self.env['mrp.bom'].search_read(\n"
            "    [('product_tmpl_id', '=', product.id)],\n"
            "    ['product_tmpl_id', 'product_qty', 'type', 'bom_line_ids']\n"
            ") if product else 'Product not found'"
        )
    },
    {
        "question": "How many manufacturing orders were completed this week?",
        "code": (
            "monday = (date.today() - timedelta(days=date.today().weekday())).strftime('%Y-%m-%d')\n"
            "result = self.env['mrp.production'].search_count(\n"
            "    [('state', '=', 'done'), ('date_finished', '>=', monday)]\n"
            ")"
        )
    },
    {
        "question": "Show total manufacturing scrap quantity per product this year.",
        "code": (
            "year = date.today().year\n"
            "result = self.env['stock.scrap'].read_group(\n"
            "    [('production_id', '!=', False), ('state', '=', 'done'),\n"
            "     ('date_done', '>=', f'{year}-01-01')],\n"
            "    ['product_id', 'scrap_qty:sum'],\n"
            "    ['product_id'],\n"
            "    orderby='scrap_qty DESC'\n"
            ")"
        )
    },
    {
        "question": "What is the efficiency rate of manufacturing orders this month (qty produced vs qty planned)?",
        "code": (
            "today = date.today()\n"
            "first_day = today.replace(day=1).strftime('%Y-%m-%d')\n"
            "orders = self.env['mrp.production'].search(\n"
            "    [('state', '=', 'done'), ('date_finished', '>=', first_day)]\n"
            ")\n"
            "total_planned = sum(o.product_qty for o in orders)\n"
            "total_produced = sum(o.qty_produced for o in orders)\n"
            "result = {\n"
            "    'total_planned': total_planned,\n"
            "    'total_produced': total_produced,\n"
            "    'efficiency_percent': round((total_produced / total_planned) * 100, 2) if total_planned else 0\n"
            "}"
        )
    },
    {
        "question": "List all work orders currently waiting for an operator (state = ready).",
        "code": (
            "result = self.env['mrp.workorder'].search_read(\n"
            "    [('state', '=', 'ready')],\n"
            "    ['name', 'production_id', 'workcenter_id', 'date_start', 'duration_expected']\n"
            ")"
        )
    },
    {
        "question": "Show the total planned manufacturing quantity per product for next week.",
        "code": (
            "today = date.today()\n"
            "next_monday = (today + timedelta(days=(7 - today.weekday()))).strftime('%Y-%m-%d')\n"
            "next_sunday = (today + timedelta(days=(13 - today.weekday()))).strftime('%Y-%m-%d')\n"
            "result = self.env['mrp.production'].read_group(\n"
            "    [('state', 'not in', ['done', 'cancel']),\n"
            "     ('date_start', '>=', next_monday),\n"
            "     ('date_start', '<=', next_sunday)],\n"
            "    ['product_id', 'product_qty:sum'],\n"
            "    ['product_id']\n"
            ")"
        )
    },
    {
        "question": "Show the number of manufacturing orders by state.",
        "code": (
            "result = self.env['mrp.production'].read_group(\n"
            "    [],\n"
            "    ['state', 'id:count'],\n"
            "    ['state']\n"
            ")"
        )
    },
    {
        "question": "Which workcenter has the most work orders this week?",
        "code": (
            "monday = (date.today() - timedelta(days=date.today().weekday())).strftime('%Y-%m-%d')\n"
            "result = self.env['mrp.workorder'].read_group(\n"
            "    [('date_start', '>=', monday)],\n"
            "    ['workcenter_id', 'id:count'],\n"
            "    ['workcenter_id'],\n"
            "    orderby='workcenter_id_count DESC',\n"
            "    limit=5\n"
            ")"
        )
    },
    {
        "question": "List all manufacturing orders planned for today.",
        "code": (
            "today = date.today().strftime('%Y-%m-%d')\n"
            "result = self.env['mrp.production'].search_read(\n"
            "    [('state', 'not in', ['done', 'cancel']),\n"
            "     ('date_start', '>=', today)],\n"
            "    ['name', 'product_id', 'product_qty', 'date_start', 'state']\n"
            ")"
        )
    },
    {
        "question": "What is the average time to complete a manufacturing order in days this year?",
        "code": (
            "year = date.today().year\n"
            "orders = self.env['mrp.production'].search(\n"
            "    [('state', '=', 'done'), ('date_start', '!=', False),\n"
            "     ('date_finished', '!=', False), ('date_finished', '>=', f'{year}-01-01')]\n"
            ")\n"
            "durations = [(o.date_finished - o.date_start).days for o in orders if o.date_finished > o.date_start]\n"
            "result = {'avg_days': round(sum(durations) / len(durations), 1) if durations else 0, 'sample_size': len(durations)}"
        )
    },
    {
        "question": "List all manufacturing orders cancelled this month.",
        "code": (
            "today = date.today()\n"
            "first_day = today.replace(day=1).strftime('%Y-%m-%d')\n"
            "result = self.env['mrp.production'].search_read(\n"
            "    [('state', '=', 'cancel'), ('write_date', '>=', first_day)],\n"
            "    ['name', 'product_id', 'product_qty', 'write_date']\n"
            ")"
        )
    },

    # ══════════════════════════════════════════════════════════════════════════
    # MODULE 8 · PROJECT  (project.project / project.task)
    # ══════════════════════════════════════════════════════════════════════════

    {
        "question": "How many tasks are currently open across all projects?",
        "code": (
            "result = self.env['project.task'].search_count(\n"
            "    [('stage_id.fold', '=', False), ('active', '=', True)]\n"
            ")"
        )
    },
    {
        "question": "List all tasks assigned to me that are overdue.",
        "code": (
            "today = date.today().strftime('%Y-%m-%d')\n"
            "uid = self.env.uid\n"
            "result = self.env['project.task'].search_read(\n"
            "    [('user_ids', 'in', [uid]), ('date_deadline', '<', today),\n"
            "     ('stage_id.fold', '=', False)],\n"
            "    ['name', 'project_id', 'date_deadline', 'priority', 'stage_id']\n"
            ")"
        )
    },
    {
        "question": "Show tasks grouped by project and stage.",
        "code": (
            "result = self.env['project.task'].read_group(\n"
            "    [('active', '=', True), ('stage_id.fold', '=', False)],\n"
            "    ['project_id', 'stage_id', 'id:count'],\n"
            "    ['project_id', 'stage_id']\n"
            ")"
        )
    },
    {
        "question": "How many high-priority tasks are currently open?",
        "code": (
            "result = self.env['project.task'].search_count(\n"
            "    [('priority', '=', '1'), ('stage_id.fold', '=', False), ('active', '=', True)]\n"
            ")"
        )
    },
    {
        "question": "Which project has the most overdue tasks?",
        "code": (
            "today = date.today().strftime('%Y-%m-%d')\n"
            "result = self.env['project.task'].read_group(\n"
            "    [('date_deadline', '<', today), ('stage_id.fold', '=', False), ('active', '=', True)],\n"
            "    ['project_id', 'id:count'],\n"
            "    ['project_id'],\n"
            "    orderby='project_id_count DESC',\n"
            "    limit=5\n"
            ")"
        )
    },
    {
        "question": "List all tasks created this week with no assignee.",
        "code": (
            "monday = (date.today() - timedelta(days=date.today().weekday())).strftime('%Y-%m-%d')\n"
            "result = self.env['project.task'].search_read(\n"
            "    [('create_date', '>=', monday), ('user_ids', '=', False), ('active', '=', True)],\n"
            "    ['name', 'project_id', 'create_date', 'stage_id']\n"
            ")"
        )
    },
    {
        "question": "What is the total number of hours logged on a project named 'Website Redesign'?",
        "code": (
            "project = self.env['project.project'].search([('name', 'ilike', 'Website Redesign')], limit=1)\n"
            "result = self.env['account.analytic.line'].read_group(\n"
            "    [('project_id', '=', project.id)],\n"
            "    ['unit_amount:sum'], []\n"
            ") if project else 'Project not found'"
        )
    },
    {
        "question": "Show tasks completed this month grouped by project.",
        "code": (
            "today = date.today()\n"
            "first_day = today.replace(day=1).strftime('%Y-%m-%d')\n"
            "result = self.env['project.task'].read_group(\n"
            "    [('stage_id.fold', '=', True), ('write_date', '>=', first_day), ('active', '=', True)],\n"
            "    ['project_id', 'id:count'],\n"
            "    ['project_id']\n"
            ")"
        )
    },
    {
        "question": "List all projects that have a deadline this month.",
        "code": (
            "today = date.today()\n"
            "first_day = today.replace(day=1).strftime('%Y-%m-%d')\n"
            "last_day = today.replace(day=((today.replace(day=28) + timedelta(days=4)).replace(day=1) - timedelta(days=1)).day).strftime('%Y-%m-%d')\n"
            "result = self.env['project.project'].search_read(\n"
            "    [('date', '>=', first_day), ('date', '<=', last_day)],\n"
            "    ['name', 'date', 'partner_id', 'user_id']\n"
            ")"
        )
    },
    {
        "question": "Show the average number of tasks per project.",
        "code": (
            "projects = self.env['project.project'].search([])\n"
            "total_tasks = self.env['project.task'].search_count([('active', '=', True)])\n"
            "result = {\n"
            "    'total_projects': len(projects),\n"
            "    'total_tasks': total_tasks,\n"
            "    'avg_tasks_per_project': round(total_tasks / len(projects), 1) if projects else 0\n"
            "}"
        )
    },
    {
        "question": "Which employee has the most tasks assigned across all projects?",
        "code": (
            "result = self.env['project.task'].read_group(\n"
            "    [('active', '=', True), ('stage_id.fold', '=', False)],\n"
            "    ['user_ids', 'id:count'],\n"
            "    ['user_ids'],\n"
            "    orderby='user_ids_count DESC',\n"
            "    limit=5\n"
            ")"
        )
    },
    {
        "question": "List all tasks that have been in the same stage for more than 10 days.",
        "code": (
            "cutoff = (date.today() - timedelta(days=10)).strftime('%Y-%m-%d')\n"
            "result = self.env['project.task'].search_read(\n"
            "    [('write_date', '<=', cutoff), ('stage_id.fold', '=', False), ('active', '=', True)],\n"
            "    ['name', 'project_id', 'stage_id', 'write_date', 'user_ids']\n"
            ")"
        )
    },
    {
        "question": "Show total hours logged per employee this month across all projects.",
        "code": (
            "today = date.today()\n"
            "first_day = today.replace(day=1).strftime('%Y-%m-%d')\n"
            "result = self.env['account.analytic.line'].read_group(\n"
            "    [('project_id', '!=', False), ('date', '>=', first_day)],\n"
            "    ['employee_id', 'unit_amount:sum'],\n"
            "    ['employee_id'],\n"
            "    orderby='unit_amount DESC'\n"
            ")"
        )
    },
    {
        "question": "List all blocked tasks (kanban state = blocked).",
        "code": (
            "result = self.env['project.task'].search_read(\n"
            "    [('kanban_state', '=', 'blocked'), ('active', '=', True)],\n"
            "    ['name', 'project_id', 'stage_id', 'user_ids', 'kanban_state']\n"
            ")"
        )
    },
    {
        "question": "How many tasks were closed (moved to a folded stage) this week?",
        "code": (
            "monday = (date.today() - timedelta(days=date.today().weekday())).strftime('%Y-%m-%d')\n"
            "result = self.env['project.task'].search_count(\n"
            "    [('stage_id.fold', '=', True), ('write_date', '>=', monday), ('active', '=', True)]\n"
            ")"
        )
    },

    # ══════════════════════════════════════════════════════════════════════════
    # MODULE 9 · HELPDESK  (helpdesk.ticket)
    # ══════════════════════════════════════════════════════════════════════════

    {
        "question": "How many open helpdesk tickets are there right now?",
        "code": (
            "result = self.env['helpdesk.ticket'].search_count(\n"
            "    [('stage_id.fold', '=', False)]\n"
            ")"
        )
    },
    {
        "question": "List all urgent helpdesk tickets (priority = urgent) that are still open.",
        "code": (
            "result = self.env['helpdesk.ticket'].search_read(\n"
            "    [('priority', '=', '3'), ('stage_id.fold', '=', False)],\n"
            "    ['name', 'partner_id', 'user_id', 'stage_id', 'create_date']\n"
            ")"
        )
    },
    {
        "question": "Show helpdesk tickets grouped by stage.",
        "code": (
            "result = self.env['helpdesk.ticket'].read_group(\n"
            "    [],\n"
            "    ['stage_id', 'id:count'],\n"
            "    ['stage_id']\n"
            ")"
        )
    },
    {
        "question": "What is the average resolution time for closed tickets this month?",
        "code": (
            "today = date.today()\n"
            "first_day = today.replace(day=1).strftime('%Y-%m-%d')\n"
            "closed_tickets = self.env['helpdesk.ticket'].search(\n"
            "    [('stage_id.fold', '=', True), ('close_date', '>=', first_day),\n"
            "     ('create_date', '!=', False), ('close_date', '!=', False)]\n"
            ")\n"
            "durations = [(t.close_date - t.create_date).total_seconds() / 3600 for t in closed_tickets]\n"
            "result = {\n"
            "    'avg_resolution_hours': round(sum(durations) / len(durations), 1) if durations else 0,\n"
            "    'tickets_closed': len(durations)\n"
            "}"
        )
    },
    {
        "question": "List all helpdesk tickets that have been open for more than 5 days with no activity.",
        "code": (
            "cutoff = (date.today() - timedelta(days=5)).strftime('%Y-%m-%d')\n"
            "result = self.env['helpdesk.ticket'].search_read(\n"
            "    [('stage_id.fold', '=', False), ('create_date', '<=', cutoff),\n"
            "     ('activity_ids', '=', False)],\n"
            "    ['name', 'partner_id', 'user_id', 'stage_id', 'create_date', 'priority']\n"
            ")"
        )
    },
    {
        "question": "Show the number of tickets created per day this week.",
        "code": (
            "monday = (date.today() - timedelta(days=date.today().weekday())).strftime('%Y-%m-%d')\n"
            "result = self.env['helpdesk.ticket'].read_group(\n"
            "    [('create_date', '>=', monday)],\n"
            "    ['create_date:day', 'id:count'],\n"
            "    ['create_date:day']\n"
            ")"
        )
    },
    {
        "question": "Which support agent has closed the most tickets this month?",
        "code": (
            "today = date.today()\n"
            "first_day = today.replace(day=1).strftime('%Y-%m-%d')\n"
            "result = self.env['helpdesk.ticket'].read_group(\n"
            "    [('stage_id.fold', '=', True), ('close_date', '>=', first_day),\n"
            "     ('user_id', '!=', False)],\n"
            "    ['user_id', 'id:count'],\n"
            "    ['user_id'],\n"
            "    orderby='user_id_count DESC',\n"
            "    limit=5\n"
            ")"
        )
    },
    {
        "question": "Show all tickets associated with a specific customer named 'Deco Addict'.",
        "code": (
            "partner = self.env['res.partner'].search([('name', 'ilike', 'Deco Addict')], limit=1)\n"
            "result = self.env['helpdesk.ticket'].search_read(\n"
            "    [('partner_id', '=', partner.id)],\n"
            "    ['name', 'stage_id', 'priority', 'create_date', 'close_date', 'user_id']\n"
            ") if partner else 'Customer not found'"
        )
    },
    {
        "question": "How many tickets were resolved within the SLA this month?",
        "code": (
            "today = date.today()\n"
            "first_day = today.replace(day=1).strftime('%Y-%m-%d')\n"
            "result = self.env['helpdesk.ticket'].search_count(\n"
            "    [('stage_id.fold', '=', True), ('close_date', '>=', first_day),\n"
            "     ('sla_deadline', '!=', False)]\n"
            ")"
        )
    },
    {
        "question": "List all tickets that have breached SLA (deadline passed and still open).",
        "code": (
            "now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')\n"
            "result = self.env['helpdesk.ticket'].search_read(\n"
            "    [('stage_id.fold', '=', False),\n"
            "     ('sla_deadline', '<', now), ('sla_deadline', '!=', False)],\n"
            "    ['name', 'partner_id', 'user_id', 'sla_deadline', 'stage_id', 'priority']\n"
            ")"
        )
    },
    {
        "question": "Show helpdesk tickets grouped by team and priority.",
        "code": (
            "result = self.env['helpdesk.ticket'].read_group(\n"
            "    [('stage_id.fold', '=', False)],\n"
            "    ['team_id', 'priority', 'id:count'],\n"
            "    ['team_id', 'priority']\n"
            ")"
        )
    },
    {
        "question": "What percentage of tickets are currently unassigned?",
        "code": (
            "total = self.env['helpdesk.ticket'].search_count([('stage_id.fold', '=', False)])\n"
            "unassigned = self.env['helpdesk.ticket'].search_count(\n"
            "    [('stage_id.fold', '=', False), ('user_id', '=', False)]\n"
            ")\n"
            "result = {\n"
            "    'total_open': total,\n"
            "    'unassigned': unassigned,\n"
            "    'unassigned_percent': round((unassigned / total) * 100, 2) if total else 0\n"
            "}"
        )
    },
    {
        "question": "List all helpdesk tickets with a rating (CSAT) of 1 (unhappy) this month.",
        "code": (
            "today = date.today()\n"
            "first_day = today.replace(day=1).strftime('%Y-%m-%d')\n"
            "result = self.env['helpdesk.ticket'].search_read(\n"
            "    [('rating_last_value', '<=', 1), ('close_date', '>=', first_day)],\n"
            "    ['name', 'partner_id', 'user_id', 'rating_last_value', 'close_date']\n"
            ")"
        )
    },
    {
        "question": "Show the total number of tickets received per month this year.",
        "code": (
            "year = date.today().year\n"
            "result = self.env['helpdesk.ticket'].read_group(\n"
            "    [('create_date', '>=', f'{year}-01-01')],\n"
            "    ['create_date:month', 'id:count'],\n"
            "    ['create_date:month']\n"
            ")"
        )
    },
    {
        "question": "Which tickets have had the most messages/interactions (by message count)?",
        "code": (
            "tickets = self.env['helpdesk.ticket'].search_read(\n"
            "    [], ['name', 'partner_id', 'message_ids', 'stage_id']\n"
            ")\n"
            "result = sorted(\n"
            "    [{'name': t['name'], 'customer': t['partner_id'][1] if t['partner_id'] else 'N/A',\n"
            "      'message_count': len(t['message_ids'])} for t in tickets],\n"
            "    key=lambda x: x['message_count'], reverse=True\n"
            ")[:10]"
        )
    },

    # ══════════════════════════════════════════════════════════════════════════
    # MODULE 10 · POINT OF SALE  (pos.order / pos.order.line / pos.session)
    # ══════════════════════════════════════════════════════════════════════════

    {
        "question": "What is the total sales revenue from POS today?",
        "code": (
            "today = date.today().strftime('%Y-%m-%d')\n"
            "result = self.env['pos.order'].read_group(\n"
            "    [('state', 'in', ['paid', 'done', 'invoiced']), ('date_order', '>=', today)],\n"
            "    ['amount_total:sum'], []\n"
            ")"
        )
    },
    {
        "question": "Show the total POS revenue grouped by shop/POS config this month.",
        "code": (
            "today = date.today()\n"
            "first_day = today.replace(day=1).strftime('%Y-%m-%d')\n"
            "result = self.env['pos.order'].read_group(\n"
            "    [('state', 'in', ['paid', 'done', 'invoiced']), ('date_order', '>=', first_day)],\n"
            "    ['config_id', 'amount_total:sum'],\n"
            "    ['config_id']\n"
            ")"
        )
    },
    {
        "question": "Which products are the top 10 best sellers at POS this month by quantity?",
        "code": (
            "today = date.today()\n"
            "first_day = today.replace(day=1).strftime('%Y-%m-%d')\n"
            "result = self.env['pos.order.line'].read_group(\n"
            "    [('order_id.state', 'in', ['paid', 'done', 'invoiced']),\n"
            "     ('order_id.date_order', '>=', first_day)],\n"
            "    ['product_id', 'qty:sum', 'price_subtotal_incl:sum'],\n"
            "    ['product_id'],\n"
            "    orderby='qty DESC',\n"
            "    limit=10\n"
            ")"
        )
    },
    {
        "question": "How many POS orders were placed today?",
        "code": (
            "today = date.today().strftime('%Y-%m-%d')\n"
            "result = self.env['pos.order'].search_count(\n"
            "    [('state', 'in', ['paid', 'done', 'invoiced']), ('date_order', '>=', today)]\n"
            ")"
        )
    },
    {
        "question": "Show total POS revenue per cashier this week.",
        "code": (
            "monday = (date.today() - timedelta(days=date.today().weekday())).strftime('%Y-%m-%d')\n"
            "result = self.env['pos.order'].read_group(\n"
            "    [('state', 'in', ['paid', 'done', 'invoiced']), ('date_order', '>=', monday)],\n"
            "    ['user_id', 'amount_total:sum'],\n"
            "    ['user_id'],\n"
            "    orderby='amount_total DESC'\n"
            ")"
        )
    },
    {
        "question": "What is the average POS transaction value this month?",
        "code": (
            "today = date.today()\n"
            "first_day = today.replace(day=1).strftime('%Y-%m-%d')\n"
            "result = self.env['pos.order'].read_group(\n"
            "    [('state', 'in', ['paid', 'done', 'invoiced']), ('date_order', '>=', first_day)],\n"
            "    ['amount_total:avg'], []\n"
            ")"
        )
    },
    {
        "question": "Show total POS revenue by payment method this month.",
        "code": (
            "today = date.today()\n"
            "first_day = today.replace(day=1).strftime('%Y-%m-%d')\n"
            "result = self.env['pos.payment'].read_group(\n"
            "    [('pos_order_id.state', 'in', ['paid', 'done', 'invoiced']),\n"
            "     ('pos_order_id.date_order', '>=', first_day)],\n"
            "    ['payment_method_id', 'amount:sum'],\n"
            "    ['payment_method_id']\n"
            ")"
        )
    },
    {
        "question": "List all currently open POS sessions.",
        "code": (
            "result = self.env['pos.session'].search_read(\n"
            "    [('state', '=', 'opened')],\n"
            "    ['name', 'config_id', 'user_id', 'start_at']\n"
            ")"
        )
    },
    {
        "question": "How many POS orders were refunded this month?",
        "code": (
            "today = date.today()\n"
            "first_day = today.replace(day=1).strftime('%Y-%m-%d')\n"
            "result = self.env['pos.order'].search_count(\n"
            "    [('state', '=', 'invoiced'), ('amount_total', '<', 0),\n"
            "     ('date_order', '>=', first_day)]\n"
            ")"
        )
    },
    {
        "question": "Show the hourly sales breakdown for POS orders today.",
        "code": (
            "today = date.today().strftime('%Y-%m-%d')\n"
            "result = self.env['pos.order'].read_group(\n"
            "    [('state', 'in', ['paid', 'done', 'invoiced']), ('date_order', '>=', today)],\n"
            "    ['date_order:hour', 'amount_total:sum', 'id:count'],\n"
            "    ['date_order:hour']\n"
            ")"
        )
    },
    {
        "question": "Which product categories generate the most POS revenue this month?",
        "code": (
            "today = date.today()\n"
            "first_day = today.replace(day=1).strftime('%Y-%m-%d')\n"
            "result = self.env['pos.order.line'].read_group(\n"
            "    [('order_id.state', 'in', ['paid', 'done', 'invoiced']),\n"
            "     ('order_id.date_order', '>=', first_day)],\n"
            "    ['product_id.categ_id', 'price_subtotal_incl:sum'],\n"
            "    ['product_id.categ_id'],\n"
            "    orderby='price_subtotal_incl DESC'\n"
            ")"
        )
    },
    {
        "question": "What is the total tax collected at POS this month?",
        "code": (
            "today = date.today()\n"
            "first_day = today.replace(day=1).strftime('%Y-%m-%d')\n"
            "result = self.env['pos.order'].read_group(\n"
            "    [('state', 'in', ['paid', 'done', 'invoiced']), ('date_order', '>=', first_day)],\n"
            "    ['amount_tax:sum'], []\n"
            ")"
        )
    },
    {
        "question": "Show daily POS revenue trend for the last 30 days.",
        "code": (
            "cutoff = (date.today() - timedelta(days=30)).strftime('%Y-%m-%d')\n"
            "result = self.env['pos.order'].read_group(\n"
            "    [('state', 'in', ['paid', 'done', 'invoiced']), ('date_order', '>=', cutoff)],\n"
            "    ['date_order:day', 'amount_total:sum', 'id:count'],\n"
            "    ['date_order:day']\n"
            ")"
        )
    },
    {
        "question": "List the top 5 customers by total spend at POS.",
        "code": (
            "result = self.env['pos.order'].read_group(\n"
            "    [('state', 'in', ['paid', 'done', 'invoiced']), ('partner_id', '!=', False)],\n"
            "    ['partner_id', 'amount_total:sum'],\n"
            "    ['partner_id'],\n"
            "    orderby='amount_total DESC',\n"
            "    limit=5\n"
            ")"
        )
    },
    {
        "question": "What is the total discount given across all POS orders this month?",
        "code": (
            "today = date.today()\n"
            "first_day = today.replace(day=1).strftime('%Y-%m-%d')\n"
            "lines = self.env['pos.order.line'].search(\n"
            "    [('order_id.state', 'in', ['paid', 'done', 'invoiced']),\n"
            "     ('order_id.date_order', '>=', first_day),\n"
            "     ('discount', '>', 0)]\n"
            ")\n"
            "result = {\n"
            "    'total_discount': sum(\n"
            "        (l.price_unit * l.qty * l.discount / 100) for l in lines\n"
            "    )\n"
            "}"
        )
    },

    # ══════════════════════════════════════════════════════════════════════════
    # BONUS: CROSS-MODULE & UTILITY QUERIES
    # ══════════════════════════════════════════════════════════════════════════

    {
        "question": "Show me a dashboard summary: total open sales orders, unpaid invoices, pending deliveries, and open tickets.",
        "code": (
            "result = {\n"
            "    'open_sale_orders': self.env['sale.order'].search_count(\n"
            "        [('state', 'in', ['draft', 'sent', 'sale'])]),\n"
            "    'unpaid_invoices': self.env['account.move'].search_count(\n"
            "        [('move_type', '=', 'out_invoice'), ('state', '=', 'posted'),\n"
            "         ('payment_state', 'in', ['not_paid', 'partial'])]),\n"
            "    'pending_deliveries': self.env['stock.picking'].search_count(\n"
            "        [('picking_type_code', '=', 'outgoing'),\n"
            "         ('state', 'in', ['confirmed', 'assigned'])]),\n"
            "    'open_helpdesk_tickets': self.env['helpdesk.ticket'].search_count(\n"
            "        [('stage_id.fold', '=', False)]),\n"
            "}"
        )
    },
    {
        "question": "List all products that have been sold in sales orders but have zero stock.",
        "code": (
            "sold_product_ids = self.env['sale.order.line'].search(\n"
            "    [('order_id.state', 'in', ['sale', 'done'])]\n"
            ").mapped('product_id.id')\n"
            "products_with_stock = self.env['stock.quant'].search(\n"
            "    [('location_id.usage', '=', 'internal'), ('quantity', '>', 0)]\n"
            ").mapped('product_id.id')\n"
            "out_of_stock_sold = list(set(sold_product_ids) - set(products_with_stock))\n"
            "result = self.env['product.product'].search_read(\n"
            "    [('id', 'in', out_of_stock_sold)],\n"
            "    ['name', 'default_code', 'categ_id']\n"
            ")"
        )
    },
    {
        "question": "Show all partners (customers/vendors) created this year.",
        "code": (
            "year = date.today().year\n"
            "result = self.env['res.partner'].search_read(\n"
            "    [('create_date', '>=', f'{year}-01-01'), ('active', '=', True)],\n"
            "    ['name', 'email', 'phone', 'customer_rank', 'supplier_rank', 'create_date']\n"
            ")"
        )
    },
    {
        "question": "Show all products with a sales price lower than their cost price.",
        "code": (
            "products = self.env['product.template'].search(\n"
            "    [('type', '=', 'consu'), ('active', '=', True)]\n"
            ")\n"
            "result = [\n"
            "    {'name': p.name, 'sales_price': p.list_price, 'cost_price': p.standard_price,\n"
            "     'loss_margin': p.list_price - p.standard_price}\n"
            "    for p in products if p.list_price < p.standard_price\n"
            "]"
        )
    },
    {
        "question": "Which customers have both open invoices and open helpdesk tickets?",
        "code": (
            "customers_with_invoices = set(self.env['account.move'].search(\n"
            "    [('move_type', '=', 'out_invoice'), ('state', '=', 'posted'),\n"
            "     ('payment_state', 'in', ['not_paid', 'partial'])]\n"
            ").mapped('partner_id.id'))\n"
            "customers_with_tickets = set(self.env['helpdesk.ticket'].search(\n"
            "    [('stage_id.fold', '=', False), ('partner_id', '!=', False)]\n"
            ").mapped('partner_id.id'))\n"
            "both = customers_with_invoices & customers_with_tickets\n"
            "result = self.env['res.partner'].search_read(\n"
            "    [('id', 'in', list(both))],\n"
            "    ['name', 'email', 'phone']\n"
            ")"
        )
    },
    {
        "question": "What is the total number of records in each main business model?",
        "code": (
            "result = {\n"
            "    'sale_orders': self.env['sale.order'].search_count([]),\n"
            "    'invoices': self.env['account.move'].search_count([('move_type', '=', 'out_invoice')]),\n"
            "    'purchase_orders': self.env['purchase.order'].search_count([]),\n"
            "    'stock_pickings': self.env['stock.picking'].search_count([]),\n"
            "    'crm_leads': self.env['crm.lead'].search_count([]),\n"
            "    'employees': self.env['hr.employee'].search_count([('active', '=', True)]),\n"
            "    'products': self.env['product.template'].search_count([('active', '=', True)]),\n"
            "    'customers': self.env['res.partner'].search_count([('customer_rank', '>', 0)]),\n"
            "}"
        )
    },
    {
        "question": "List all customers who have both purchased and have open invoices but never submitted a support ticket.",
        "code": (
            "customers_with_orders = set(self.env['sale.order'].search(\n"
            "    [('state', 'in', ['sale', 'done'])]\n"
            ").mapped('partner_id.id'))\n"
            "customers_with_invoices = set(self.env['account.move'].search(\n"
            "    [('move_type', '=', 'out_invoice'), ('state', '=', 'posted')]\n"
            ").mapped('partner_id.id'))\n"
            "customers_with_tickets = set(self.env['helpdesk.ticket'].search([]).mapped('partner_id.id'))\n"
            "qualified = (customers_with_orders & customers_with_invoices) - customers_with_tickets\n"
            "result = self.env['res.partner'].search_read(\n"
            "    [('id', 'in', list(qualified))],\n"
            "    ['name', 'email', 'phone']\n"
            ")"
        )
    },
    {
        "question": "Show all products that appear in both active sale orders and active purchase orders.",
        "code": (
            "sold_products = set(self.env['sale.order.line'].search(\n"
            "    [('order_id.state', 'in', ['sale', 'done'])]\n"
            ").mapped('product_id.id'))\n"
            "purchased_products = set(self.env['purchase.order.line'].search(\n"
            "    [('order_id.state', 'in', ['purchase', 'done'])]\n"
            ").mapped('product_id.id'))\n"
            "common = sold_products & purchased_products\n"
            "result = self.env['product.product'].search_read(\n"
            "    [('id', 'in', list(common))],\n"
            "    ['name', 'default_code', 'categ_id']\n"
            ")"
        )
    },
]

def cosine_sim(a, b):
    dot_product = sum(x * y for x, y in zip(a, b))
    mag_a = math.sqrt(sum(x * x for x in a))
    mag_b = math.sqrt(sum(y * y for y in b))
    if mag_a * mag_b == 0: return 0
    return dot_product / (mag_a * mag_b)

_CACHED_EXAMPLE_EMBS = None

def get_db_mode_prompt(question, embeddings_model, top_k=2):
    """
    Builds the full Prompt Template including rules, history, and dynamically 
    selected top-K few-shot examples using vector similarity.
    """
    global _CACHED_EXAMPLE_EMBS
    # 1. Embed query
    query_emb = embeddings_model.embed_query(question)
    
    # 2. Embed examples
    example_questions = [ex["question"] for ex in FEW_SHOT_EXAMPLES]
    try:
        if _CACHED_EXAMPLE_EMBS is None:
            _CACHED_EXAMPLE_EMBS = embeddings_model.embed_documents(example_questions)
        example_embs = _CACHED_EXAMPLE_EMBS
    except Exception:
        # Fallback if embedding fails
        example_embs = None

    selected_examples = []
    if example_embs:
        # 3. Score examples
        scored_examples = []
        for i, ex in enumerate(FEW_SHOT_EXAMPLES):
            sim = cosine_sim(query_emb, example_embs[i])
            scored_examples.append((sim, ex))
        
        # 4. Sort and pick top K
        scored_examples.sort(key=lambda x: x[0], reverse=True)
        selected_examples = [ex for score, ex in scored_examples[:top_k]]
    else:
        # Fallback to first K
        selected_examples = FEW_SHOT_EXAMPLES[:top_k]

    # Build the dynamic list of Few-Shot tuples
    dynamic_messages = [("system", DB_MODE_SYSTEM_TEMPLATE)]
    for ex in selected_examples:
        q_escaped = ex["question"].replace("{", "{{").replace("}", "}}")
        code_escaped = ex["code"].replace("{", "{{").replace("}", "}}")
        dynamic_messages.append(("human", q_escaped))
        dynamic_messages.append(("ai", f"```python\n{code_escaped}\n```"))
    
    dynamic_messages.append(MessagesPlaceholder(variable_name="history"))
    dynamic_messages.append(("human", "{question}"))

    return ChatPromptTemplate.from_messages(dynamic_messages)
