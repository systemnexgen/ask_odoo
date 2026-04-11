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

If the schema does not contain the fields needed or if the question is unrelated to the database schema below, clearly state that you cannot execute it instead of hallucinating code.

SCHEMA:
{context}"""

# Hand-crafted examples of Odoo ORM expressions mapped to natural language queries.
FEW_SHOT_EXAMPLES = [
    {
        "question": "Show me the total number of active users.",
        "code": "result = self.env['res.users'].search_count([('active', '=', True)])"
    },
    {
        "question": "What are the names and creation dates of the top 5 largest sale orders by amount?",
        "code": "result = self.env['sale.order'].search_read([('state', 'in', ['sale', 'done'])], ['name', 'create_date', 'amount_total'], order='amount_total DESC', limit=5)"
    },
    {
        "question": "Show total sales grouped by sales person.",
        "code": "result = self.env['sale.order'].read_group([('state', 'in', ['sale', 'done'])], ['user_id', 'amount_total:sum'], ['user_id'])"
    },
    {
        "question": "List all customers who don't have an email address.",
        "code": "result = self.env['res.partner'].search_read([('email', '=', False)], ['name', 'phone'])"
    },
    {
        "question": "Get the email of the administrator.",
        "code": "admin = self.env['res.users'].search([('login', '=', 'admin')], limit=1)\nresult = admin.email if admin else 'Not found'"
    },
    
]

def cosine_sim(a, b):
    dot_product = sum(x * y for x, y in zip(a, b))
    mag_a = math.sqrt(sum(x * x for x in a))
    mag_b = math.sqrt(sum(y * y for y in b))
    if mag_a * mag_b == 0: return 0
    return dot_product / (mag_a * mag_b)

def get_db_mode_prompt(question, embeddings_model, top_k=2):
    """
    Builds the full Prompt Template including rules, history, and dynamically 
    selected top-K few-shot examples using vector similarity.
    """
    # 1. Embed query
    query_emb = embeddings_model.embed_query(question)
    
    # 2. Embed examples
    example_questions = [ex["question"] for ex in FEW_SHOT_EXAMPLES]
    try:
        example_embs = embeddings_model.embed_documents(example_questions)
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
        dynamic_messages.append(("human", ex["question"]))
        dynamic_messages.append(("ai", f"```python\n{ex['code']}\n```"))
    
    dynamic_messages.append(MessagesPlaceholder(variable_name="history"))
    dynamic_messages.append(("human", "{question}"))

    return ChatPromptTemplate.from_messages(dynamic_messages)
