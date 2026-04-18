"""Shared utility helpers for the ask_odoo module."""
from odoo.tools import config


def get_pg_connection_string(env):
    """Build a SQLAlchemy connection string from Odoo's DB config.

    Centralised here to avoid duplicating the logic across
    llm.py and knowledge_base.py.
    """
    db_name = env.cr.dbname
    user = config.get('db_user')
    password = config.get('db_password')
    host = config.get('db_host') or 'localhost'
    port = config.get('db_port') or 5432
    return f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{db_name}"
