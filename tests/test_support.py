import importlib.util
import sys
import types
from datetime import date, datetime
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = REPO_ROOT / "ask_odoo" / "models"


def _ensure_module(name, module):
    if name not in sys.modules:
        sys.modules[name] = module
    return sys.modules[name]


def install_stubs():
    odoo = _ensure_module("odoo", types.ModuleType("odoo"))
    odoo_models = types.ModuleType("odoo.models")
    odoo_models.Model = object
    _ensure_module("odoo.models", odoo_models)

    def _field(*args, **kwargs):
        return None

    class _DateTime:
        @staticmethod
        def now():
            return datetime(2026, 1, 1, 0, 0, 0)

    class _Date:
        @staticmethod
        def today():
            return date(2026, 1, 1)

    odoo_fields = types.ModuleType("odoo.fields")
    odoo_fields.Char = _field
    odoo_fields.Text = _field
    odoo_fields.Many2one = _field
    odoo_fields.One2many = _field
    odoo_fields.Boolean = _field
    odoo_fields.Selection = _field
    odoo_fields.Binary = _field
    odoo_fields.Datetime = _DateTime
    odoo_fields.Date = _Date
    _ensure_module("odoo.fields", odoo_fields)

    odoo_api = types.ModuleType("odoo.api")
    odoo_api.model = lambda f: f
    _ensure_module("odoo.api", odoo_api)

    odoo.models = odoo_models
    odoo.fields = odoo_fields
    odoo.api = odoo_api

    odoo_tools = _ensure_module("odoo.tools", types.ModuleType("odoo.tools"))
    odoo_tools.config = types.SimpleNamespace(get=lambda *_args, **_kwargs: None)

    safe_eval_mod = types.ModuleType("odoo.tools.safe_eval")
    safe_eval_mod.safe_eval = lambda *args, **kwargs: None
    _ensure_module("odoo.tools.safe_eval", safe_eval_mod)

    lc_prompts = types.ModuleType("langchain_core.prompts")

    class MessagesPlaceholder:
        def __init__(self, variable_name):
            self.variable_name = variable_name

    class ChatPromptTemplate:
        def __init__(self, messages):
            self.messages = messages

        @classmethod
        def from_messages(cls, messages):
            return cls(messages)

        def format_messages(self, **kwargs):
            return kwargs

    lc_prompts.MessagesPlaceholder = MessagesPlaceholder
    lc_prompts.ChatPromptTemplate = ChatPromptTemplate
    _ensure_module("langchain_core.prompts", lc_prompts)

    lc_messages = types.ModuleType("langchain_core.messages")

    class _Msg:
        def __init__(self, content):
            self.content = content

    lc_messages.HumanMessage = _Msg
    lc_messages.AIMessage = _Msg
    _ensure_module("langchain_core.messages", lc_messages)

    lc_docs = types.ModuleType("langchain_core.documents")

    class Document:
        def __init__(self, page_content="", metadata=None):
            self.page_content = page_content
            self.metadata = metadata or {}

    lc_docs.Document = Document
    _ensure_module("langchain_core.documents", lc_docs)

    splitters = types.ModuleType("langchain_text_splitters")
    splitters.CharacterTextSplitter = type("CharacterTextSplitter", (), {})
    splitters.RecursiveCharacterTextSplitter = type("RecursiveCharacterTextSplitter", (), {})
    _ensure_module("langchain_text_splitters", splitters)

    pg = types.ModuleType("langchain_postgres.vectorstores")
    pg.PGVector = type("PGVector", (), {})
    _ensure_module("langchain_postgres.vectorstores", pg)

    groq = types.ModuleType("langchain_groq")
    groq.ChatGroq = type("ChatGroq", (), {})
    _ensure_module("langchain_groq", groq)

    hf = types.ModuleType("langchain_huggingface")
    hf.HuggingFaceEmbeddings = type("HuggingFaceEmbeddings", (), {})
    _ensure_module("langchain_huggingface", hf)

    loaders = types.ModuleType("langchain_community.document_loaders")
    loaders.PyPDFLoader = type("PyPDFLoader", (), {})
    _ensure_module("langchain_community.document_loaders", loaders)

    dotenv = types.ModuleType("dotenv")
    dotenv.load_dotenv = lambda: None
    _ensure_module("dotenv", dotenv)

    pandas = types.ModuleType("pandas")
    pandas.DataFrame = list
    pandas.to_numeric = lambda value, errors=None: value
    pandas.api = types.SimpleNamespace(
        types=types.SimpleNamespace(
            is_numeric_dtype=lambda _v: False,
            is_object_dtype=lambda _v: False,
            is_string_dtype=lambda _v: False,
        )
    )
    _ensure_module("pandas", pandas)

    ask_odoo_pkg = types.ModuleType("ask_odoo")
    ask_odoo_pkg.__path__ = [str(REPO_ROOT / "ask_odoo")]
    _ensure_module("ask_odoo", ask_odoo_pkg)

    ask_odoo_models_pkg = types.ModuleType("ask_odoo.models")
    ask_odoo_models_pkg.__path__ = [str(MODELS_DIR)]
    _ensure_module("ask_odoo.models", ask_odoo_models_pkg)


def load_module(module_name, file_name):
    install_stubs()
    path = MODELS_DIR / file_name
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module
