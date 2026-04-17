import unittest

from tests.test_support import load_module


ask_odoo_model = load_module("ask_odoo.models.ask_odoo_model", "ask_odoo_model.py")


class AskOdooModelRoutingTests(unittest.TestCase):
    def test_process_message_routes_to_rag_for_conversation_mode(self):
        model = ask_odoo_model.AskOdooModel()
        model.process_message_rag = lambda message, conversation_id, chat_mode: {
            "handler": "rag",
            "message": message,
            "conversation_id": conversation_id,
            "chat_mode": chat_mode,
        }
        model.process_message_db = lambda message, conversation_id: {"handler": "db"}

        result = model.process_message("hello", conversation_id=12, chat_mode="conversation")

        self.assertEqual(result["handler"], "rag")
        self.assertEqual(result["conversation_id"], 12)
        self.assertEqual(result["chat_mode"], "conversation")

    def test_process_message_routes_to_db_for_non_conversation_mode(self):
        model = ask_odoo_model.AskOdooModel()
        model.process_message_rag = lambda *_args, **_kwargs: {"handler": "rag"}
        model.process_message_db = lambda message, conversation_id: {
            "handler": "db",
            "message": message,
            "conversation_id": conversation_id,
        }

        result = model.process_message("hello", conversation_id=99, chat_mode="database")

        self.assertEqual(result["handler"], "db")
        self.assertEqual(result["conversation_id"], 99)


if __name__ == "__main__":
    unittest.main()
