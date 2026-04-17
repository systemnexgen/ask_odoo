import unittest

from tests.test_support import load_module


schema = load_module("ask_odoo.models.schema", "schema.py")


class FakeEmbeddings:
    def __init__(self, query_vec, doc_vecs, fail=False):
        self.query_vec = query_vec
        self.doc_vecs = doc_vecs
        self.fail = fail

    def embed_query(self, _question):
        return self.query_vec

    def embed_documents(self, field_strings):
        if self.fail:
            raise RuntimeError("boom")
        return [self.doc_vecs[s] for s in field_strings]


class SchemaHelperTests(unittest.TestCase):
    def setUp(self):
        self.model = schema.AskOdooModel()

    def test_vector_shortlist_fields_keeps_id_and_best_matches(self):
        metadata = {
            "model": "sale.order",
            "fields": [
                {"name": "id", "string": "ID", "type": "integer"},
                {"name": "name", "string": "Order Ref", "type": "char"},
                {"name": "amount_total", "string": "Total", "type": "float"},
                {"name": "state", "string": "Status", "type": "selection"},
            ],
        }
        strings = {
            "Order Ref (name): char": [0.0, 1.0],
            "Total (amount_total): float": [1.0, 0.0],
            "Status (state): selection": [0.6, 0.4],
        }
        emb = FakeEmbeddings(query_vec=[1.0, 0.0], doc_vecs=strings)

        result = self.model._vector_shortlist_fields("total sales", metadata, emb, top_n=2)
        names = [f["name"] for f in result["fields"]]

        self.assertEqual(names, ["id", "amount_total"])

    def test_vector_shortlist_fields_returns_original_metadata_when_embedding_fails(self):
        metadata = {
            "model": "res.partner",
            "fields": [{"name": "name", "string": "Name", "type": "char"}],
        }
        emb = FakeEmbeddings(query_vec=[1.0], doc_vecs={}, fail=True)

        result = self.model._vector_shortlist_fields("partner name", metadata, emb, top_n=1)

        self.assertEqual(result, metadata)

    def test_schema_to_text_includes_field_relations_and_names_list(self):
        metadata = {
            "name": "Sales Order",
            "model": "sale.order",
            "fields": [
                {"name": "name", "string": "Order Reference", "type": "char"},
                {"name": "partner_id", "string": "Customer", "type": "many2one", "relation": "res.partner"},
            ],
        }

        text = self.model._schema_to_text(metadata)

        self.assertIn("Model: Sales Order (sale.order)", text)
        self.assertIn("All field names: name, partner_id", text)
        self.assertIn("- Customer (partner_id): many2one -> res.partner", text)


if __name__ == "__main__":
    unittest.main()
