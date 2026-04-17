import unittest

from tests.test_support import load_module


prompts = load_module("ask_odoo.models.prompts", "prompts.py")


class FakeEmbeddings:
    def __init__(self, query_vec, doc_vecs, fail_docs=False):
        self.query_vec = query_vec
        self.doc_vecs = doc_vecs
        self.fail_docs = fail_docs

    def embed_query(self, _question):
        return self.query_vec

    def embed_documents(self, questions):
        if self.fail_docs:
            raise RuntimeError("embedding failure")
        return [self.doc_vecs[q] for q in questions]


class PromptsTests(unittest.TestCase):
    def test_cosine_sim_returns_zero_when_any_vector_magnitude_is_zero(self):
        self.assertEqual(prompts.cosine_sim([0, 0], [1, 2]), 0)

    def test_get_db_mode_prompt_selects_top_k_examples_by_similarity(self):
        example_questions = [ex["question"] for ex in prompts.FEW_SHOT_EXAMPLES]
        doc_vecs = {q: [0.0, 1.0] for q in example_questions}
        doc_vecs[example_questions[2]] = [1.0, 0.0]
        doc_vecs[example_questions[4]] = [0.8, 0.2]
        emb = FakeEmbeddings(query_vec=[1.0, 0.0], doc_vecs=doc_vecs)

        template = prompts.get_db_mode_prompt("group sales by user", emb, top_k=2)
        selected_humans = [m[1] for m in template.messages if isinstance(m, tuple) and m[0] == "human"][:2]

        self.assertEqual(selected_humans, [example_questions[2], example_questions[4]])

    def test_get_db_mode_prompt_falls_back_to_first_examples_when_embedding_fails(self):
        example_questions = [ex["question"] for ex in prompts.FEW_SHOT_EXAMPLES]
        emb = FakeEmbeddings(query_vec=[1.0], doc_vecs={}, fail_docs=True)

        template = prompts.get_db_mode_prompt("anything", emb, top_k=2)
        selected_humans = [m[1] for m in template.messages if isinstance(m, tuple) and m[0] == "human"][:2]

        self.assertEqual(selected_humans, example_questions[:2])


if __name__ == "__main__":
    unittest.main()
