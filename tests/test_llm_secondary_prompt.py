import asyncio
import unittest
from unittest.mock import patch

from pipeline.utils.llm import BaseAsyncReviewer, expand_experiment_rows, parse_rating_response


class FakeExperiment:
    experiment_id = 123
    variants = [object(), object()]

    def __init__(self):
        self.calls = []

    def prompt(self, variant_idx, ascending=True, secondary_prompt=False):
        self.calls.append(
            {
                "variant_idx": variant_idx,
                "ascending": ascending,
                "secondary_prompt": secondary_prompt,
            }
        )
        return f"prompt {variant_idx} {ascending} {secondary_prompt}"


class FakeReviewer(BaseAsyncReviewer):
    def _get_extra_body(self):
        return None

    def _get_extra_headers(self):
        return None


class FakeMessage:
    def __init__(self, content):
        self.content = content


class FakeChoice:
    def __init__(self, content):
        self.message = FakeMessage(content)


class FakeResponse:
    model = "fake-model"

    def __init__(self, content):
        self.choices = [FakeChoice(content)]


class FakeCompletions:
    def __init__(self, content):
        self.content = content
        self.kwargs = None

    async def create(self, **kwargs):
        self.kwargs = kwargs
        return FakeResponse(self.content)


class FakeChat:
    def __init__(self, content):
        self.completions = FakeCompletions(content)


class FakeClient:
    def __init__(self, content):
        self.chat = FakeChat(content)


class LLMSecondaryPromptTests(unittest.TestCase):
    def test_parse_rating_response_accepts_valid_ratings(self):
        self.assertEqual(parse_rating_response('{"rating": 1}'), 1.0)
        self.assertEqual(parse_rating_response('{"rating": 10}'), 10.0)
        self.assertEqual(parse_rating_response("7"), 7.0)
        self.assertEqual(parse_rating_response('{"rating": 3\n \n \n '), 3.0)

    def test_parse_rating_response_rejects_invalid_ratings(self):
        self.assertIsNone(parse_rating_response("0"))
        self.assertIsNone(parse_rating_response("11"))
        self.assertIsNone(parse_rating_response(""))
        self.assertIsNone(parse_rating_response("seven"))
        self.assertIsNone(parse_rating_response('{"choice": "A"}'))

    def test_expand_experiment_rows_secondary_prompt_only_uses_ascending_true(self):
        experiment = FakeExperiment()
        with patch("pipeline.utils.llm.build_options", return_value=[]):
            rows = expand_experiment_rows(experiment, secondary_prompt=True)

        self.assertEqual(len(rows), len(experiment.variants))
        self.assertTrue(all(row["ascending"] is True for row in rows))
        self.assertTrue(all(call["ascending"] is True for call in experiment.calls))
        self.assertTrue(all(call["secondary_prompt"] is True for call in experiment.calls))

    def test_process_prompt_row_secondary_prompt_returns_rating(self):
        reviewer = FakeReviewer("fake-model")
        client = FakeClient('{"rating": 8}')
        row = {
            "experiment": FakeExperiment(),
            "experiment_id": 1,
            "variant_idx": 2,
            "ascending": True,
            "prompt": "Rate this applicant.",
            "options": ["unused"],
        }

        result = asyncio.run(
            reviewer._process_prompt_row(
                client,
                row,
                asyncio.Semaphore(1),
                system_prompt=None,
                secondary_prompt=True,
            )
        )

        self.assertEqual(result["choice"], 8.0)
        self.assertEqual(result["picked_perturbed"], 8.0)
        self.assertTrue(result["ascending"])
        schema = client.chat.completions.kwargs["response_format"]["json_schema"]["schema"]
        self.assertIn("rating", schema["properties"])


if __name__ == "__main__":
    unittest.main()
