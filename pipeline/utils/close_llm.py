import json

from dotenv import load_dotenv
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

try:
    from pipeline.utils.llm import (
        BaseAsyncReviewer,
        check_output,
        default_system_prompt,
        expand_experiment_rows,
        fuzzy_match_option,
        parse_rating_response,
        rating_schema,
        rating_system_prompt,
    )
except ModuleNotFoundError:
    from llm import (
        BaseAsyncReviewer,
        check_output,
        default_system_prompt,
        expand_experiment_rows,
        fuzzy_match_option,
        parse_rating_response,
        rating_schema,
        rating_system_prompt,
    )

from llm_utils.wrapper import LLMWrapper

load_dotenv()


class ClosedLLMReviewer(BaseAsyncReviewer):
    """LLM Reviewer using llm-utils package for closed-source models."""

    def __init__(
        self,
        llm: str = "gpt-4o",
        n_workers: int = 10,
        verbosity: str = "low",
        reasoning: str = "minimal",
    ):
        super().__init__(llm)
        self.n_workers = n_workers
        self.verbosity = verbosity
        self.reasoning = reasoning
        self.wrapper = LLMWrapper()
        print(f"ClosedLLMReviewer initialized for model: {self.llm}")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        return False

    def _get_extra_body(self) -> dict | None:
        return None

    def _get_extra_headers(self) -> dict | None:
        return None

    def _build_response_format(self, options: list[str]) -> dict:
        """Build JSON schema response format for structured output."""
        return {
            "name": "review",
            "type": "json_schema",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "choice": {
                        "type": "string",
                        "enum": options,
                    }
                },
                "additionalProperties": False,
                "required": ["choice"],
            },
        }

    def _build_rating_response_format(self) -> dict:
        """Build JSON schema response format for rating output."""
        return {
            "name": "review",
            "type": "json_schema",
            "strict": True,
            "schema": rating_schema(),
        }

    def _parse_wrapper_result(self, result, options: list[str]) -> str | None:
        if isinstance(result, dict):
            if "choice" in result and isinstance(result["choice"], str):
                return fuzzy_match_option(result["choice"], options)
            if "error" in result:
                return None
            try:
                return self._parse_response(json.dumps(result), options)
            except TypeError:
                return None
        return self._parse_response(str(result), options)

    def review(
        self,
        experiments,
        max_concurrent: int = 10,
        top_n: int = 2,
        system_prompt: str | None = None,
        secondary_prompt: bool = False,
    ):
        """Review experiments using llm-utils LLMWrapper with batch processing."""
        del max_concurrent
        del top_n

        prompt_text = system_prompt or (
            rating_system_prompt() if secondary_prompt else default_system_prompt()
        )
        rows = [
            row
            for experiment in experiments
            for row in expand_experiment_rows(experiment, secondary_prompt=secondary_prompt)
        ]

        ids: list[str] = []
        user_messages: list[str] = []
        response_formats: list[dict] = []
        row_map: dict[str, dict] = {}

        for row in rows:
            row_id = (
                f"{row['experiment_id']}::"
                f"{row['variant_idx']}::"
                f"{int(row['ascending'])}"
            )
            ids.append(row_id)
            user_messages.append(row["prompt"])
            if secondary_prompt:
                response_formats.append(self._build_rating_response_format())
            else:
                response_formats.append(self._build_response_format(row["options"]))
            row_map[row_id] = row

        try:
            responses = self.wrapper.ask(
                system_message=prompt_text,
                ids=ids,
                user_messages=user_messages,
                response_formats=response_formats,
                model=self.llm,
                n_workers=self.n_workers,
                verbosity=self.verbosity,
                reasoning=self.reasoning,
            )
        except Exception as exc:
            print(f"Batch request failed: {exc}")
            return [
                {
                    "experiment_id": row["experiment_id"],
                    "variant_idx": row["variant_idx"],
                    "ascending": row["ascending"],
                    "choice": None,
                    "picked_perturbed": None,
                }
                for row in rows
            ]

        results = []
        for row_id in ids:
            row = row_map[row_id]
            result = responses.get(row_id, {})

            if isinstance(result, dict) and "error" in result:
                print(f"Error for {row_id}: {result['error']}")
                matched_choice = None
            else:
                if secondary_prompt:
                    if isinstance(result, dict):
                        rating = parse_rating_response(json.dumps(result))
                    else:
                        rating = parse_rating_response(str(result))
                    matched_choice = rating
                else:
                    matched_choice = self._parse_wrapper_result(result, row["options"])

            if matched_choice is None:
                results.append(
                    {
                        "experiment_id": row["experiment_id"],
                        "variant_idx": row["variant_idx"],
                        "ascending": row["ascending"],
                        "choice": None,
                        "picked_perturbed": None,
                    }
                )
                continue

            if secondary_prompt:
                results.append(
                    {
                        "experiment_id": row["experiment_id"],
                        "variant_idx": row["variant_idx"],
                        "ascending": row["ascending"],
                        "choice": matched_choice,
                        "picked_perturbed": matched_choice,
                    }
                )
                continue

            positive = check_output(
                row["experiment"],
                matched_choice,
                row["options"],
                row["variant_idx"],
            )
            results.append(
                {
                    "experiment_id": row["experiment_id"],
                    "variant_idx": row["variant_idx"],
                    "ascending": row["ascending"],
                    "choice": matched_choice,
                    "picked_perturbed": positive,
                }
            )

        return results


def main():
    """Test ClosedLLMReviewer with a simple prompt."""
    import argparse

    parser = argparse.ArgumentParser(description="Test llm-utils connection")
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4o-mini",
        help="Model to use (default: gpt-4o-mini)",
    )
    args = parser.parse_args()

    print(f"Testing llm-utils with model: {args.model}")

    wrapper = LLMWrapper()
    response = wrapper.ask(
        system_message="You are a helpful assistant. Respond briefly.",
        ids=["test"],
        user_messages=["What is the capital of France?"],
        response_formats=[None],
        model=args.model,
        verbosity="low",
    )

    print(f"Response: {response.get('test')}")
    print("Test completed successfully!")


if __name__ == "__main__":
    main()
