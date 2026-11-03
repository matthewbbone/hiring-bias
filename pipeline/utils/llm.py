import asyncio
import json
import re
from abc import ABC, abstractmethod
from typing import Any

from openai import AsyncOpenAI, RateLimitError
from tqdm.asyncio import tqdm


def sanitize_option(s: str) -> str:
    """Sanitize option string for JSON schema enum (no newlines, normalize quotes)."""
    return s.replace("\n", " ").replace("\r", " ").replace('"', "'").strip()


def normalize_for_comparison(s: str) -> str:
    """Normalize string for fuzzy comparison (lowercase, collapse whitespace)."""
    return re.sub(r"\s+", " ", s.lower().strip())


def fuzzy_match_option(pick: str, options: list[str]) -> str | None:
    """Find the best matching option for a picked value, or None if no match."""
    pick_normalized = normalize_for_comparison(pick)

    for option in options:
        if pick == option:
            return option

    for option in options:
        if normalize_for_comparison(option) == pick_normalized:
            return option

    for option in options:
        option_normalized = normalize_for_comparison(option)
        if pick_normalized in option_normalized or option_normalized in pick_normalized:
            return option

    return None


def default_system_prompt() -> str:
    return (
        "Pick the single best choice from the options provided. "
        'Respond with ONLY a JSON object in this exact format: {"choice": "Option"}'
    )


def plain_text_system_prompt() -> str:
    return (
        "Pick the single best choice from the options provided. "
        "Respond with ONLY the exact chosen option text and nothing else."
    )


def rating_system_prompt() -> str:
    return (
        "Rate the suitability on a 1 to 10 scale. "
        'Respond with ONLY a JSON object in this exact format: {"rating": int}'
    )


def plain_text_rating_system_prompt() -> str:
    return (
        "Rate the suitability on a 1 to 10 scale. "
        "Respond with ONLY a single integer from 1 to 10 and nothing else."
    )


def _iter_ordered_values(items: dict[int, Any]) -> list[Any]:
    return [items[idx] for idx in range(len(items))]


def build_options(experiment, variant_idx: int) -> list[str]:
    """Build the exact option strings shown in the prompt for one prompt variant."""
    from pipeline.utils.experiments import ApplyingExperiment, HiringExperiment

    if isinstance(experiment, HiringExperiment):
        options: list[str] = []
        for idx in range(len(experiment.profiles)):
            profile = experiment.profiles[idx]
            if idx == experiment.perturbed_idx:
                profile = experiment.variants[variant_idx]
            options.append(sanitize_option(f"{profile.first_name} {profile.last_name}"))
        return options

    if isinstance(experiment, ApplyingExperiment):
        return [
            sanitize_option(f"{vacancy.title} at {vacancy.company_name}")
            for vacancy in _iter_ordered_values(experiment.vacancies)
        ]

    raise ValueError("Unknown experiment type")


def expand_experiment_rows(experiment, secondary_prompt: bool = False) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for variant_idx in range(len(experiment.variants)):
        ascending_values = (True,) if secondary_prompt else (True, False)
        for ascending in ascending_values:
            rows.append(
                {
                    "experiment": experiment,
                    "experiment_id": experiment.experiment_id,
                    "variant_idx": variant_idx,
                    "ascending": ascending,
                    "prompt": experiment.prompt(
                        variant_idx=variant_idx,
                        ascending=ascending,
                        secondary_prompt=secondary_prompt,
                    ),
                    "options": build_options(experiment, variant_idx),
                }
            )
    return rows


def check_output(
    experiment,
    choice: str,
    options: list[str],
    variant_idx: int,
) -> bool | None:
    """Check if the positive option was picked for one expanded prompt."""
    from pipeline.utils.experiments import ApplyingExperiment, HiringExperiment

    matched_choice = fuzzy_match_option(choice, options)
    if matched_choice is None:
        print(f"Warning: Could not match choice to options: {choice!r}")
        return None

    if isinstance(experiment, HiringExperiment):
        correct_profile = experiment.variants[variant_idx]
        correct_answer = f"{correct_profile.first_name} {correct_profile.last_name}"
    elif isinstance(experiment, ApplyingExperiment):
        top_choice = max(_iter_ordered_values(experiment.vacancies), key=lambda v: v.salary)
        correct_answer = f"{top_choice.title} at {top_choice.company_name}"
    else:
        raise ValueError("Unknown experiment type")

    return sanitize_option(correct_answer) == matched_choice


def rating_schema() -> dict[str, object]:
    return {
        "type": "object",
        "properties": {
            "rating": {
                "type": "integer",
                "enum": list(range(1, 11)),
            }
        },
        "additionalProperties": False,
        "required": ["rating"],
    }


def _coerce_rating(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        rating = value
    elif isinstance(value, float) and value.is_integer():
        rating = int(value)
    elif isinstance(value, str) and re.fullmatch(r"\s*\d+(?:\.0+)?\s*", value):
        rating = int(float(value.strip()))
    else:
        return None

    if 1 <= rating <= 10:
        return float(rating)
    return None


def _parse_rating_field(content: str) -> float | None:
    match = re.search(
        r'(?i)(?:"rating"|rating)\s*:\s*"?(\d+(?:\.0+)?)"?',
        content,
    )
    if match is None:
        return None
    return _coerce_rating(match.group(1))


def parse_rating_response(content: str) -> float | None:
    """Parse a 1-10 integer rating from JSON or plain text output."""
    content = content.strip()
    if not content:
        return None

    if content.startswith("```"):
        lines = content.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        content = "\n".join(lines).strip()

    try:
        if "{" in content and "}" in content:
            start = content.find("{")
            end = content.rfind("}")
            if start != -1 and end != -1 and end > start:
                parsed = json.loads(content[start : end + 1])
                if isinstance(parsed, dict):
                    return _coerce_rating(parsed.get("rating"))

        if content.startswith("[") and content.endswith("]"):
            return None
    except json.JSONDecodeError:
        pass

    rating = _parse_rating_field(content)
    if rating is not None:
        return rating

    return _coerce_rating(content.strip().strip("\"'"))


class BaseAsyncReviewer(ABC):
    """Base class for async LLM reviewers using OpenAI-compatible APIs."""

    def __init__(self, llm: str):
        self.llm = llm
        self.client: AsyncOpenAI | None = None

    @abstractmethod
    def _get_extra_body(self) -> dict | None:
        """Return extra_body parameters for the API call, or None."""
        pass

    @abstractmethod
    def _get_extra_headers(self) -> dict | None:
        """Return extra headers for the API call, or None."""
        pass

    def _preprocess_messages(self, messages: list[dict]) -> list[dict]:
        """Hook to preprocess messages before sending. Override if needed."""
        return messages

    def _use_json_schema(self) -> bool:
        """Whether to use JSON schema for structured output. Override to disable."""
        return True

    def _get_max_retries(self) -> int:
        """Max retries for rate limit errors. Override to customize."""
        return 5

    def _get_retry_base_delay(self) -> float:
        """Base delay in seconds for exponential backoff. Override to customize."""
        return 2.0

    def _parse_response(self, content: str, options: list[str]) -> str | None:
        """Parse the model response to extract a single chosen option."""
        content = content.strip()

        if not content:
            return None

        if content.startswith("```"):
            lines = content.splitlines()
            if lines and lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].startswith("```"):
                lines = lines[:-1]
            content = "\n".join(lines).strip()

        try:
            if "{" in content and "}" in content:
                start = content.find("{")
                end = content.rfind("}")
                if start != -1 and end != -1 and end > start:
                    parsed = json.loads(content[start : end + 1])
                    if isinstance(parsed, dict):
                        choice = parsed.get("choice")
                        if isinstance(choice, str):
                            return fuzzy_match_option(choice, options)
                        return None

            if content.startswith("[") and content.endswith("]"):
                return None
        except json.JSONDecodeError:
            pass

        stripped = content.strip().strip("\"'")
        matched = fuzzy_match_option(stripped, options)
        if matched:
            return matched

        found: list[str] = []
        content_normalized = normalize_for_comparison(content)
        for option in options:
            if normalize_for_comparison(option) in content_normalized:
                found.append(option)

        if len(found) == 1:
            return found[0]

        for part in re.split(r"[,\n;]", content):
            matched = fuzzy_match_option(part.strip().strip("\"'"), options)
            if matched:
                return matched

        return None

    async def _process_prompt_row(
        self,
        client: AsyncOpenAI,
        row: dict[str, Any],
        semaphore: asyncio.Semaphore,
        system_prompt: str | None,
        secondary_prompt: bool = False,
    ) -> dict[str, Any]:
        """Process a single expanded prompt row with rate limiting and retries."""
        exp = row["experiment"]
        options = row["options"]

        async with semaphore:
            if system_prompt is not None:
                prompt_text = system_prompt
            elif secondary_prompt and self._use_json_schema():
                prompt_text = rating_system_prompt()
            elif secondary_prompt:
                prompt_text = plain_text_rating_system_prompt()
            elif self._use_json_schema():
                prompt_text = default_system_prompt()
            else:
                prompt_text = plain_text_system_prompt()
            messages = [
                {"role": "system", "content": prompt_text},
                {"role": "user", "content": row["prompt"]},
            ]
            messages = self._preprocess_messages(messages)

            kwargs = {
                "model": self.llm,
                "messages": messages,
                "max_tokens": 1024,
                "timeout": 60.0,
                "temperature": 1.0,
            }

            if self._use_json_schema():
                if secondary_prompt:
                    inner_schema = rating_schema()
                else:
                    inner_schema = {
                        "type": "object",
                        "properties": {
                            "choice": {
                                "type": "string",
                                "enum": options,
                            }
                        },
                        "additionalProperties": False,
                        "required": ["choice"],
                    }

                if "claude" in self.llm.lower() or "anthropic" in self.llm.lower():
                    kwargs["response_format"] = {
                        "type": "json_schema",
                        "json_schema": {"strict": True, "schema": inner_schema},
                    }
                else:
                    kwargs["response_format"] = {
                        "type": "json_schema",
                        "json_schema": {
                            "name": "review",
                            "strict": True,
                            "schema": inner_schema,
                        },
                    }

            extra_body = self._get_extra_body()
            if extra_body:
                kwargs["extra_body"] = extra_body

            extra_headers = self._get_extra_headers()
            if extra_headers:
                kwargs["extra_headers"] = extra_headers

            max_retries = self._get_max_retries()
            base_delay = self._get_retry_base_delay()

            for attempt in range(max_retries + 1):
                try:
                    response = await client.chat.completions.create(**kwargs)
                    api_choice = response.choices[0]
                    content = api_choice.message.content or ""

                    if not content:
                        finish_reason = getattr(api_choice, "finish_reason", "unknown")
                        refusal = getattr(api_choice.message, "refusal", None)
                        print(
                            "Debug: Empty response - "
                            f"finish_reason={finish_reason}, refusal={refusal}, model={response.model}"
                        )

                    if secondary_prompt:
                        rating = parse_rating_response(content)
                        if rating is None:
                            print(
                                "Warning: Could not parse rating response: "
                                f"{repr(content[:200] if content else '')}"
                            )
                            return {
                                "experiment_id": row["experiment_id"],
                                "variant_idx": row["variant_idx"],
                                "ascending": row["ascending"],
                                "choice": None,
                                "picked_perturbed": None,
                            }

                        return {
                            "experiment_id": row["experiment_id"],
                            "variant_idx": row["variant_idx"],
                            "ascending": row["ascending"],
                            "choice": rating,
                            "picked_perturbed": rating,
                        }

                    matched_choice = self._parse_response(content, options)
                    if matched_choice is None:
                        print(
                            "Warning: Could not parse response: "
                            f"{repr(content[:200] if content else '')}"
                        )
                        return {
                            "experiment_id": row["experiment_id"],
                            "variant_idx": row["variant_idx"],
                            "ascending": row["ascending"],
                            "choice": None,
                            "picked_perturbed": None,
                        }

                    positive = check_output(
                        exp,
                        matched_choice,
                        options,
                        row["variant_idx"],
                    )
                    return {
                        "experiment_id": row["experiment_id"],
                        "variant_idx": row["variant_idx"],
                        "ascending": row["ascending"],
                        "choice": matched_choice,
                        "picked_perturbed": positive,
                    }

                except RateLimitError as exc:
                    if attempt < max_retries:
                        await asyncio.sleep(base_delay * (2 ** attempt))
                        continue
                    print(f"Rate limit exceeded after {max_retries} retries: {exc}")
                    return {
                        "experiment_id": row["experiment_id"],
                        "variant_idx": row["variant_idx"],
                        "ascending": row["ascending"],
                        "choice": None,
                        "picked_perturbed": None,
                    }

                except Exception as exc:
                    print(f"Error processing experiment: {exc}")
                    return {
                        "experiment_id": row["experiment_id"],
                        "variant_idx": row["variant_idx"],
                        "ascending": row["ascending"],
                        "choice": None,
                        "picked_perturbed": None,
                    }

    def review(
        self,
        experiments,
        max_concurrent: int = 10,
        top_n: int = 2,
        system_prompt: str | None = None,
        secondary_prompt: bool = False,
    ):
        """Review experiments using async chat completions."""
        del top_n
        if self.client is None:
            raise RuntimeError("Client not initialized.")
        return asyncio.run(
            self._review_async(
                experiments,
                max_concurrent,
                system_prompt,
                secondary_prompt=secondary_prompt,
            )
        )

    async def _review_async(
        self,
        experiments,
        max_concurrent: int = 10,
        system_prompt: str | None = None,
        secondary_prompt: bool = False,
    ):
        semaphore = asyncio.Semaphore(max_concurrent)
        rows = [
            row
            for experiment in experiments
            for row in expand_experiment_rows(experiment, secondary_prompt=secondary_prompt)
        ]
        tasks = [
            self._process_prompt_row(
                self.client,
                row,
                semaphore,
                system_prompt,
                secondary_prompt=secondary_prompt,
            )
            for row in rows
        ]
        return await tqdm.gather(*tasks, desc="Processing experiment variants")
