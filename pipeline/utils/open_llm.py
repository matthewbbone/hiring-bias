import os
import asyncio

from openai import AsyncOpenAI
from dotenv import load_dotenv

try:
    from pipeline.utils.llm import BaseAsyncReviewer
except ModuleNotFoundError:
    from llm import BaseAsyncReviewer

load_dotenv()

class OpenRouterReviewer(BaseAsyncReviewer):
    """LLM Reviewer using OpenRouter API."""

    def __init__(
        self,
        llm: str = "openai/gpt-4o",
        api_key: str | None = None,
        site_url: str | None = None,
        site_name: str | None = None,
        use_json_schema: bool = False,
    ):
        super().__init__(llm)
        self.api_key = api_key or os.environ.get("OPENROUTER_API_KEY")
        if not self.api_key:
            raise ValueError(
                "OpenRouter API key not provided. "
                "Set OPENROUTER_API_KEY environment variable or pass api_key."
            )

        self.site_url = site_url
        self.site_name = site_name
        self._json_schema_enabled = use_json_schema

        self.client = AsyncOpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=self.api_key,
        )
        print(f"OpenRouter client initialized for model: {self.llm}")
        if not use_json_schema:
            print("JSON schema disabled (free models may not support it)")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        return False

    def _get_extra_body(self) -> dict | None:
        if "kimi" in self.llm:
            return {
                "provider": {"order": ["moonshotai"], "allow_fallbacks": False},
            }
        elif "glm" in self.llm:
            return {
                "reasoning": {"enabled": False},
                "provider": {"order": ["parasail/fp8"], "allow_fallbacks": False},
            }
        elif "deepseek" in self.llm:
            return {
                "reasoning": {"enabled": False},
                "provider": {"order": ["parasail/fp8"], "allow_fallbacks": False},
            }
        return None

    def _get_extra_headers(self) -> dict | None:
        headers = {}
        if self.site_url:
            headers["HTTP-Referer"] = self.site_url
        if self.site_name:
            headers["X-Title"] = self.site_name
        return headers if headers else None

    def _use_json_schema(self) -> bool:
        """Disable JSON schema for free models that don't support it."""
        return self._json_schema_enabled

    def _get_max_retries(self) -> int:
        """More retries for OpenRouter rate limits."""
        return 8

    def _get_retry_base_delay(self) -> float:
        """Longer base delay for OpenRouter (free tier is 20 req/min)."""
        return 3.0

def main():
    """Test OpenRouter connection with a simple prompt."""
    import argparse

    parser = argparse.ArgumentParser(description="Test OpenRouter API connection")
    parser.add_argument(
        "--model",
        type=str,
        default="allenai/molmo-2-8b:free",
        help="Model to use (default: allenai/molmo-2-8b:free)"
    )
    args = parser.parse_args()

    print(f"Testing OpenRouter with model: {args.model}")

    with OpenRouterReviewer(llm=args.model) as reviewer:
        # Simple test using the async client directly
        async def test_completion():
            response = await reviewer.client.chat.completions.create(
                model=reviewer.llm,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant. Respond briefly."},
                    {"role": "user", "content": "What is the capital of France?"},
                ],
                max_tokens=50,
                temperature=1.0,
            )
            return response.choices[0].message.content

        result = asyncio.run(test_completion())
        print(f"Response: {result}")
        print("Test completed successfully!")

if __name__ == "__main__":
    main()
