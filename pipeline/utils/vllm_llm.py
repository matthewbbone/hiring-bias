import time
import os
import subprocess
import asyncio
import json
from pathlib import Path
from dotenv import load_dotenv
import torch
from openai import AsyncOpenAI
from transformers import AutoTokenizer

load_dotenv()

try:
    from pipeline.utils.llm import BaseAsyncReviewer
except ModuleNotFoundError:
    from llm import BaseAsyncReviewer

class VLLMReviewer(BaseAsyncReviewer):
    """LLM Reviewer using a local vLLM server."""

    PLAIN_TEXT_MODELS = {
        "moonshotai/Kimi-Linear-48B-A3B-Base",
    }

    def __init__(
        self,
        llm: str = "google/gemma-3-27b-it",
        n_gpus: int = 1,
        port: int = 8000,
        gpu_ids: str = "0",
        quantization: str | None = None,
        dtype: str | None = None,
    ):
        super().__init__(llm)
        self.server = None
        self.n_gpus = n_gpus
        self.port = port
        self.gpu_ids = gpu_ids
        self.quantization = quantization
        self.dtype = dtype
        self._start_server()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        # Force GPU memory cleanup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return False

    def _get_extra_body(self) -> dict | None:
        if not self._use_json_schema():
            return None
        return {"guided_decoding_backend": "xgrammar"}

    def _get_extra_headers(self) -> dict | None:
        return None

    def _use_json_schema(self) -> bool:
        return self.llm not in self.PLAIN_TEXT_MODELS

    def _preprocess_messages(self, messages: list[dict]) -> list[dict]:
        """Add special tokens for specific models."""
        if self.llm == "nvidia/NVIDIA-Nemotron-Nano-12B-v2":
            messages = [{"role": "system", "content": "/no_think"}] + messages
        return messages

    def _start_server(self):
        """Start the vLLM server and wait for it to be ready."""
        base_to_instruct = {
            "meta-llama/Llama-3.1-8B": "meta-llama/Llama-3.1-8B-Instruct",
            "meta-llama/Llama-3.2-3B": "meta-llama/Llama-3.2-3B-Instruct",
            "google/gemma-3-27b-pt": "google/gemma-3-27b-it",
            "Qwen/Qwen3-30B-A3B-Base": "Qwen/Qwen3-30B-A3B-Instruct-2507",
            "deepseek-ai/DeepSeek-V2-Lite": "deepseek-ai/DeepSeek-V2-Lite-Chat",
            "allenai/Olmo-3-1025-7B": "allenai/Olmo-3-7B-Instruct",
            "allenai/Olmo-3-1125-32B": "allenai/Olmo-3.1-32B-Instruct",
            "ibm-granite/granite-4.0-h-small-base": "ibm-granite/granite-4.0-h-small",
            "tiiuae/Falcon-H1-34B-Base": "tiiuae/Falcon-H1-34B-Instruct",
            "zai-org/GLM-4-32B-Base-0414": "zai-org/GLM-4-32B-0414",
            "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-Base-BF16": "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16",
            "nvidia/NVIDIA-Nemotron-Nano-12B-v2-Base": "nvidia/NVIDIA-Nemotron-Nano-12B-v2",
            "moonshotai/Moonlight-16B-A3B": "moonshotai/Moonlight-16B-A3B-Instruct"
        }
        force_instruct_chat_template = {
            "allenai/Olmo-3-1125-32B",
            "ibm-granite/granite-4.0-h-small-base",
            "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-Base-BF16",
            "moonshotai/Moonlight-16B-A3B",
        }

        env = os.environ.copy()
        if self.gpu_ids:
            env["CUDA_VISIBLE_DEVICES"] = self.gpu_ids

        resolved_quantization = self._resolve_quantization()
        resolved_dtype = self._resolve_dtype(resolved_quantization)

        cmd = [
            "vllm", "serve", self.llm,
            "--max_model_len", "5000",
            "--tensor-parallel-size", str(self.n_gpus),
            "--trust-remote-code",
            "--gpu-memory-utilization", "0.9",
            "--max-num-seqs", "16",
            "--enable-prefix-caching",
            "--disable-log-stats",
            "--port", str(self.port)
        ]

        if resolved_dtype:
            cmd.extend(["--dtype", resolved_dtype])
        if resolved_quantization:
            cmd.extend(["--quantization", resolved_quantization])

        if "qwen" in self.llm.lower():
            cmd.extend(["--attention-backend", "FLASH_ATTN"])
            cmd.extend(["--gdn-prefill-backend", "triton"])
            cmd.extend([
                "--limit-mm-per-prompt.video", "0",
                "--limit-mm-per-prompt.image", "0",
            ])

        if self.llm == "allenai/Olmo-3-7B-Instruct":
            cache_home = env.get("XDG_CACHE_HOME") or env.get("HF_HOME") or str(Path.home() / ".cache")
            env["VLLM_CACHE_ROOT"] = cache_home
            env["VLLM_ASSETS_CACHE"] = cache_home
            env["HF_HOME"] = cache_home
            env["HUGGINGFACE_HUB_CACHE"] = cache_home
            env["TRANSFORMERS_CACHE"] = os.path.join(cache_home, "transformers")
            env["HF_DATASETS_CACHE"] = os.path.join(cache_home, "datasets")
            cmd.extend(["--download_dir", cache_home])

        if "nvidia" in self.llm:
            cmd.extend(["--mamba_ssm_cache_dtype", "float32"])

        if self.llm in base_to_instruct.keys():

            tokenizer = AutoTokenizer.from_pretrained(
                self.llm,
                use_fast=True,
                trust_remote_code=True,
                cache_dir=env.get("XDG_CACHE_HOME", None) if "allenai" in self.llm else None
            )

            if (
                self.llm in force_instruct_chat_template
                or not hasattr(tokenizer, "chat_template")
                or tokenizer.chat_template is None
            ):
                print(
                    f"Using instruct chat template from {base_to_instruct[self.llm]} "
                    f"for {self.llm}."
                )
                tokenizer = AutoTokenizer.from_pretrained(
                    base_to_instruct[self.llm],
                    use_fast=True,
                    trust_remote_code=True,
                    cache_dir=env.get("XDG_CACHE_HOME", None) if "allenai" in self.llm else None,
                )

                if not hasattr(tokenizer, "chat_template") or tokenizer.chat_template is None:
                    raise ValueError(f"No chat template found for {base_to_instruct[self.llm]} either.")

                cmd.extend([
                    "--chat-template",
                    tokenizer.chat_template
                ])

        if "mistralai" in self.llm:
            cmd.extend([
                "--tokenizer_mode", "mistral",
                "--load_format", "mistral",
                "--config_format", "mistral"
            ])

        log_file = open(
            os.environ["CACHE_DIR"] + "/logs/"
            f"vllm_{self.llm.replace('/', '_')}_server.log",
            "w"
        )
        self.server = subprocess.Popen(
            cmd,
            stdout=log_file,
            stderr=log_file,
            env=env
        )

        openai_api_key = "EMPTY"
        openai_api_base = f"http://localhost:{self.port}/v1"

        self.client = AsyncOpenAI(
            api_key=openai_api_key,
            base_url=openai_api_base
        )

        # Wait for server to be ready
        print(f"Starting vLLM server for {self.llm}...")
        asyncio.run(self._wait_for_server())
        print("Server ready!")

    def _resolve_quantization(self) -> str | None:
        if self.quantization is not None:
            return self.quantization

        config_path = Path(self.llm) / "config.json"
        if not config_path.exists():
            return None

        try:
            config = json.loads(config_path.read_text())
        except json.JSONDecodeError:
            return None

        if "quantization_config" in config:
            return "compressed-tensors"
        return None

    def _resolve_dtype(self, quantization: str | None) -> str | None:
        if self.dtype is not None:
            return self.dtype
        if quantization == "compressed-tensors":
            return "auto"
        return "bfloat16"

    async def _wait_for_server(self, timeout=60000):
        """Wait for the vLLM server to be ready."""
        start = time.time()
        while True:
            try:
                await self.client.models.list()
                break
            except Exception as e:
                if time.time() - start > timeout:
                    self.close()
                    raise RuntimeError(f"VLLM server did not start within {timeout/60:.1f} minutes") from e
                await asyncio.sleep(1)

    def close(self):
        """Terminate the vLLM server."""
        if self.server is not None:
            print(f"Shutting down vLLM server for {self.llm}...")
            self.server.terminate()
            try:
                self.server.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.server.kill()
            self.server = None
            self.client = None
            print("Server shut down.")

    def review(
        self,
        experiments,
        max_concurrent=10,
        top_n=2,
        system_prompt=None,
        secondary_prompt=False,
    ):
        """Review experiments using async chat completions."""
        del top_n
        if self.server is None or self.client is None:
            raise RuntimeError("Server not running. Create a new VLLMReviewer instance.")
        return asyncio.run(
            self._review_async(
                experiments,
                max_concurrent,
                system_prompt,
                secondary_prompt=secondary_prompt,
            )
        )
    
def main():
    """Test vLLM server with a simple prompt."""
    import argparse

    parser = argparse.ArgumentParser(description="Test vLLM server connection")
    parser.add_argument(
        "--model",
        type=str,
        default="google/gemma-3-27b-it",
        help="Model to use (default: google/gemma-3-27b-it)"
    )
    parser.add_argument(
        "--n-gpus",
        type=int,
        default=1,
        help="Number of GPUs (default: 1)"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Server port (default: 8000)"
    )
    parser.add_argument(
        "--gpu-ids",
        type=str,
        default="0",
        help="GPU IDs to use (default: 0)"
    )
    # parser.add_argument(
    #     "--quantization",
    #     type=str,
    #     default=None,
    #     help="Override vLLM quantization mode."
    # )
    parser.add_argument(
        "--dtype",
        type=str,
        default=None,
        help="Override vLLM dtype."
    )
    args = parser.parse_args()

    print(f"Testing vLLM with model: {args.model}")

    with VLLMReviewer(
        llm=args.model,
        n_gpus=args.n_gpus,
        port=args.port,
        gpu_ids=args.gpu_ids,
        quantization=args.quantization,
        dtype=args.dtype,
    ) as reviewer:
        # Simple test using the async client directly
        async def test_completion():
            response = await reviewer.client.chat.completions.create(
                model=reviewer.llm,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant. Respond briefly."},
                    {"role": "user", "content": "Say 'vLLM connection successful!' and nothing else."}
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
