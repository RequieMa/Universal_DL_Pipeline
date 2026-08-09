"""Serving script for the fine-tuned distilgpt2 (M6).

Loads the distilgpt2 base model from HuggingFace, reapplies the same LoRA
adapter used during fine-tuning, restores the fine-tuned weights from the
``distilgpt2-instruct.pt`` checkpoint, and exposes an OpenAI-compatible
``/v1/chat/completions`` endpoint.

This completes the M6 cycle: ``train -> export -> serve -> chat``. The
notebook fine-tuned the model on a tiny instruction dataset and exported it
via ``TorchCheckpoint``; this script serves that checkpoint over HTTP.

Run::

    uvicorn examples.m6_llm_distilgpt2.serve:app

or::

    python examples/m6-llm-distilgpt2/serve.py

The checkpoint ``distilgpt2-instruct.pt`` must sit next to this file.
"""

from __future__ import annotations

import sys
from pathlib import Path

from fastapi import FastAPI  # lightweight; uvicorn (the server) stays lazy
from pydantic import BaseModel, Field

_HERE = Path(__file__).resolve().parent
# Repo root = examples/m6-llm-distilgpt2 -> ../.. so ``pipeline`` is importable
# no matter what the working directory is when this script is launched.
_REPO_ROOT = _HERE.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

CHECKPOINT_PATH = _HERE / "distilgpt2-instruct.pt"
MODEL_NAME = "distilgpt2"

# Same LoRA config used in the M6 notebook so the adapter matches the
# checkpoint's state_dict layout.
LORA_CONFIG = {
    "r": 8,
    "lora_alpha": 32,
    "lora_dropout": 0.1,
    "target_modules": ["c_attn"],  # distilgpt2 attention layers
}

_DEVICE: str = "cpu"
_model = None       # PEFT model instance (frozen for inference)
_tokenizer = None   # HuggingFace tokenizer


def get_device() -> str:
    """Return the active device string, preferring CUDA when available.

    Returns:
        ``"cuda"`` if a CUDA device is available, else ``"cpu"``.
    """
    import torch  # type: ignore[import-not-found]

    return "cuda" if torch.cuda.is_available() else "cpu"


def _load_model() -> None:
    """Load the base model, apply LoRA, and restore the fine-tuned weights.

    The checkpoint was saved by ``TorchCheckpoint.save`` on the PEFT model,
    so ``model_state_dict`` holds the full PEFT state (base weights plus the
    ``lora_`` adapter weights). Recreating ``get_peft_model`` yields an
    identical module layout, so ``TorchCheckpoint.load`` restores the whole
    thing in one call — the adapter needs no special handling.

    Raises:
        RuntimeError: If the checkpoint file is missing.
    """
    global _model, _tokenizer, _DEVICE

    if not CHECKPOINT_PATH.exists():
        raise RuntimeError(
            f"Checkpoint not found: {CHECKPOINT_PATH}\n"
            "Run the M6 notebook (distilgpt2.ipynb) first to create it."
        )

    from peft import LoraConfig, TaskType, get_peft_model  # type: ignore[import-not-found]
    from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore[import-not-found]

    from pipeline.adapters.torch_adapter import TorchModel
    from pipeline.config import Config
    from pipeline.export.checkpoint import TorchCheckpoint
    from pipeline.pipeline import PipelineState

    _DEVICE = get_device()

    base_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(_DEVICE)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token

    peft_model = get_peft_model(
        base_model,
        LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=LORA_CONFIG["r"],
            lora_alpha=LORA_CONFIG["lora_alpha"],
            lora_dropout=LORA_CONFIG["lora_dropout"],
            target_modules=LORA_CONFIG["target_modules"],
        ),
    )

    # Restore the full PEFT state (base weights + adapter) from the checkpoint.
    state = PipelineState(config=Config(output_dir=str(_HERE)))
    state.model = TorchModel(peft_model)
    TorchCheckpoint.load(state, CHECKPOINT_PATH)

    peft_model.to(_DEVICE)
    peft_model.eval()

    _model = peft_model
    _tokenizer = tokenizer


def generate_response(prompt: str, max_new_tokens: int = 128) -> str:
    """Generate a response string for a user prompt.

    Mirrors the M6 notebook's ``generate`` helper: wraps the prompt in the
    instruction/response format, samples with the same decoding settings,
    and returns just the response portion.

    Args:
        prompt: The user's message content.
        max_new_tokens: Maximum number of tokens to generate.

    Returns:
        The assistant's response text.
    """
    import torch  # type: ignore[import-not-found]

    full_prompt = f"Instruction: {prompt}\nResponse:"
    inputs = _tokenizer(full_prompt, return_tensors="pt").to(_DEVICE)

    with torch.no_grad():
        outputs = _model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.2,  # discourage degenerate loops
            pad_token_id=_tokenizer.eos_token_id,
        )

    generated = _tokenizer.decode(outputs[0], skip_special_tokens=True)
    if "Response:" in generated:
        return generated.split("Response:")[-1].strip()
    return generated.strip()


# ---------------------------------------------------------------------------
# Pydantic models (OpenAI-compatible request/response shapes)
# ---------------------------------------------------------------------------


class ChatMessage(BaseModel):
    """A single message in a chat conversation.

    Attributes:
        role: Who sent the message (e.g. ``"user"`` or ``"assistant"``).
        content: The message body.
    """

    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    """OpenAI-compatible chat completion request body.

    Attributes:
        model: Model identifier (expected ``"distilgpt2"``).
        messages: Ordered conversation history.
        max_tokens: Maximum tokens to generate (default 128).
    """

    model: str = Field(default=MODEL_NAME)
    messages: list[ChatMessage] = Field(default_factory=list)
    max_tokens: int = Field(default=128, ge=1, le=1024)


class AssistantMessage(BaseModel):
    """Assistant output message.

    Attributes:
        role: Always ``"assistant"``.
        content: The generated response text.
    """

    role: str = "assistant"
    content: str


class Choice(BaseModel):
    """A single completion choice.

    Attributes:
        message: The assistant message produced by the model.
    """

    message: AssistantMessage


class ChatCompletionResponse(BaseModel):
    """OpenAI-compatible chat completion response body.

    Attributes:
        choices: List of generated completions (we return exactly one).
    """

    choices: list[Choice]


# ---------------------------------------------------------------------------
# FastAPI application
# ---------------------------------------------------------------------------


def chat_completions(req: ChatCompletionRequest) -> ChatCompletionResponse:
    """Handle a ``POST /v1/chat/completions`` request.

    Uses the last user message as the prompt (matching the single
    instruction/response format the model was fine-tuned on), generates a
    response, and returns it in OpenAI shape.

    Args:
        req: Validated request body (Pydantic).

    Returns:
        An OpenAI-compatible completion response.
    """
    if _model is None:
        raise RuntimeError("Model not loaded yet; call _load_model() first.")

    user_msgs = [m.content for m in req.messages if m.role == "user"]
    prompt = user_msgs[-1] if user_msgs else ""

    content = generate_response(prompt, max_new_tokens=req.max_tokens)
    return ChatCompletionResponse(choices=[Choice(message=AssistantMessage(content=content))])


def build_app() -> FastAPI:
    """Construct and return the FastAPI application.

    Defined as a helper so it can be imported and tested without launching
    uvicorn. The heavyweight model load happens in the startup handler.

    Returns:
        A configured ``FastAPI`` instance (``app``).
    """
    from contextlib import asynccontextmanager

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        """Load the model once when the server starts."""
        _load_model()
        yield

    fastapi_app = FastAPI(
        title="distilgpt2-instruct", version="1.0.0", lifespan=lifespan
    )

    @fastapi_app.get("/health")
    def _health() -> dict[str, str]:
        """Report service health and the active device.

        Returns:
            A small JSON document with status and device.
        """
        return {"status": "ok", "model": MODEL_NAME, "device": _DEVICE}

    fastapi_app.post("/v1/chat/completions")(chat_completions)
    return fastapi_app


def main() -> None:
    """Entry point: build the app, load the model, and start uvicorn.

    uvicorn is imported here (bottom of module) so importing this file to
    build the app does not require the web-server stack.
    """
    import uvicorn

    app = build_app()
    uvicorn.run(app, host="0.0.0.0", port=8000)


# Module-level app for ``uvicorn examples.m6_llm_distilgpt2.serve:app``.
app = build_app()


if __name__ == "__main__":
    main()
