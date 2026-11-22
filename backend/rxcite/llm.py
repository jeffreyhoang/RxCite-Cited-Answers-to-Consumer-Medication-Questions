# One way to call Claude: responses are cached on disk so repeated calls skip the API

import hashlib
import json
from dataclasses import asdict, dataclass, fields

import anthropic

from rxcite.config import CACHE_DIR, MAX_TOKENS, anthropic_api_key

LLM_CACHE = CACHE_DIR / "llm"
SONNET_FALLBACK_BETA = "server-side-fallback-2026-07-01"

_client: anthropic.Anthropic | None = None


@dataclass
class LLMResponse:
    text: str
    model: str
    input_tokens: int
    output_tokens: int
    stop_reason: str | None
    cached: bool = False


def client() -> anthropic.Anthropic:
    global _client
    if _client is None:
        _client = anthropic.Anthropic(api_key=anthropic_api_key())
    return _client


# Same request -> same file, so reruns skip the API
def _cache_path(request: dict):
    key = hashlib.sha256(json.dumps(request, sort_keys=True).encode()).hexdigest()
    return LLM_CACHE / f"{key}.json"


def call(
    prompt: str,
    model: str,
    system: str | None = None,
    max_tokens: int = MAX_TOKENS,
    effort: str = "low",
    schema: dict | None = None,  # JSON schema: the reply is then guaranteed to be valid JSON
) -> LLMResponse:
    request = {
        "model": model,
        "system": system,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "effort": effort,
        "schema": schema,
    }
    path = _cache_path(request)
    if path.exists():
        saved = json.loads(path.read_text())
        known = {f.name for f in fields(LLMResponse)}
        return LLMResponse(**{k: v for k, v in saved.items() if k in known} | {"cached": True})

    kwargs = {
        "model": model,
        "max_tokens": max_tokens,
        "messages": request["messages"],
        "output_config": {"effort": effort},  # no temperature
    }
    if schema:
        kwargs["output_config"]["format"] = {"type": "json_schema", "schema": schema}
    if system:
        kwargs["system"] = system
    if model.startswith("claude-sonnet-5-5"):
        # if Sonnet declines on a safety classifier, the API retries on a fallback model
        resp = client().beta.messages.create(
            **kwargs, betas=[SONNET_FALLBACK_BETA], fallbacks="default"
        )
    else:
        resp = client().messages.create(**kwargs)

    out = LLMResponse(
        text="".join(b.text for b in resp.content if b.type == "text"),
        model=model,
        input_tokens=resp.usage.input_tokens,
        output_tokens=resp.usage.output_tokens,
        stop_reason=resp.stop_reason,
    )
    LLM_CACHE.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(out)))
    return out
