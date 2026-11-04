"""LLM backends for the harness.

Two implementations behind one interface:
  - OpenAILLM: real calls against an OpenAI-compatible chat-completions API,
               authenticated with a standard `OPENAI_API_KEY` bearer key.
  - MockLLM:   deterministic, offline, no credentials. Used by tests and by
               anyone who wants to see the mechanics without burning tokens.

Auth is the plain OpenAI scheme: a long-lived secret key (`sk-...`) read from
the environment and handed to the SDK as `api_key=`. `OPENAI_BASE_URL` points
the same client at any compatible endpoint - Azure OpenAI, vLLM, LiteLLM,
OpenRouter, Ollama - without touching this file.

Whatever endpoint you use, it must return a `usage` block on each completion:
per-call token accounting is the entire point of this comparison, and an arm
that reports zero tokens silently invalidates the cost columns.
"""

from __future__ import annotations

import os
import random
import re
import sys
import time
from dataclasses import dataclass, field

import httpx

DEFAULT_BASE_URL = "https://api.openai.com/v1"
DEFAULT_MODEL = "gpt-4o-mini"


class BudgetExceeded(RuntimeError):
    """Raised when an arm tries to exceed its LLM-call ceiling.

    This is a first-class outcome, not a crash: the `budget_cap` condition
    exists precisely to show what each architecture does when it runs out of
    room. Arms catch it and answer with whatever they have.
    """


class MissingAPIKey(RuntimeError):
    """No `OPENAI_API_KEY` in the environment."""


@dataclass
class LLMResult:
    text: str
    model_used: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_s: float = 0.0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


@dataclass
class CallBudget:
    """Shared across one question's run. `max_calls=None` means unlimited."""

    max_calls: int | None = None
    calls_made: int = 0

    def charge(self) -> None:
        if self.max_calls is not None and self.calls_made >= self.max_calls:
            raise BudgetExceeded(
                f"LLM call budget of {self.max_calls} exhausted"
            )
        self.calls_made += 1


# --------------------------------------------------------------------------
# OpenAI-compatible backend
# --------------------------------------------------------------------------


def _resolve_verify():
    """Value for httpx `verify=`.

    Order: OPENAI_CA_BUNDLE (explicit path) -> OPENAI_VERIFY_SSL=false (disable,
    insecure) -> a trust-store bundle we exported earlier -> system default.

    The bundle fallback matters behind a TLS-inspecting corporate proxy: its
    root sits in the OS trust store, which Python's ssl module does not read on
    macOS, so the endpoint fails to verify even though it works in a browser.
    See harness/dataset.py::bootstrap_ca.
    """
    ca_bundle = os.getenv("OPENAI_CA_BUNDLE")
    if ca_bundle:
        return ca_bundle
    if os.getenv("OPENAI_VERIFY_SSL", "true").lower() in ("false", "0", "no"):
        return False
    from .dataset import CA_BUNDLE_PATH, bootstrap_ca

    if CA_BUNDLE_PATH.exists():
        return str(CA_BUNDLE_PATH)
    if sys.platform == "darwin" and bootstrap_ca():
        return str(CA_BUNDLE_PATH)
    return True


def get_api_key() -> str:
    """The secret key, or a pointed error telling you how to supply one.

    Never logged, never written to a trace, never echoed in an error message -
    only its absence is ever reported.
    """
    key = os.getenv("OPENAI_API_KEY", "").strip()
    if not key:
        raise MissingAPIKey(
            "OPENAI_API_KEY is not set.\n"
            "  export OPENAI_API_KEY=sk-...            (or put it in .env - see .env.example)\n"
            "  python -m harness.run --backend openai --env-file path/to/.env\n"
            "  python -m harness.run --backend mock    (no key needed, offline)"
        )
    return key


def base_url() -> str:
    return os.getenv("OPENAI_BASE_URL", DEFAULT_BASE_URL)


class OpenAILLM:
    """Chat completions against an OpenAI-compatible endpoint."""

    def __init__(self, model: str | None = None, temperature: float = 0.0,
                 max_retries: int = 4):
        self.model = model or os.getenv("OPENAI_MODEL", DEFAULT_MODEL)
        self.temperature = temperature
        self.max_retries = max_retries
        self._client = None
        # Parameters this model has rejected. Some models drop `stop` and pin
        # `temperature`, and the API reports that as a 400 naming the offending
        # param. We drop it and carry on rather than failing the arm -
        # otherwise a model-capability detail would read as an architecture
        # result in the comparison.
        self._unsupported: set[str] = set()

    def _ensure_client(self):
        from openai import OpenAI

        if self._client is None:
            self._client = OpenAI(
                api_key=get_api_key(),
                base_url=base_url(),
                http_client=httpx.Client(verify=_resolve_verify(), timeout=60),
            )
        return self._client

    def complete(self, messages: list[dict], stop: list[str] | None = None,
                 max_tokens: int | None = None) -> LLMResult:
        """One chat completion. Retries transient errors with exponential
        backoff; the elapsed time we report includes those retries, because
        that is the latency a user actually experiences."""
        t0 = time.perf_counter()
        last_exc: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                kwargs: dict = {"model": self.model, "messages": messages}
                if "temperature" not in self._unsupported:
                    kwargs["temperature"] = self.temperature
                if stop and "stop" not in self._unsupported:
                    kwargs["stop"] = stop
                if max_tokens and "max_tokens" not in self._unsupported:
                    kwargs["max_tokens"] = max_tokens
                resp = self._ensure_client().chat.completions.create(**kwargs)
                usage = getattr(resp, "usage", None)
                return LLMResult(
                    text=resp.choices[0].message.content or "",
                    model_used=getattr(resp, "model", "") or self.model,
                    prompt_tokens=int(getattr(usage, "prompt_tokens", 0) or 0),
                    completion_tokens=int(getattr(usage, "completion_tokens", 0) or 0),
                    latency_s=time.perf_counter() - t0,
                )
            except Exception as e:  # noqa: BLE001 - the API surfaces many shapes
                last_exc = e
                dropped = _unsupported_param(e)
                if dropped and dropped not in self._unsupported:
                    self._unsupported.add(dropped)
                    _llm_log(f"[LLM] model {self.model} rejected {dropped!r}; "
                             f"dropping it for the rest of the run")
                    continue  # immediate retry without the offending parameter
                if not _is_retryable(e) or attempt == self.max_retries - 1:
                    raise
                time.sleep(min(2 ** attempt, 8) + random.random())
        raise last_exc  # unreachable; keeps type checkers happy

def _llm_log(msg: str) -> None:
    print(msg, flush=True)


_UNSUPPORTED_RE = re.compile(
    r"unsupported parameter:\s*'([^']+)'|"
    r"'([^']+)' is not supported with this model|"
    r"unsupported value:\s*'([^']+)'",
    re.IGNORECASE,
)


def _unsupported_param(exc: Exception) -> str | None:
    """Name of the parameter the gateway rejected as unsupported, if any.

    Prefers the structured `param` field the OpenAI error body carries, and
    falls back to parsing the message.
    """
    body = getattr(exc, "body", None)
    if isinstance(body, dict):
        err = body.get("error") or {}
        if isinstance(err, dict):
            code = str(err.get("code") or "")
            param = err.get("param")
            if param and ("unsupported" in code or "unsupported" in str(err.get("message", "")).lower()):
                return str(param)
    m = _UNSUPPORTED_RE.search(str(exc))
    if m:
        return next((g for g in m.groups() if g), None)
    return None


def _is_retryable(exc: Exception) -> bool:
    status = getattr(exc, "status_code", None) or getattr(
        getattr(exc, "response", None), "status_code", None
    )
    if status in (408, 409, 429, 500, 502, 503, 504):
        return True
    return isinstance(exc, (httpx.TimeoutException, httpx.ConnectError))


# --------------------------------------------------------------------------
# Mock
# --------------------------------------------------------------------------

@dataclass
class MockLLM:
    """Deterministic stand-in so the harness runs with no credentials.

    It is not trying to be smart. It pattern-matches the prompt's role in the
    pipeline and emits a well-formed response of the right shape, pulling real
    entity names out of the context it was given. That is enough to exercise
    every control-flow path (including malformed-action and non-termination
    handling) and to make the viewer show real traces offline.

    Token counts are approximated at 4 characters/token so the cost columns
    are populated and the dashboard renders; they are clearly flagged as
    estimated in the output so nobody mistakes them for metered usage.
    """

    seed: int = 0
    model_used: str = "mock-deterministic"
    _rng: random.Random = field(init=False, repr=False)

    def __post_init__(self):
        self._rng = random.Random(self.seed)

    def complete(self, messages: list[dict], stop: list[str] | None = None,
                 max_tokens: int | None = None) -> LLMResult:
        t0 = time.perf_counter()
        prompt = "\n".join(m.get("content") or "" for m in messages)
        text = self._respond(prompt)
        # Simulated think-time keeps the latency chart from being all zeros and
        # scales with output length, the way a real decode does.
        time.sleep(0.002 + 0.0005 * self._rng.random())
        return LLMResult(
            text=text,
            model_used=self.model_used,
            prompt_tokens=max(1, len(prompt) // 4),
            completion_tokens=max(1, len(text) // 4),
            latency_s=time.perf_counter() - t0,
        )

    def _respond(self, prompt: str) -> str:
        low = prompt.lower()
        if "decompose" in low:
            return self._decompose(prompt)
        if "thought" in low and "action" in low:
            return self._react_step(prompt)
        if "classify" in low or "how many hops" in low:
            return self._rng.choice(["1", "2", "2", "2", "3"])
        return self._synthesize(prompt)

    # -- shape helpers ----------------------------------------------------
    def _question(self, prompt: str) -> str:
        m = re.search(r"Question:\s*(.+)", prompt)
        return m.group(1).strip() if m else ""

    def _entities(self, prompt: str) -> list[str]:
        """Capitalised multi-word spans from the observations - a crude proxy
        for the titles a real model would latch onto."""
        obs = "\n".join(re.findall(r"Observation[^:]*:\s*(.+)", prompt))
        found = re.findall(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,3})\b", obs)
        seen, out = set(), []
        for f in found:
            if f not in seen and len(f) > 3:
                seen.add(f)
                out.append(f)
        return out[:8]

    def _decompose(self, prompt: str) -> str:
        q = self._question(prompt)
        head = " ".join(q.split()[:6])
        return f"1. {head}?\n2. What is the attribute asked about in: {q}"

    def _react_step(self, prompt: str) -> str:
        n_obs = prompt.count("Observation")
        ents = self._entities(prompt)
        # Occasionally emit a malformed action so the parser's error path is
        # exercised in offline runs too.
        if n_obs == 1 and self._rng.random() < 0.08:
            return "Thought: I should look this up.\nAction: search Barack Obama"
        if n_obs == 0:
            q = self._question(prompt)
            seed_term = " ".join(q.split()[1:5]) or "topic"
            return (f"Thought: I need to find the entity the question bridges through.\n"
                    f"Action: search[{seed_term}]")
        if n_obs < 3 and ents:
            return (f"Thought: The first result points at {ents[0]}; I need its attribute.\n"
                    f"Action: search[{ents[min(1, len(ents) - 1)]}]")
        answer = ents[0] if ents else "unknown"
        return (f"Thought: I now have both hops and can answer.\n"
                f"Action: finish[{answer}]")

    def _synthesize(self, prompt: str) -> str:
        ents = self._entities(prompt)
        if "yes or no" in prompt.lower():
            return self._rng.choice(["yes", "no"])
        return ents[0] if ents else "unknown"


def build_llm(backend: str, model: str | None = None, seed: int = 0):
    if backend == "mock":
        return MockLLM(seed=seed)
    if backend == "openai":
        return OpenAILLM(model=model)
    raise ValueError(f"unknown backend {backend!r}; expected 'openai' or 'mock'")
