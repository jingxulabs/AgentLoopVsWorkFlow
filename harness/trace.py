"""One trace schema, emitted identically by every arm.

The viewer replays these side-by-side, so the schema is the contract between
the harness and the dashboard. Keep it additive.

A Step is one unit of control flow. For `workflow` the steps are the DAG's
named nodes; for `react` they are loop iterations. Both carry the same
accounting fields, which is what lets the dashboard put them on one axis.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field


@dataclass
class Step:
    index: int
    kind: str              # "llm" | "tool" | "node"
    label: str             # "decompose", "react_step_2", "retrieve_hop1", ...
    thought: str = ""
    action: str = ""       # "search[Scott Derrickson]"
    observation: str = ""
    llm_calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    tool_calls: int = 0
    tool_failures: int = 0
    latency_s: float = 0.0
    note: str = ""         # parser errors, injected faults, budget stops

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass
class RunResult:
    """One (arm, condition, question) run."""

    arm: str
    condition: str
    question_id: str
    question: str
    gold_answer: str
    question_type: str     # bridge | comparison
    level: str             # easy | medium | hard
    predicted_answer: str = ""
    steps: list = field(default_factory=list)
    terminated: str = "answered"   # answered | step_limit | budget | error | malformed
    error: str = ""
    seed: int = 0
    scores: dict = field(default_factory=dict)
    backend: str = ""
    model_used: str = ""
    tokens_estimated: bool = False

    # -- derived accounting ----------------------------------------------
    @property
    def llm_calls(self) -> int:
        return sum(s.llm_calls for s in self.steps)

    @property
    def prompt_tokens(self) -> int:
        return sum(s.prompt_tokens for s in self.steps)

    @property
    def completion_tokens(self) -> int:
        return sum(s.completion_tokens for s in self.steps)

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    @property
    def tool_calls(self) -> int:
        return sum(s.tool_calls for s in self.steps)

    @property
    def tool_failures(self) -> int:
        return sum(s.tool_failures for s in self.steps)

    @property
    def latency_s(self) -> float:
        return sum(s.latency_s for s in self.steps)

    def as_dict(self) -> dict:
        d = {
            "arm": self.arm,
            "condition": self.condition,
            "question_id": self.question_id,
            "question": self.question,
            "gold_answer": self.gold_answer,
            "question_type": self.question_type,
            "level": self.level,
            "predicted_answer": self.predicted_answer,
            "terminated": self.terminated,
            "error": self.error,
            "seed": self.seed,
            "scores": self.scores,
            "backend": self.backend,
            "model_used": self.model_used,
            "tokens_estimated": self.tokens_estimated,
            "steps": [s.as_dict() for s in self.steps],
        }
        d.update({
            "llm_calls": self.llm_calls,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "tool_calls": self.tool_calls,
            "tool_failures": self.tool_failures,
            "latency_s": round(self.latency_s, 4),
            "n_steps": len(self.steps),
        })
        return d
