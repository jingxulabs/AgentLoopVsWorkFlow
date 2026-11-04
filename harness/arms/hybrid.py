"""Arm 3 - router + bounded loop.

    classify -> { 1-hop: retrieve+answer
                | comparison: retrieve both entities, answer
                | else: a ReAct loop capped at 4 steps }

The deterministic skeleton is the engineer's; the adaptivity is bounded and
only paid for on the questions that need it. This is the shape most production
systems converge on, and it exists in this experiment so the comparison is not
a strawman two-way fight.

Worth watching in the results: the router is itself an LLM call that can be
wrong, and a misroute is more expensive than either pure arm - you pay for the
classification and then run the wrong plan. The dashboard surfaces router
accuracy against HotpotQA's own `type` label so that cost is visible rather
than assumed away.
"""

from __future__ import annotations

from ..llm import BudgetExceeded, CallBudget
from ..tools import ToolEnv
from ..trace import RunResult, Step
from . import react as react_arm
from .workflow import _SYNTH_SYS

_ROUTER_SYS = (
    "Classify how a multi-hop question must be answered. Reply with exactly "
    "one word:\n"
    "  single     - answerable from one paragraph about one entity\n"
    "  comparison - compares two named entities on some attribute\n"
    "  bridge     - needs an intermediate entity found by a first lookup\n"
    "Output the single word only."
)

_VALID_ROUTES = ("single", "comparison", "bridge")


def _classify(text: str) -> str:
    low = (text or "").strip().lower()
    for route in _VALID_ROUTES:
        if route in low:
            return route
    return "bridge"  # the safest default: the most general plan


def _answer_from(llm, budget, res, item, ctx, label):
    ctx_block = "\n\n".join(f"=== {t} ===\n{b}" for t, b in ctx) or "(nothing retrieved)"
    budget.charge()
    out = llm.complete([
        {"role": "system", "content": _SYNTH_SYS},
        {"role": "user", "content":
         f"{ctx_block}\n\nQuestion: {item['question']}\nAnswer:"},
    ])
    res.predicted_answer = (out.text or "").strip().split("\n")[0].strip()
    res.model_used = out.model_used
    res.steps.append(Step(
        index=len(res.steps), kind="llm", label=label,
        action=label, observation=res.predicted_answer,
        llm_calls=1, prompt_tokens=out.prompt_tokens,
        completion_tokens=out.completion_tokens, latency_s=out.latency_s,
    ))


def run(item: dict, llm, env: ToolEnv, budget: CallBudget,
        max_steps: int = 4, seed: int = 0) -> RunResult:
    res = RunResult(
        arm="hybrid",
        condition="",
        question_id=item["_id"],
        question=item["question"],
        gold_answer=item["answer"],
        question_type=item.get("type", ""),
        level=item.get("level", ""),
        seed=seed,
    )
    question = item["question"]

    # -- node 1: route ----------------------------------------------------
    try:
        budget.charge()
        out = llm.complete([
            {"role": "system", "content": _ROUTER_SYS},
            {"role": "user", "content": f"Question: {question}"},
        ])
        route = _classify(out.text)
        res.model_used = out.model_used
        res.steps.append(Step(
            index=0, kind="llm", label="route",
            action="classify", observation=route,
            llm_calls=1, prompt_tokens=out.prompt_tokens,
            completion_tokens=out.completion_tokens, latency_s=out.latency_s,
            note=f"raw={out.text.strip()[:60]!r}",
        ))
    except BudgetExceeded as e:
        res.terminated, res.error = "budget", str(e)
        return res
    except Exception as e:  # noqa: BLE001
        res.terminated, res.error = "error", f"{type(e).__name__}: {e}"
        return res

    res.scores["route"] = route

    try:
        if route in ("single", "comparison"):
            k = 2 if route == "single" else 4
            before_fail = env.n_failed_calls
            ctx = env.top_paragraphs(question, k=k)
            res.steps.append(Step(
                index=len(res.steps), kind="tool", label=f"retrieve_{route}",
                action=f"retrieve[{question[:60]}]",
                observation=", ".join(t for t, _ in ctx) or "nothing retrieved",
                tool_calls=1, tool_failures=env.n_failed_calls - before_fail,
            ))
            _answer_from(llm, budget, res, item, ctx, f"answer_{route}")
            if not res.predicted_answer:
                res.terminated = "malformed"
            return res

        # -- bridge: hand off to a tightly bounded ReAct loop --------------
        sub = react_arm.run(item, llm, env, budget, max_steps=max_steps, seed=seed)
        offset = len(res.steps)
        for s in sub.steps:
            s.index += offset
            s.label = f"loop/{s.label}"
            res.steps.append(s)
        res.predicted_answer = sub.predicted_answer
        res.terminated = sub.terminated
        res.error = sub.error
        res.model_used = sub.model_used or res.model_used
        return res

    except BudgetExceeded as e:
        res.terminated, res.error = "budget", str(e)
        return res
    except Exception as e:  # noqa: BLE001
        res.terminated, res.error = "error", f"{type(e).__name__}: {e}"
        return res
