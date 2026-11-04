"""Arm 2 - the ReAct agent loop.

    while not done:  Thought -> Action -> Observation

The model chooses every action, including when to stop. The prompt is the
plan. Control flow is data, decided at run time.

What this buys: the number of hops is discovered, not assumed. A failed tool
call is just another observation, so retrying is already in the vocabulary -
no extra code path. A question that needs one hop costs one hop; a question
that needs four can have four.
What it costs: the step count - and therefore cost and latency - is a random
variable with a long tail. A model that never emits Finish burns the whole
budget. Malformed actions are a live failure mode. The same question asked
twice can take different paths, which makes it hard to put in an SLA and
harder to debug.

Action grammar is text-parsed rather than native tool-calling, matching the
original ReAct formulation and keeping the arm portable across gateway routes
that do not expose a `tools` parameter.
"""

from __future__ import annotations

import re

from ..llm import BudgetExceeded, CallBudget
from ..tools import ToolEnv
from ..trace import RunResult, Step

_SYS = (
    "Answer a multi-hop question by interleaving reasoning and tool use.\n"
    "On each turn output EXACTLY two lines:\n"
    "Thought: <one sentence of reasoning>\n"
    "Action: <one of the actions below>\n\n"
    "Actions:\n"
    "  search[query]    - retrieve the best-matching paragraph\n"
    "  lookup[keyword]  - find the next sentence containing keyword in the "
    "paragraph you last searched\n"
    "  finish[answer]   - give the final answer and stop\n\n"
    "Rules:\n"
    "- Use square brackets exactly as shown. Emit one action per turn.\n"
    "- If an observation is an error, you may retry or try a different query.\n"
    "- The final answer must be the shortest possible span: a name, a date, a "
    "number, or exactly 'yes' or 'no'.\n"
    "- Call finish[...] as soon as you can answer. Do not keep searching to "
    "confirm an answer you already have."
)

# Tolerant of `search[x]`, `search(x)`, `search: x`, and a bare `Action: x`.
_ACTION_RE = re.compile(
    r"^\s*(search|lookup|finish)\s*(?:\[(.*?)\]|\((.*?)\)|:\s*(.*?))\s*$",
    re.IGNORECASE | re.DOTALL,
)


def _truncate_at_observation(text: str) -> str:
    """Cut the model's output at any 'Observation:' it wrote itself.

    With a `stop` sequence the gateway would do this for us, but newer models
    reject `stop`, so a generation can run on and invent its own observation.
    Feeding that back would let the loop hallucinate evidence and "answer"
    without ever calling a tool - which would flatter this arm badly.
    """
    i = text.lower().find("\nobservation:")
    return text[:i] if i != -1 else text


def _parse(text: str) -> tuple[str, str, str, str]:
    """-> (thought, action_name, argument, parse_error)."""
    text = _truncate_at_observation(text or "")
    thought, action_line = "", ""
    for raw in (text or "").splitlines():
        ln = raw.strip()
        if not ln:
            continue
        low = ln.lower()
        if low.startswith("thought:"):
            thought = ln.split(":", 1)[1].strip()
        elif low.startswith("action:"):
            action_line = ln.split(":", 1)[1].strip()
            break
        elif not thought and not action_line:
            thought = ln

    if not action_line:
        return thought, "", "", "no Action line found"

    m = _ACTION_RE.match(action_line)
    if not m:
        return thought, "", "", f"unparseable action: {action_line!r}"
    name = m.group(1).lower()
    arg = next((g for g in m.group(2, 3, 4) if g is not None), "")
    return thought, name, arg.strip(), ""


def run(item: dict, llm, env: ToolEnv, budget: CallBudget,
        max_steps: int = 8, seed: int = 0) -> RunResult:
    res = RunResult(
        arm="react",
        condition="",
        question_id=item["_id"],
        question=item["question"],
        gold_answer=item["answer"],
        question_type=item.get("type", ""),
        level=item.get("level", ""),
        seed=seed,
    )
    transcript = f"Question: {item['question']}"
    consecutive_parse_errors = 0

    for i in range(max_steps):
        try:
            budget.charge()
            out = llm.complete(
                [
                    {"role": "system", "content": _SYS},
                    {"role": "user", "content": transcript},
                ],
                stop=["Observation:"],
            )
        except BudgetExceeded as e:
            res.terminated, res.error = "budget", str(e)
            res.steps.append(Step(index=len(res.steps), kind="node",
                                  label=f"step_{i + 1}",
                                  note="halted: LLM call budget exhausted"))
            break
        except Exception as e:  # noqa: BLE001
            res.terminated, res.error = "error", f"{type(e).__name__}: {e}"
            break

        res.model_used = out.model_used
        thought, name, arg, parse_err = _parse(out.text)

        step = Step(
            index=len(res.steps), kind="llm", label=f"step_{i + 1}",
            thought=thought,
            action=(f"{name}[{arg}]" if name else (out.text or "").strip()[:120]),
            llm_calls=1, prompt_tokens=out.prompt_tokens,
            completion_tokens=out.completion_tokens, latency_s=out.latency_s,
        )

        if parse_err:
            consecutive_parse_errors += 1
            step.note = parse_err
            obs = (
                "Error: could not parse your action. Reply with exactly two "
                "lines, the second being Action: search[...], lookup[...], or "
                "finish[...]."
            )
            step.observation = obs
            res.steps.append(step)
            transcript += f"\n{_truncate_at_observation(out.text).strip()}\nObservation: {obs}"
            # Two bad actions in a row means the loop is not going to recover
            # on its own; stop rather than spend the rest of the budget.
            if consecutive_parse_errors >= 2:
                res.terminated = "malformed"
                break
            continue

        consecutive_parse_errors = 0

        if name == "finish":
            res.predicted_answer = arg.strip()
            step.observation = f"FINAL: {res.predicted_answer}"
            step.kind = "llm"
            res.steps.append(step)
            res.terminated = "answered" if res.predicted_answer else "malformed"
            break

        before_fail = env.n_failed_calls
        tc = env.call(name, arg)
        step.observation = tc.result
        step.tool_calls = 1
        step.tool_failures = env.n_failed_calls - before_fail
        if tc.injected_fault:
            step.note = f"injected: {tc.injected_fault}"
        res.steps.append(step)
        transcript += (
            f"\nThought: {thought}\nAction: {name}[{arg}]\nObservation: {tc.result}"
        )
    else:
        # Loop ran to max_steps without ever emitting finish[...]. This is the
        # agent-loop failure mode the workflow arm structurally cannot have.
        res.terminated = "step_limit"

    if res.terminated in ("step_limit", "budget") and not res.predicted_answer:
        # Salvage: force one answer from whatever was gathered, so the arm is
        # scored on its best effort rather than an empty string. Charged to the
        # budget like any other call, and recorded as a distinct node so the
        # dashboard can separate "answered" from "answered under duress".
        obs = "\n".join(
            s.observation for s in res.steps if s.observation and s.tool_calls
        )
        try:
            budget.charge()
            out = llm.complete([
                {"role": "system", "content":
                 "Give the shortest possible answer span from the notes. "
                 "A name, a date, a number, or exactly 'yes' or 'no'. Nothing else."},
                {"role": "user", "content":
                 f"Notes:\n{obs}\n\nQuestion: {item['question']}\nAnswer:"},
            ])
            res.predicted_answer = (out.text or "").strip().split("\n")[0].strip()
            res.steps.append(Step(
                index=len(res.steps), kind="llm", label="forced_answer",
                action="forced_answer", observation=res.predicted_answer,
                llm_calls=1, prompt_tokens=out.prompt_tokens,
                completion_tokens=out.completion_tokens, latency_s=out.latency_s,
                note=f"loop ended as '{res.terminated}'; answer forced from notes",
            ))
        except Exception:  # noqa: BLE001 - salvage is best-effort by design
            pass

    return res
