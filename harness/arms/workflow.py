"""Arm 1 - the fixed workflow (DAG).

    decompose -> retrieve(hop 1) -> retrieve(hop 2) -> synthesize

Four nodes, always. Exactly 2 LLM calls plus 2 retrievals, every question,
regardless of difficulty. No node decides what runs next; the graph is the
plan, written by the engineer, not the model.

What this buys: a flat cost curve, a latency you can put in an SLA, and a
trace that is the same shape every time, so a failure is localised to a node.
What it costs: the shape is a bet that every question is a 2-hop bridge. When
that bet is wrong - a comparison question, a 1-hop question, a poisoned first
retrieval - there is no mechanism to notice or recover. It fails quietly and
confidently.
"""

from __future__ import annotations

from ..llm import BudgetExceeded, CallBudget
from ..tools import ToolEnv
from ..trace import RunResult, Step

_DECOMPOSE_SYS = (
    "You decompose a multi-hop question into the retrieval queries needed to "
    "answer it. Output exactly two lines:\n"
    "1. <the query that finds the bridge entity>\n"
    "2. <the query that finds the attribute asked about>\n"
    "Output nothing else - no preamble, no explanation."
)

_SYNTH_SYS = (
    "Answer the question using only the provided paragraphs. Reply with the "
    "shortest possible answer span - a name, a date, a number, or exactly "
    "'yes' or 'no'. No sentence, no explanation, no punctuation beyond what "
    "belongs to the answer itself. If the paragraphs do not contain the "
    "answer, reply with your single best guess anyway."
)


def _parse_two_queries(text: str, fallback: str) -> tuple[str, str]:
    lines = [
        ln.strip().lstrip("12.-) ").strip()
        for ln in (text or "").splitlines()
        if ln.strip()
    ]
    lines = [ln for ln in lines if ln]
    if len(lines) >= 2:
        return lines[0], lines[1]
    if len(lines) == 1:
        return lines[0], fallback
    return fallback, fallback


def run(item: dict, llm, env: ToolEnv, budget: CallBudget,
        max_steps: int = 0, seed: int = 0) -> RunResult:
    """`max_steps` is accepted for signature parity with the other arms and
    ignored - the DAG's length is fixed by construction, which is the point."""
    res = RunResult(
        arm="workflow",
        condition="",
        question_id=item["_id"],
        question=item["question"],
        gold_answer=item["answer"],
        question_type=item.get("type", ""),
        level=item.get("level", ""),
        seed=seed,
    )
    question = item["question"]

    # -- node 1: decompose ------------------------------------------------
    try:
        budget.charge()
        out = llm.complete([
            {"role": "system", "content": _DECOMPOSE_SYS},
            {"role": "user", "content": f"Question: {question}"},
        ])
        q1, q2 = _parse_two_queries(out.text, question)
        res.model_used = out.model_used
        res.steps.append(Step(
            index=0, kind="llm", label="decompose",
            action="decompose", observation=f"hop1={q1!r}  hop2={q2!r}",
            llm_calls=1, prompt_tokens=out.prompt_tokens,
            completion_tokens=out.completion_tokens, latency_s=out.latency_s,
        ))
    except BudgetExceeded as e:
        res.terminated, res.error = "budget", str(e)
        q1 = q2 = question
        res.steps.append(Step(index=0, kind="node", label="decompose",
                              note="skipped: budget exhausted"))
    except Exception as e:  # noqa: BLE001
        res.terminated, res.error = "error", f"{type(e).__name__}: {e}"
        return res

    # -- nodes 2 & 3: two fixed retrievals --------------------------------
    paragraphs: list[tuple[str, str]] = []
    for hop, q in enumerate((q1, q2), start=1):
        before_fail = env.n_failed_calls
        hits = env.top_paragraphs(q, k=2)
        paragraphs.extend(hits)
        res.steps.append(Step(
            index=len(res.steps), kind="tool", label=f"retrieve_hop{hop}",
            action=f"retrieve[{q}]",
            observation=(", ".join(t for t, _ in hits) if hits
                         else "retrieval returned nothing"),
            tool_calls=1,
            tool_failures=env.n_failed_calls - before_fail,
            note=("" if hits else "hop produced no paragraphs; "
                  "the DAG continues anyway"),
        ))

    # Deduplicate by title, preserving retrieval order.
    seen, ctx = set(), []
    for title, body in paragraphs:
        if title not in seen:
            seen.add(title)
            ctx.append((title, body))

    # -- node 4: synthesize -----------------------------------------------
    ctx_block = "\n\n".join(f"=== {t} ===\n{b}" for t, b in ctx) or "(no paragraphs retrieved)"
    try:
        budget.charge()
        out = llm.complete([
            {"role": "system", "content": _SYNTH_SYS},
            {"role": "user", "content": f"{ctx_block}\n\nQuestion: {question}\nAnswer:"},
        ])
        res.predicted_answer = (out.text or "").strip().split("\n")[0].strip()
        res.model_used = out.model_used
        res.steps.append(Step(
            index=len(res.steps), kind="llm", label="synthesize",
            action="synthesize", observation=res.predicted_answer,
            llm_calls=1, prompt_tokens=out.prompt_tokens,
            completion_tokens=out.completion_tokens, latency_s=out.latency_s,
        ))
        if res.terminated == "answered" and not res.predicted_answer:
            res.terminated = "malformed"
    except BudgetExceeded as e:
        res.terminated, res.error = "budget", str(e)
        res.steps.append(Step(index=len(res.steps), kind="node",
                              label="synthesize", note="skipped: budget exhausted"))
    except Exception as e:  # noqa: BLE001
        res.terminated, res.error = "error", f"{type(e).__name__}: {e}"

    return res
