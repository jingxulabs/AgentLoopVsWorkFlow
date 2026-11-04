"""Turn raw runs into the summary the dashboard reads.

Two reporting choices worth knowing about:

1. We report the median and p95 of per-question cost and latency, not just the
   mean. The mean hides the thing that most distinguishes these architectures:
   a workflow's cost distribution is nearly a point mass, an agent loop's has a
   long right tail. A comparison on means alone would understate the
   operational difference.

2. The headline metric is cost per correct answer, not accuracy. An
   architecture that is 4 points more accurate for 3x the tokens is a different
   decision at different scales, and that only shows up when you divide.
"""

from __future__ import annotations

import json
import statistics
from math import comb
from pathlib import Path


def _pct(values: list[float], q: float) -> float:
    """Nearest-rank percentile; stable for the small n this harness runs at."""
    if not values:
        return 0.0
    s = sorted(values)
    k = max(0, min(len(s) - 1, int(round(q * (len(s) - 1)))))
    return s[k]


def load_pricing(path: Path | None = None) -> dict:
    path = path or Path(__file__).resolve().parent.parent / "pricing.json"
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def cost_usd(run: dict, pricing: dict) -> float:
    models = pricing.get("models", {})
    rate = models.get(run.get("model_used") or "", None) or models.get("default", {})
    return (
        run.get("prompt_tokens", 0) / 1e6 * rate.get("prompt", 0.0)
        + run.get("completion_tokens", 0) / 1e6 * rate.get("completion", 0.0)
    )


def summarize(runs: list[dict], pricing: dict | None = None) -> dict:
    pricing = pricing or load_pricing()
    cells: dict[tuple, list] = {}
    for r in runs:
        cells.setdefault((r["arm"], r["condition"]), []).append(r)

    out_cells = []
    for (arm, condition), rs in sorted(cells.items()):
        n = len(rs)
        ems = [r["scores"].get("em", 0.0) for r in rs]
        f1s = [r["scores"].get("f1", 0.0) for r in rs]
        costs = [cost_usd(r, pricing) for r in rs]
        lats = [r["latency_s"] for r in rs]
        calls = [r["llm_calls"] for r in rs]
        toks = [r["total_tokens"] for r in rs]
        steps = [r["n_steps"] for r in rs]
        n_correct = sum(ems)

        term: dict[str, int] = {}
        for r in rs:
            term[r["terminated"]] = term.get(r["terminated"], 0) + 1

        out_cells.append({
            "arm": arm,
            "condition": condition,
            "n": n,
            "em": sum(ems) / n,
            "f1": sum(f1s) / n,
            "llm_calls_mean": sum(calls) / n,
            "llm_calls_p95": _pct([float(c) for c in calls], 0.95),
            "llm_calls_max": max(calls),
            "steps_mean": sum(steps) / n,
            "tokens_mean": sum(toks) / n,
            "tokens_p95": _pct([float(t) for t in toks], 0.95),
            "tokens_total": sum(toks),
            "cost_mean": sum(costs) / n,
            "cost_p50": _pct(costs, 0.50),
            "cost_p95": _pct(costs, 0.95),
            "cost_total": sum(costs),
            # The decision metric. None when nothing was answered correctly -
            # rendered as "n/a" rather than silently as infinity.
            "cost_per_correct": (sum(costs) / n_correct) if n_correct else None,
            "latency_mean": sum(lats) / n,
            "latency_p50": _pct(lats, 0.50),
            "latency_p95": _pct(lats, 0.95),
            # Coefficient of variation: the one-number summary of "can I put
            # this in an SLA".
            "cost_cv": (statistics.pstdev(costs) / (sum(costs) / n)) if sum(costs) else 0.0,
            "steps_cv": (statistics.pstdev(steps) / (sum(steps) / n)) if sum(steps) else 0.0,
            "tool_calls_mean": sum(r["tool_calls"] for r in rs) / n,
            "tool_failures_total": sum(r["tool_failures"] for r in rs),
            "terminated": term,
            "nonterminating_rate": (term.get("step_limit", 0) + term.get("malformed", 0)) / n,
        })

    # Router accuracy for the hybrid arm, scored against HotpotQA's own `type`.
    router = {}
    hybrid_runs = [r for r in runs if r["arm"] == "hybrid" and "route" in r.get("scores", {})]
    if hybrid_runs:
        correct = sum(
            1 for r in hybrid_runs
            if (r["scores"]["route"] == "comparison") == (r["question_type"] == "comparison")
        )
        router = {
            "n": len(hybrid_runs),
            "comparison_detection_accuracy": correct / len(hybrid_runs),
            "route_counts": _counts(r["scores"]["route"] for r in hybrid_runs),
        }

    return {
        "cells": out_cells,
        "arms": sorted({c["arm"] for c in out_cells}),
        "conditions": sorted({c["condition"] for c in out_cells}),
        "by_arm": by_arm(runs, pricing),
        "pairwise": pairwise(runs, pricing),
        "by_question_type": by_question_type(runs),
        "router": router,
        "pricing_source": pricing.get("_source", "unknown"),
        "pricing_note": pricing.get("_comment", ""),
    }


def _counts(it) -> dict:
    out: dict[str, int] = {}
    for v in it:
        out[v] = out.get(v, 0) + 1
    return out


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar p-value for a paired win/loss count.

    `b` and `c` are the discordant cells: questions one arm got right and the
    other got wrong. Concordant pairs carry no information about which arm is
    better, so they are correctly ignored - that is the whole point of pairing,
    and it is why this is far more sensitive than comparing two means at n=20.
    """
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(comb(n, i) for i in range(k + 1)) * (0.5 ** n)
    return min(1.0, 2.0 * tail)


def pairwise(runs: list[dict], pricing: dict | None = None) -> list[dict]:
    """Head-to-head comparison on identical questions.

    For every (condition, arm-pair) this reports how many questions each arm
    got right that the other got wrong. Two arms can post the same mean EM
    while disagreeing on half the questions - which means they have different
    competencies, not equivalent ones. Averages hide that completely; this is
    the measurement that separates the architectures.
    """
    pricing = pricing or load_pricing()
    # (condition, question_id, seed) -> {arm: (em, cost, calls)}
    cells: dict[tuple, dict] = {}
    for r in runs:
        key = (r["condition"], r["question_id"], r["seed"])
        cells.setdefault(key, {})[r["arm"]] = (
            r["scores"].get("em", 0.0), cost_usd(r, pricing), r["llm_calls"],
        )

    arms = sorted({r["arm"] for r in runs})
    conds = sorted({r["condition"] for r in runs})
    out = []
    for cond in conds:
        keys = [k for k in cells if k[0] == cond]
        for i, a in enumerate(arms):
            for b_arm in arms[i + 1:]:
                both = a_only = b_only = neither = 0
                extra_cost = 0.0
                for k in keys:
                    cell = cells[k]
                    if a not in cell or b_arm not in cell:
                        continue
                    ea, eb = cell[a][0] >= 1.0, cell[b_arm][0] >= 1.0
                    extra_cost += cell[b_arm][1] - cell[a][1]
                    if ea and eb:
                        both += 1
                    elif ea:
                        a_only += 1
                    elif eb:
                        b_only += 1
                    else:
                        neither += 1
                n = both + a_only + b_only + neither
                if not n:
                    continue
                out.append({
                    "condition": cond,
                    "arm_a": a,
                    "arm_b": b_arm,
                    "n": n,
                    "both_correct": both,
                    "only_a": a_only,
                    "only_b": b_only,
                    "neither": neither,
                    "disagreement_rate": (a_only + b_only) / n,
                    "net_b_minus_a": b_only - a_only,
                    "p_value": mcnemar_exact(a_only, b_only),
                    # What arm B's extra spend bought, per question.
                    "extra_cost_b": extra_cost / n,
                    "cost_per_extra_correct": (
                        (extra_cost / (b_only - a_only)) if (b_only - a_only) > 0 else None
                    ),
                })
    return out


def by_arm(runs: list[dict], pricing: dict | None = None) -> list[dict]:
    """One row per architecture, rolled up across every condition.

    Conditions carry equal n, so a plain mean weights them equally - that is
    intentional: the roll-up answers "across a mix of easy and adversarial
    environments, what does each architecture cost and deliver", which is closer
    to a production question than any single condition is.

    `latency_per_call_p50` is reported instead of per-question p95 because the
    gateway's own tail (~7% of calls at ~48s) swamps the architectural signal in
    any p95; the architecture-attributable part of latency is the call count.
    """
    pricing = pricing or load_pricing()
    out = []
    for arm in sorted({r["arm"] for r in runs}):
        rs = [r for r in runs if r["arm"] == arm]
        n = len(rs)
        ems = [r["scores"].get("em", 0.0) for r in rs]
        costs = [cost_usd(r, pricing) for r in rs]
        calls = [float(r["llm_calls"]) for r in rs]
        toks = [float(r["total_tokens"]) for r in rs]
        per_call = sorted(
            s["latency_s"] for r in rs for s in r["steps"] if s.get("llm_calls")
        )
        n_correct = sum(ems)
        term: dict[str, int] = {}
        for r in rs:
            term[r["terminated"]] = term.get(r["terminated"], 0) + 1
        mean_cost = sum(costs) / n
        out.append({
            "arm": arm,
            "n": n,
            "em": sum(ems) / n,
            "f1": sum(r["scores"].get("f1", 0.0) for r in rs) / n,
            "llm_calls_mean": sum(calls) / n,
            "llm_calls_p95": _pct(calls, 0.95),
            "llm_calls_max": int(max(calls)),
            "tokens_mean": sum(toks) / n,
            "tokens_p95": _pct(toks, 0.95),
            "tokens_p95_over_mean": (_pct(toks, 0.95) / (sum(toks) / n)) if sum(toks) else 0.0,
            "cost_mean": mean_cost,
            "cost_per_correct": (sum(costs) / n_correct) if n_correct else None,
            "cost_cv": (statistics.pstdev(costs) / mean_cost) if mean_cost else 0.0,
            "latency_per_call_p50": _pct(per_call, 0.50) if per_call else 0.0,
            "tool_calls_mean": sum(r["tool_calls"] for r in rs) / n,
            "tool_failures_total": sum(r["tool_failures"] for r in rs),
            "clean_termination_rate": term.get("answered", 0) / n,
            "terminated": term,
        })
    return out


def by_question_type(runs: list[dict]) -> list[dict]:
    """EM split by HotpotQA's own `type`.

    The sharpest single differentiator between a fixed graph and a loop: a
    2-hop DAG is a bet that every question bridges through an intermediate
    entity. Comparison questions do not - they need two independent lookups -
    so this is where the fixed shape should visibly cost accuracy.
    """
    cells: dict[tuple, list] = {}
    for r in runs:
        cells.setdefault((r["arm"], r["condition"], r["question_type"]), []).append(r)
    out = []
    for (arm, cond, qtype), rs in sorted(cells.items()):
        if not qtype:
            continue
        out.append({
            "arm": arm, "condition": cond, "question_type": qtype, "n": len(rs),
            "em": sum(x["scores"].get("em", 0.0) for x in rs) / len(rs),
            "f1": sum(x["scores"].get("f1", 0.0) for x in rs) / len(rs),
            "llm_calls_mean": sum(x["llm_calls"] for x in rs) / len(rs),
        })
    return out


def print_table(summary: dict) -> None:
    ba = summary.get("by_arm") or []
    if ba:
        print("Performance by architecture (rolled up across all conditions):")
        h = (f"{'arm':<10} {'n':>4} {'EM':>6} {'F1':>6} {'calls':>6} {'p95':>5} "
             f"{'max':>4} {'tokens':>7} {'tok p95':>8} {'tail':>6} {'s/call':>7} "
             f"{'$/correct':>10} {'cost CV':>8} {'clean end':>10}")
        print(h)
        print("-" * len(h))
        for r in ba:
            cpc = (f"{r['cost_per_correct']:.5f}"
                   if r["cost_per_correct"] is not None else "n/a")
            print(f"{r['arm']:<10} {r['n']:>4} {r['em']:>6.3f} {r['f1']:>6.3f} "
                  f"{r['llm_calls_mean']:>6.2f} {r['llm_calls_p95']:>5.0f} "
                  f"{r['llm_calls_max']:>4} {r['tokens_mean']:>7.0f} "
                  f"{r['tokens_p95']:>8.0f} {r['tokens_p95_over_mean']:>5.1f}x "
                  f"{r['latency_per_call_p50']:>7.2f} {cpc:>10} "
                  f"{r['cost_cv']:>8.2f} {100 * r['clean_termination_rate']:>9.0f}%")
        print()

    hdr = (f"{'arm':<10} {'condition':<17} {'n':>4} {'EM':>6} {'F1':>6} "
           f"{'calls':>6} {'tok':>7} {'p95tok':>7} {'lat_p50':>8} {'lat_p95':>8} "
           f"{'$/corr':>9} {'stall%':>7}")
    print(hdr)
    print("-" * len(hdr))
    for c in summary["cells"]:
        cpc = f"{c['cost_per_correct']:.5f}" if c["cost_per_correct"] is not None else "n/a"
        print(
            f"{c['arm']:<10} {c['condition']:<17} {c['n']:>4} "
            f"{c['em']:>6.3f} {c['f1']:>6.3f} {c['llm_calls_mean']:>6.2f} "
            f"{c['tokens_mean']:>7.0f} {c['tokens_p95']:>7.0f} "
            f"{c['latency_p50']:>8.2f} {c['latency_p95']:>8.2f} {cpc:>9} "
            f"{100 * c['nonterminating_rate']:>6.1f}%"
        )
    pw = summary.get("pairwise") or []
    if pw:
        print("\nHead-to-head on identical questions (paired; McNemar exact):")
        h2 = (f"{'condition':<17} {'A':<10} {'B':<10} {'both':>5} {'A only':>7} "
              f"{'B only':>7} {'none':>5} {'disagree':>9} {'p':>7} {'$/extra':>10}")
        print(h2)
        print("-" * len(h2))
        for r in pw:
            cpe = (f"{r['cost_per_extra_correct']:.5f}"
                   if r["cost_per_extra_correct"] is not None else "-")
            print(f"{r['condition']:<17} {r['arm_a']:<10} {r['arm_b']:<10} "
                  f"{r['both_correct']:>5} {r['only_a']:>7} {r['only_b']:>7} "
                  f"{r['neither']:>5} {100 * r['disagreement_rate']:>8.1f}% "
                  f"{r['p_value']:>7.3f} {cpe:>10}")

    bt = summary.get("by_question_type") or []
    if bt and len({r["question_type"] for r in bt}) > 1:
        print("\nEM by question type (the fixed 2-hop bet is only right for 'bridge'):")
        h3 = f"{'arm':<10} {'condition':<17} {'type':<12} {'n':>4} {'EM':>6} {'calls':>6}"
        print(h3)
        print("-" * len(h3))
        for r in bt:
            print(f"{r['arm']:<10} {r['condition']:<17} {r['question_type']:<12} "
                  f"{r['n']:>4} {r['em']:>6.3f} {r['llm_calls_mean']:>6.2f}")

    if summary.get("router"):
        r = summary["router"]
        print(f"\nhybrid router: comparison detected correctly in "
              f"{100 * r['comparison_detection_accuracy']:.1f}% of {r['n']} runs "
              f"({r['route_counts']})")
    if summary.get("pricing_source") == "placeholder":
        print("\nNOTE: dollar figures use placeholder rates from pricing.json. "
              "Token counts are real; costs are not authoritative.")
