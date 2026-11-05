"""Grounded EM - was the answer retrieved, or did the model already know it?

EM scores the answer string against the gold string and stops there. On a
benchmark built from Wikipedia, that lets a model's prior count as a win: an
arm can retrieve nothing useful, answer from memory, and score exactly like an
arm that read the sentence. The accuracy table then reports a tie between
control flows when what it is really reporting is shared training data.

This module scores the stricter thing. An answer is **grounded** when it is
exact-match correct *and* the gold string occurs in the paragraph text that
arm actually put in front of its own model. What that text is differs by arm,
so it is reconstructed from the trace rather than assumed:

- `workflow`, and `hybrid` on its `single` route, call `top_paragraphs`, which
  hands the synthesis node whole paragraphs. The trace records only the
  titles, so the bodies are read back out of the corpus by title.
- `react`, and `hybrid` on its `bridge` route, see what `search` returned:
  the first three sentences of one paragraph. The `(Similar results: ...)`
  trailer is **excluded** - a title glimpsed in a trailer is a pointer, not
  evidence, and counting it is what makes an unread paragraph look read.
- `lookup` hits are included; a `No mention ... found` reply is not.

Two of the sampled questions have yes/no gold answers, which no paragraph can
be said to contain. They are dropped for every arm, so the pairing that makes
the McNemar test valid survives.

    python -m harness.grounding                      # table over results/runs.json
    python -m harness.grounding --by-condition
    python -m harness.grounding --json

Requires the corpus (`python -m harness.dataset download`), because the
workflow arm's paragraph bodies are not in the trace.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

from .aggregate import mcnemar_exact
from .dataset import DEFAULT_PATH
from .metrics import normalize_answer

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RUNS = ROOT / "results" / "runs.json"

# search[] appends "(Similar results: a, b, c)" on its own line. A title seen
# there is not evidence - excluding it is the whole point of the metric.
_TRAILER = re.compile(r"\n\(Similar results:[^\n]*\)")

_NOT_ASSESSABLE = ("yes", "no")


def seen_text(run: dict, item: dict) -> tuple[list[str], str]:
    """Reconstruct the paragraph text this run put in front of its model.

    Returns (titles, text). Titles are in retrieval order, deduplicated.
    """
    bodies = {title: " ".join(sents).strip() for title, sents in item["context"]}
    titles: list[str] = []
    parts: list[str] = []

    for step in run.get("steps", []):
        obs = step.get("observation") or ""
        action = step.get("action") or ""
        if obs.startswith("Error"):
            continue

        if step.get("kind") == "tool" and action.startswith("retrieve["):
            # top_paragraphs: whole bodies, recorded in the trace as titles only.
            for title in bodies:
                if title in obs and title not in titles:
                    titles.append(title)
                    parts.append(bodies[title])
        elif action.startswith("search["):
            body = _TRAILER.sub("", obs)
            parts.append(body)
            title = body.split(":")[0].strip()
            if title in bodies and title not in titles:
                titles.append(title)
        elif action.startswith("lookup[") and obs.startswith("(Result"):
            parts.append(obs)

    return titles, "\n".join(parts)


def is_grounded(gold: str, text: str) -> bool | None:
    """True/False, or None when the gold answer is not a retrievable string."""
    g = normalize_answer(gold)
    if g in _NOT_ASSESSABLE:
        return None
    return g in normalize_answer(text)


def score_runs(runs: list[dict], items: dict[str, dict]) -> dict:
    """Per-(question, condition) cell -> per-arm {em, grounded}.

    Cells whose gold answer is yes/no are omitted entirely, for every arm.
    """
    cells: dict[tuple[str, str], dict[str, dict]] = defaultdict(dict)
    dropped: set[tuple[str, str]] = set()
    for run in runs:
        item = items.get(run["question_id"])
        if item is None:
            raise KeyError(
                f"question {run['question_id']} is not in the corpus - "
                "run `python -m harness.dataset download` first"
            )
        key = (run["question_id"], run["condition"])
        _, text = seen_text(run, item)
        grounded = is_grounded(run["gold_answer"], text)
        if grounded is None:
            dropped.add(key)
            continue
        cells[key][run["arm"]] = {
            "em": run["scores"]["em"] == 1.0,
            "grounded": grounded,
        }
    for key in dropped:
        cells.pop(key, None)
    return {"cells": dict(cells), "dropped": sorted(dropped)}


def by_arm(cells: dict, condition: str | None = None) -> list[dict]:
    rows = []
    arms = sorted({a for c in cells.values() for a in c})
    for arm in arms:
        vals = [c[arm] for k, c in cells.items()
                if arm in c and (condition is None or k[1] == condition)]
        if not vals:
            continue
        correct = sum(v["em"] for v in vals)
        grounded = sum(v["em"] and v["grounded"] for v in vals)
        rows.append({
            "arm": arm,
            "n": len(vals),
            "em": correct / len(vals),
            "grounded_em": grounded / len(vals),
            "correct": correct,
            "grounded": grounded,
            "unevidenced": correct - grounded,
        })
    return rows


def pairwise(cells: dict, condition: str | None = None) -> list[dict]:
    """Exact McNemar on *grounded* correctness, for each pair of arms."""
    arms = sorted({a for c in cells.values() for a in c})
    out = []
    for i, a in enumerate(arms):
        for b in arms[i + 1:]:
            pairs = [c for k, c in cells.items()
                     if a in c and b in c and (condition is None or k[1] == condition)]
            def won(cell, arm):
                return cell[arm]["em"] and cell[arm]["grounded"]
            a_only = sum(won(c, a) and not won(c, b) for c in pairs)
            b_only = sum(won(c, b) and not won(c, a) for c in pairs)
            out.append({
                "a": a, "b": b, "n": len(pairs),
                "a_only": a_only, "b_only": b_only,
                "p_value": mcnemar_exact(a_only, b_only),
            })
    return out


def print_report(scored: dict, by_condition: bool = False) -> None:
    cells = scored["cells"]
    conditions = sorted({k[1] for k in cells})
    n_dropped = len(scored["dropped"])
    print(f"grounded EM over {len(cells)} paired cells "
          f"({n_dropped} dropped: yes/no gold answers)\n")

    hdr = f"{'arm':10s} {'n':>4s} {'EM':>7s} {'grounded':>9s} {'correct':>8s} {'unevidenced':>12s}"
    print(hdr)
    for r in by_arm(cells):
        print(f"{r['arm']:10s} {r['n']:4d} {r['em']:7.3f} {r['grounded_em']:9.3f} "
              f"{r['correct']:8d} {r['unevidenced']:12d}")

    if by_condition:
        for cond in conditions:
            print(f"\n-- {cond}")
            for r in by_arm(cells, cond):
                print(f"{r['arm']:10s} {r['n']:4d} {r['em']:7.3f} {r['grounded_em']:9.3f} "
                      f"{r['correct']:8d} {r['unevidenced']:12d}")

    print("\nexact McNemar on grounded correctness")
    for row in pairwise(cells):
        print(f"  {row['a']:9s} vs {row['b']:9s}  "
              f"{row['a']} only {row['a_only']:3d}, {row['b']} only {row['b_only']:3d}  "
              f"p = {row['p_value']:.4f}")
    for cond in conditions:
        for row in pairwise(cells, cond):
            print(f"  [{cond:17s}] {row['a']:9s} vs {row['b']:9s}  "
                  f"{row['a_only']:3d} / {row['b_only']:3d}  p = {row['p_value']:.4f}")
    print("\nThe pooled test is not independent: the conditions reuse the same "
          "questions,\nso read it as direction, not as significance.")


def main(argv=None):
    p = argparse.ArgumentParser(
        description="Score runs for grounding: correct AND the gold string "
                    "present in what that arm retrieved.")
    p.add_argument("--runs", type=Path, default=DEFAULT_RUNS)
    p.add_argument("--data", type=Path, default=DEFAULT_PATH,
                   help="HotpotQA dev-distractor JSON (needed for paragraph bodies)")
    p.add_argument("--by-condition", action="store_true")
    p.add_argument("--json", action="store_true", help="machine-readable summary")
    args = p.parse_args(argv)

    runs = json.loads(args.runs.read_text())
    if not args.data.exists():
        raise SystemExit(
            f"corpus not found at {args.data}\n"
            "run: python -m harness.dataset download")
    items = {d["_id"]: d for d in json.loads(args.data.read_text())}

    scored = score_runs(runs, items)
    if args.json:
        conditions = sorted({k[1] for k in scored["cells"]})
        print(json.dumps({
            "overall": by_arm(scored["cells"]),
            "by_condition": {c: by_arm(scored["cells"], c) for c in conditions},
            "pairwise": pairwise(scored["cells"]),
            "dropped": scored["dropped"],
        }, indent=1))
    else:
        print_report(scored, by_condition=args.by_condition)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
