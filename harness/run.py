"""The experiment runner.

    python -m harness.run --backend mock --n 30
    python -m harness.run --backend openai --n 50 --repeats 3

Fairness invariants this file is responsible for, and which the whole
comparison rests on:

  * Every arm sees the same questions, in the same order, for a given
    (condition, seed). The comparison is paired.
  * Every arm gets a fresh ToolEnv per question, seeded from
    (question_id, seed), so an injected fault lands on the same question for
    every arm rather than being re-rolled per arm.
  * Every arm shares one prompt vocabulary and one retriever.
  * Only control flow differs.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from . import chain
from . import conditions as conditions_mod
from .aggregate import load_pricing, print_table, summarize
from .arms import ARMS
from .dataset import DEFAULT_PATH, gold_titles, load_items, sample
from .llm import CallBudget, build_llm
from .metrics import score_answer
from .tools import ToolEnv

ROOT = Path(__file__).resolve().parent.parent


def _env_seed(question_id: str, seed: int) -> int:
    """Stable per-question seed, identical across arms."""
    return (hash((question_id, seed)) & 0x7FFFFFFF)


def _load_credentials(env_file: Path | None, env_dir: Path | None = None) -> None:
    """Push OPENAI_API_KEY (and friends) into os.environ from a dotenv file.

    Search order: --env-dir, --env-file, $AGENTLOOP_ENV_FILE, ./.env. The first
    one that exists wins; already-exported variables are not overridden, so an
    exported key beats a file.

    Pointing at an existing credentials file is preferred over copying one here
    - a secret duplicated into a second location is a second thing to rotate and
    a second thing to leak. Only the resolved *path* is ever printed; values go
    straight into the environment and are never read or logged by this harness.
    """
    try:
        from dotenv import load_dotenv
    except ImportError:
        return  # credentials may already be exported in the environment

    candidates: list[Path] = []
    if env_dir:
        candidates.append(Path(env_dir).expanduser() / ".env")
    if env_file:
        candidates.append(Path(env_file).expanduser())
    from_env = os.getenv("AGENTLOOP_ENV_FILE")
    if from_env:
        candidates.append(Path(from_env).expanduser())
    candidates.append(ROOT / ".env")

    for c in candidates:
        if c.is_file():
            load_dotenv(c, override=False)
            print(f"credentials loaded from {c}")
            return
    tried = ", ".join(str(c) for c in candidates)
    if env_file or env_dir:
        raise SystemExit(f"no credentials file found. Tried: {tried}")


def _preflight(model: str | None = None) -> None:
    """Make one real completion before the sweep starts.

    Without this, a bad key or an unreachable endpoint produces one auth error
    per question - a full sweep would spend its whole wall clock recording the
    same failure 2,250 times. The probe also checks that the endpoint returns a
    `usage` block, because every cost column in the results depends on it.
    """
    from .llm import MissingAPIKey, OpenAILLM, base_url, get_api_key

    try:
        get_api_key()
    except MissingAPIKey as e:
        raise SystemExit(str(e)) from None

    try:
        res = OpenAILLM(model=model, max_retries=2).complete(
            [{"role": "user", "content": "Reply with the single word: ok"}],
            max_tokens=5,
        )
    except Exception as e:  # noqa: BLE001
        raise SystemExit(
            f"preflight failed against {base_url()}: {type(e).__name__}: {e}\n\n"
            "Check OPENAI_API_KEY, OPENAI_BASE_URL and OPENAI_MODEL, or use "
            "--backend mock to run offline."
        ) from None

    print(f"preflight OK - {res.model_used} at {base_url()}")
    if res.total_tokens == 0:
        print("  WARNING: no `usage` block returned by this endpoint. Token and "
              "cost columns will read zero; the comparison loses its main metric.")


def run_cell(arm_name, items, condition, llm, backend, seed, verbose=False):
    arm_fn = ARMS[arm_name]
    out = []
    for i, item in enumerate(items, 1):
        env = ToolEnv(
            context=item["context"],
            gold_titles=gold_titles(item),
            flaky_p=condition.flaky_p,
            poison_first_search=condition.poison_first_search,
            seed=_env_seed(item["_id"], seed),
        )
        budget = CallBudget(max_calls=condition.max_llm_calls)
        t0 = time.perf_counter()
        try:
            res = arm_fn(item, llm, env, budget,
                         max_steps=condition.max_steps, seed=seed)
        except Exception as e:  # noqa: BLE001 - never let one question kill a sweep
            from .trace import RunResult
            res = RunResult(
                arm=arm_name, condition=condition.name, question_id=item["_id"],
                question=item["question"], gold_answer=item["answer"],
                question_type=item.get("type", ""), level=item.get("level", ""),
                terminated="error", error=f"{type(e).__name__}: {e}", seed=seed,
            )
        res.condition = condition.name
        res.backend = backend
        res.tokens_estimated = backend == "mock"
        res.scores.update(score_answer(res.predicted_answer, item["answer"]))
        d = res.as_dict()
        # Wall-clock for the whole question, including any retry sleeps that
        # the per-step latencies do not capture.
        d["wall_clock_s"] = round(time.perf_counter() - t0, 4)
        out.append(d)

        if verbose:
            mark = "OK " if res.scores["em"] else "XX "
            print(f"  {mark}[{i}/{len(items)}] {arm_name}/{condition.name} "
                  f"calls={res.llm_calls} steps={len(res.steps)} "
                  f"term={res.terminated} "
                  f"pred={res.predicted_answer[:40]!r} gold={item['answer'][:40]!r}")
        else:
            print(f"\r  {arm_name}/{condition.name}: {i}/{len(items)}",
                  end="", flush=True)
    if not verbose:
        print()
    return out


def main(argv=None):
    p = argparse.ArgumentParser(
        description="Compare an agent loop against a fixed workflow on HotpotQA.")
    p.add_argument("--backend", choices=["mock", "openai"], default="mock",
                   help="mock runs offline with no credentials (default); "
                        "openai calls any OpenAI-compatible endpoint")
    p.add_argument("--model", default=None, help="overrides OPENAI_MODEL")
    p.add_argument("--arms", default="workflow,react,hybrid")
    p.add_argument("--conditions", default="clean,flaky_tools,hop_mismatch,"
                                           "distractor_heavy,budget_cap")
    p.add_argument("--n", type=int, default=20, help="questions per condition")
    p.add_argument("--repeats", type=int, default=1,
                   help="repeat the whole sweep with different seeds; "
                        "variance is the point, so >1 is recommended for real runs")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--data", type=Path, default=DEFAULT_PATH)
    p.add_argument("--out", type=Path, default=ROOT / "results")
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--env-file", type=Path, default=None,
                   help="path to a dotenv file holding OPENAI_API_KEY. Values "
                        "are loaded straight into the process environment and "
                        "never logged.")
    p.add_argument("--env-dir", type=Path, default=None,
                   help="directory containing that dotenv file, e.g. another "
                        "project's backend/ - avoids duplicating the secret here")
    args = p.parse_args(argv)

    if args.backend == "openai":
        _load_credentials(args.env_file, args.env_dir)
        _preflight(args.model)

    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    for a in arms:
        if a not in ARMS:
            raise SystemExit(f"unknown arm {a!r}. Available: {', '.join(ARMS)}")
    conds = [conditions_mod.get(c.strip()) for c in args.conditions.split(",") if c.strip()]

    # Only touch the dataset if a condition actually draws from it, so a
    # chain-only run needs no download.
    items_all: list[dict] = []
    if any(c.source == "hotpot" for c in conds):
        items_all = load_items(args.data)
        print(f"loaded {len(items_all)} HotpotQA dev-distractor items from {args.data}")

    runs = []
    t_start = time.perf_counter()
    for rep in range(args.repeats):
        seed = args.seed + rep
        llm = build_llm(args.backend, model=args.model, seed=seed)
        for cond in conds:
            # Built/sampled once per (condition, seed) and reused across arms -
            # this is what makes the comparison paired.
            if cond.source == "chain":
                items = chain.make_items(args.n, cond.chain_depths, seed=seed)
            else:
                items = sample(items_all, args.n, types=cond.types,
                               levels=cond.levels, seed=seed)
            print(f"\n[rep {rep + 1}/{args.repeats}] condition={cond.name} "
                  f"n={len(items)} — {cond.description}")
            for arm in arms:
                runs.extend(run_cell(arm, items, cond, llm, args.backend,
                                     seed, verbose=args.verbose))

    elapsed = time.perf_counter() - t_start
    args.out.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    pricing = load_pricing()
    summary = summarize(runs, pricing)
    summary["meta"] = {
        "generated_at": stamp,
        "backend": args.backend,
        "model": args.model or "(env default)",
        "n_per_condition": args.n,
        "repeats": args.repeats,
        "seed": args.seed,
        "arms": arms,
        "conditions": [c.name for c in conds],
        "condition_detail": {
            c.name: {"description": c.description, "hypothesis": c.hypothesis}
            for c in conds
        },
        "total_runs": len(runs),
        "wall_clock_s": round(elapsed, 2),
        "dataset": "HotpotQA dev-distractor (CC BY-SA 4.0)",
    }

    runs_path = args.out / "runs.json"
    summary_path = args.out / "summary.json"

    # runs.json and summary.json are committed so the dashboard works on a
    # fresh clone - which means the quick start's mock sweep lands on top of a
    # real one. The timestamped copy below is gitignored and does not save you.
    # Both files are in git, so this is recoverable, but only if you notice.
    if args.backend == "mock" and runs_path.exists():
        try:
            prior = json.loads(runs_path.read_text(encoding="utf-8"))
            live = sorted({r.get("backend") for r in prior} - {"mock", None})
        except (ValueError, OSError, AttributeError, TypeError):
            live = []
        if live:
            print(f"\n  ! {runs_path} holds results from a live run "
                  f"({'/'.join(live)}); this mock sweep replaces them.")
            print(f"    restore:   git checkout -- {runs_path} {summary_path}")
            print("    keep both: re-run with --out results_mock")

    args.out.mkdir(parents=True, exist_ok=True)
    runs_path.write_text(json.dumps(runs, indent=1), encoding="utf-8")
    summary_path.write_text(json.dumps(summary, indent=1), encoding="utf-8")
    # Timestamped copy of this sweep. Gitignored: it is a local audit trail,
    # not a backup of whatever runs.json held a moment ago.
    (args.out / f"runs_{stamp}.json").write_text(json.dumps(runs), encoding="utf-8")

    print(f"\n{len(runs)} runs in {elapsed:.1f}s -> {runs_path.name}, {summary_path.name}\n")
    print_table(summary)
    # file:// cannot fetch the results JSON, so the dashboard must be served.
    print(f"\nOpen the dashboard:  {sys.executable} {ROOT / 'serve.py'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
