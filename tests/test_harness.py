"""Tests for the parts where a silent bug would invalidate the comparison.

Run:  .venv/bin/python -m pytest tests/ -q
      (or .venv/bin/python tests/test_harness.py for a dependency-free run)

The scoring and the action parser get the most attention: a wrong metric makes
every chart wrong, and a too-strict parser would hand the workflow arm an
unearned win by inflating the loop's malformed-action rate.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from harness.arms.react import _parse  # noqa: E402
from harness.arms.workflow import _parse_two_queries  # noqa: E402
from harness.conditions import CONDITIONS  # noqa: E402
from harness.dataset import sample  # noqa: E402
from harness.llm import BudgetExceeded, CallBudget  # noqa: E402
from harness.metrics import exact_match_score, f1_score, normalize_answer  # noqa: E402
from harness.tools import ToolEnv  # noqa: E402

CONTEXT = [
    ["Scott Derrickson", ["Scott Derrickson (born July 16, 1966) is an American director.",
                          "He lives in Los Angeles."]],
    ["Ed Wood", ["Edward Davis Wood Jr. was an American filmmaker.",
                 "He was born in Poughkeepsie, New York."]],
    ["Tyler Bates", ["Tyler Bates is an American composer.", "He scored many films."]],
    ["Doctor Strange", ["Doctor Strange is a 2016 superhero film."]],
]


def test_normalize_answer():
    assert normalize_answer("The Dog.") == "dog"
    assert normalize_answer("  a  CAT,  ") == "cat"
    assert normalize_answer("Yes") == "yes"


def test_exact_match():
    assert exact_match_score("the Nerdist Podcast", "Nerdist Podcast")
    assert not exact_match_score("Nerdist", "Nerdist Podcast")


def test_f1_partial_credit():
    f1, p, r = f1_score("Nerdist Podcast show", "Nerdist Podcast")
    assert 0 < f1 < 1
    assert p == 2 / 3 and r == 1.0


def test_f1_yes_no_short_circuit():
    # A span-free answer must score zero against a span, not partial credit.
    assert f1_score("yes", "Scott Derrickson")[0] == 0.0
    assert f1_score("Scott Derrickson", "yes")[0] == 0.0
    assert f1_score("yes", "yes")[0] == 1.0
    assert f1_score("no", "yes")[0] == 0.0


def test_f1_no_overlap():
    assert f1_score("completely different", "Scott Derrickson")[0] == 0.0


def test_parse_bracket_form():
    t, n, a, err = _parse("Thought: I should look.\nAction: search[Scott Derrickson]")
    assert not err and n == "search" and a == "Scott Derrickson"
    assert t == "I should look."


def test_parse_tolerant_forms():
    # The loop must not be penalised for cosmetic format drift.
    for line, name, arg in [
        ("Action: lookup(director)", "lookup", "director"),
        ("Action: finish: Ed Wood", "finish", "Ed Wood"),
        ("Action: SEARCH[Tyler Bates]", "search", "Tyler Bates"),
    ]:
        _t, n, a, err = _parse(f"Thought: x\n{line}")
        assert not err, line
        assert (n, a) == (name, arg), line


def test_parse_malformed_is_flagged():
    _t, n, _a, err = _parse("Thought: hmm.\nAction: go and look it up")
    assert err and not n
    _t2, n2, _a2, err2 = _parse("I have no idea what the format is")
    assert err2 and not n2


def test_parse_finish_with_brackets_in_answer():
    _t, n, a, err = _parse("Thought: done\nAction: finish[The Matrix (1999)]")
    assert not err and n == "finish" and a == "The Matrix (1999)"


def test_workflow_query_parsing():
    q1, q2 = _parse_two_queries("1. Who directed X?\n2. Where was they born?", "fb")
    assert q1 == "Who directed X?" and q2 == "Where was they born?"
    # Degenerate output must fall back rather than crash.
    assert _parse_two_queries("", "fb") == ("fb", "fb")
    assert _parse_two_queries("only one line", "fb") == ("only one line", "fb")


def test_retrieval_finds_the_right_paragraph():
    env = ToolEnv(context=CONTEXT, gold_titles={"Scott Derrickson"})
    tc = env.call("search", "Scott Derrickson director")
    assert tc.ok and tc.result.startswith("Scott Derrickson:")


def test_lookup_requires_a_prior_search():
    env = ToolEnv(context=CONTEXT)
    tc = env.call("lookup", "American")
    assert not tc.ok and "search" in tc.result.lower()


def test_lookup_advances_cursor():
    env = ToolEnv(context=CONTEXT)
    env.call("search", "Ed Wood filmmaker")
    a = env.call("lookup", "He")
    assert a.ok and "Result 1/1" in a.result


def test_unknown_tool_is_an_error_not_a_crash():
    env = ToolEnv(context=CONTEXT)
    tc = env.call("browse", "http://example.com")
    assert not tc.ok and "unknown tool" in tc.result


def test_poisoned_first_search_returns_a_distractor():
    env = ToolEnv(context=CONTEXT, gold_titles={"Scott Derrickson"},
                  poison_first_search=True)
    first = env.call("search", "Scott Derrickson director")
    assert first.ok
    assert not first.result.startswith("Scott Derrickson:")
    assert first.injected_fault == "poisoned_first_search"
    # Only the first search is poisoned; the loop can recover afterwards.
    second = env.call("search", "Scott Derrickson director")
    assert second.result.startswith("Scott Derrickson:")


def test_flaky_tools_are_deterministic_for_a_given_seed():
    def pattern(seed):
        env = ToolEnv(context=CONTEXT, flaky_p=0.5, seed=seed)
        return [env.call("search", "Ed Wood").ok for _ in range(8)]

    assert pattern(7) == pattern(7)          # reproducible
    assert any(not ok for ok in pattern(7))  # and actually injecting


def test_budget_raises_once_exhausted():
    b = CallBudget(max_calls=2)
    b.charge(); b.charge()
    try:
        b.charge()
    except BudgetExceeded:
        pass
    else:
        raise AssertionError("budget did not raise")
    assert b.calls_made == 2


def test_budget_unlimited_by_default():
    b = CallBudget()
    for _ in range(50):
        b.charge()
    assert b.calls_made == 50


def _fake_items():
    out = []
    for i in range(40):
        out.append({
            "_id": f"b{i}", "question": f"bridge q{i}", "answer": "x",
            "type": "bridge", "level": "hard", "supporting_facts": [], "context": [],
        })
    for i in range(40):
        out.append({
            "_id": f"c{i}", "question": f"comparison q{i}", "answer": "x",
            "type": "comparison", "level": "hard", "supporting_facts": [], "context": [],
        })
    return out


def test_sampling_is_paired_across_calls():
    items = _fake_items()
    a = [x["_id"] for x in sample(items, 10, types=("bridge",), seed=3)]
    b = [x["_id"] for x in sample(items, 10, types=("bridge",), seed=3)]
    assert a == b, "same seed must give every arm the same questions"
    c = [x["_id"] for x in sample(items, 10, types=("bridge",), seed=4)]
    assert a != c, "different seeds must give different questions"


def test_sampling_balances_types():
    got = sample(_fake_items(), 10, types=("bridge", "comparison"), seed=1)
    kinds = [g["type"] for g in got]
    assert kinds.count("bridge") == 5 and kinds.count("comparison") == 5


def test_hop_mismatch_condition_actually_mixes_types():
    assert set(CONDITIONS["hop_mismatch"].types) == {"bridge", "comparison"}
    assert CONDITIONS["budget_cap"].max_llm_calls == 3
    assert CONDITIONS["flaky_tools"].flaky_p > 0
    assert CONDITIONS["distractor_heavy"].poison_first_search


def test_end_to_end_mock_run_produces_scored_traces():
    """The whole pipeline on a hand-built item, with no network and no key."""
    from harness.arms import ARMS
    from harness.llm import MockLLM
    from harness.metrics import score_answer

    item = {
        "_id": "t1", "question": "Who directed Doctor Strange?", "answer": "Scott Derrickson",
        "type": "bridge", "level": "hard",
        "supporting_facts": [["Scott Derrickson", 0]], "context": CONTEXT,
    }
    for name, fn in ARMS.items():
        env = ToolEnv(context=CONTEXT, gold_titles={"Scott Derrickson"})
        res = fn(item, MockLLM(seed=1), env, CallBudget(), max_steps=6, seed=1)
        res.scores.update(score_answer(res.predicted_answer, item["answer"]))
        assert res.arm == name
        assert res.steps, f"{name} produced no steps"
        assert res.llm_calls > 0, f"{name} made no LLM calls"
        assert res.terminated in ("answered", "step_limit", "budget", "malformed")
        d = res.as_dict()
        assert d["total_tokens"] == d["prompt_tokens"] + d["completion_tokens"]
        assert 0.0 <= d["scores"]["f1"] <= 1.0


def test_budget_cap_is_actually_enforced_in_a_real_run():
    from harness.arms import ARMS
    from harness.llm import MockLLM

    item = {
        "_id": "t2", "question": "Who directed Doctor Strange?", "answer": "Scott Derrickson",
        "type": "bridge", "level": "hard",
        "supporting_facts": [["Scott Derrickson", 0]], "context": CONTEXT,
    }
    for name, fn in ARMS.items():
        env = ToolEnv(context=CONTEXT, gold_titles={"Scott Derrickson"})
        res = fn(item, MockLLM(seed=2), env, CallBudget(max_calls=3), max_steps=8, seed=2)
        assert res.llm_calls <= 3, f"{name} exceeded its cap with {res.llm_calls} calls"


def test_mcnemar_exact_matches_known_values():
    from harness.aggregate import mcnemar_exact
    # No discordant pairs -> no evidence either way.
    assert mcnemar_exact(0, 0) == 1.0
    # Symmetric disagreement -> p = 1.
    assert abs(mcnemar_exact(5, 5) - 1.0) < 1e-9
    # 0 vs 6 one-sided wins: 2 * 0.5^6 = 0.03125
    assert abs(mcnemar_exact(0, 6) - 0.03125) < 1e-9
    assert abs(mcnemar_exact(6, 0) - 0.03125) < 1e-9
    # Small lopsided counts are not significant.
    assert mcnemar_exact(1, 3) > 0.05
    # p is a probability.
    for b in range(8):
        for c in range(8):
            assert 0.0 <= mcnemar_exact(b, c) <= 1.0


def test_pairwise_counts_are_paired_and_correct():
    from harness.aggregate import pairwise
    pricing = {"models": {"default": {"prompt": 0.0, "completion": 0.0}}}

    def mk(arm, qid, em):
        return {
            "arm": arm, "condition": "clean", "question_id": qid, "seed": 0,
            "scores": {"em": em}, "llm_calls": 2, "model_used": "m",
            "prompt_tokens": 0, "completion_tokens": 0,
        }
    runs = [
        mk("workflow", "q1", 1.0), mk("react", "q1", 1.0),   # both
        mk("workflow", "q2", 1.0), mk("react", "q2", 0.0),   # only workflow
        mk("workflow", "q3", 0.0), mk("react", "q3", 1.0),   # only react
        mk("workflow", "q4", 0.0), mk("react", "q4", 1.0),   # only react
        mk("workflow", "q5", 0.0), mk("react", "q5", 0.0),   # neither
    ]
    out = pairwise(runs, pricing)
    assert len(out) == 1
    r = out[0]
    assert (r["arm_a"], r["arm_b"]) == ("react", "workflow")
    # react is arm_a here (alphabetical), so only_a = react-only = 2
    assert r["both_correct"] == 1 and r["neither"] == 1
    assert r["only_a"] == 2 and r["only_b"] == 1
    assert r["n"] == 5
    assert abs(r["disagreement_rate"] - 0.6) < 1e-9
    assert r["net_b_minus_a"] == -1


def test_pairwise_ignores_unpaired_questions():
    from harness.aggregate import pairwise
    pricing = {"models": {"default": {"prompt": 0.0, "completion": 0.0}}}

    def mk(arm, qid, em):
        return {
            "arm": arm, "condition": "clean", "question_id": qid, "seed": 0,
            "scores": {"em": em}, "llm_calls": 1, "model_used": "m",
            "prompt_tokens": 0, "completion_tokens": 0,
        }
    # q2 only exists for one arm and must not be counted.
    runs = [mk("workflow", "q1", 1.0), mk("react", "q1", 0.0), mk("react", "q2", 1.0)]
    r = pairwise(runs, pricing)[0]
    assert r["n"] == 1


def test_by_question_type_splits_bridge_and_comparison():
    from harness.aggregate import by_question_type
    runs = [
        {"arm": "workflow", "condition": "hop_mismatch", "question_type": "bridge",
         "scores": {"em": 1.0, "f1": 1.0}, "llm_calls": 2},
        {"arm": "workflow", "condition": "hop_mismatch", "question_type": "comparison",
         "scores": {"em": 0.0, "f1": 0.0}, "llm_calls": 2},
    ]
    out = by_question_type(runs)
    assert len(out) == 2
    got = {r["question_type"]: r["em"] for r in out}
    assert got == {"bridge": 1.0, "comparison": 0.0}


def test_chain_item_is_well_formed_and_solvable():
    """A chain the retriever cannot walk would make chain_deep meaningless."""
    from harness.chain import make_chain_item
    from harness.dataset import gold_titles
    from harness.tools import ToolEnv

    for depth in (1, 3, 5):
        it = make_chain_item(depth=depth, seed=depth * 7 + 1)
        assert it["type"] == "chain"
        assert it["chain_depth"] == depth
        assert re.fullmatch(r"[A-Z]{2}-\d{4}", it["answer"]), it["answer"]
        # Titles must be unique, or retrieval is ambiguous by construction.
        titles = [t for t, _ in it["context"]]
        assert len(titles) == len(set(titles))
        # The gold code must appear exactly once in the whole context.
        blob = " ".join(" ".join(s) for _t, s in it["context"])
        assert blob.count(it["answer"]) == 1

        # Walk it with the real retriever: solvable in exactly depth+1 searches.
        env = ToolEnv(context=it["context"], gold_titles=gold_titles(it))
        cur = re.search(r"station (\w+) and", it["question"]).group(1)
        hops, found = 0, None
        for _ in range(depth + 3):
            res = env.call("search", cur).result
            hops += 1
            m = re.search(r"successor of \w+ is (\w+)", res)
            if m:
                cur = m.group(1)
                continue
            m2 = re.search(r"catalogue code of \w+ is ([A-Z]{2}-\d{4})", res)
            found = m2.group(1) if m2 else None
            break
        assert found == it["answer"], f"depth {depth} unsolvable (got {found})"
        assert hops == depth + 1, f"depth {depth} took {hops} searches"


def test_chain_depth_exceeds_what_a_two_retrieval_graph_can_reach():
    """The premise of chain_deep: depth>=3 is out of reach for 2 retrievals."""
    from harness.chain import make_items
    from harness.conditions import CONDITIONS

    deep = CONDITIONS["chain_deep"]
    assert deep.source == "chain"
    assert min(deep.chain_depths) >= 3
    items = make_items(9, deep.chain_depths, seed=0)
    assert len(items) == 9
    assert {i["chain_depth"] for i in items} == set(deep.chain_depths)
    # Required retrievals = depth+1, always more than the workflow's fixed 2.
    assert all(i["chain_depth"] + 1 > 2 for i in items)

    shallow = CONDITIONS["chain_shallow"]
    assert shallow.chain_depths == (1,)
    assert all(i["chain_depth"] + 1 <= 2
               for i in make_items(4, shallow.chain_depths, seed=0))


def test_chain_items_are_deterministic_per_seed():
    from harness.chain import make_items
    a = make_items(5, (3, 4), seed=12)
    b = make_items(5, (3, 4), seed=12)
    assert [x["answer"] for x in a] == [x["answer"] for x in b]
    assert [x["question"] for x in a] == [x["question"] for x in b]
    c = make_items(5, (3, 4), seed=13)
    assert [x["answer"] for x in a] != [x["answer"] for x in c]


def test_chain_names_are_invented_not_real_entities():
    """Real entity names would let a model answer from memory without tools."""
    from harness.chain import make_chain_item
    it = make_chain_item(depth=4, seed=3)
    titles = {t.lower() for t, _ in it["context"]}
    for common in ("obama", "paris", "google", "wood", "derrickson", "london"):
        assert common not in titles


def test_chain_answer_is_not_retrievable_without_following_the_chain():
    """Regression test for a leak that silently invalidated chain_deep.

    The gold end-paragraph once contained wording ("terminal station") that the
    question also used and the decoys lacked, so a bag-of-words retriever
    returned the answer directly. The fixed-workflow arm scored 1.000 on depth-5
    chains with 2 retrievals.

    The question necessarily contains "catalogue code", so every chain's end
    paragraph is retrievable - that is expected and fine. What must NOT happen
    is the *gold* end paragraph outranking the decoy ends, because then the
    answer is identifiable without walking the chain. With `decoy_chains`
    decoys there are decoy_chains+1 end paragraphs, so a leak-free generator
    puts the gold on top about 1/(decoy_chains+1) of the time.
    """
    from harness.chain import make_chain_item
    from harness.dataset import gold_titles
    from harness.tools import ToolEnv

    trials, top_gold = 40, 0
    for s in range(trials):
        it = make_chain_item(depth=4, seed=1000 + s)
        env = ToolEnv(context=it["context"], gold_titles=gold_titles(it))
        ranked = env._rank(it["question"])
        ends = [i for i in ranked if any("catalogue code" in x
                                         for x in it["context"][i][1])]
        assert len(ends) >= 4, "expected several candidate end paragraphs"
        best = it["context"][ends[0]][1]
        if any(it["answer"] in x for x in best):
            top_gold += 1
    # Uniform would be ~1/5 = 20%. Allow headroom for retriever noise, but a
    # systematic lexical cue would push this toward 100%.
    assert top_gold <= trials * 0.45, (
        f"gold end paragraph outranks the decoys in {top_gold}/{trials} items - "
        "the question leaks which chain is the real one"
    )


def test_chain_end_paragraphs_are_surface_identical():
    """Gold and decoy end paragraphs must differ only in names and codes."""
    import re as _re
    from harness.chain import make_chain_item

    it = make_chain_item(depth=3, seed=77)
    ends = [(t, s) for t, s in it["context"]
            if any("catalogue code" in x for x in s)]
    assert len(ends) >= 3, "expected the gold end plus decoy ends"
    shapes = set()
    for t, sents in ends:
        norm = " | ".join(
            _re.sub(r"[A-Z]{2}-\d{4}", "<CODE>", x).replace(t, "<NAME>")
            for x in sents
        )
        shapes.add(_re.sub(r"\b\d{4}s\b", "<YEAR>s", norm))
    assert len(shapes) == 1, f"end paragraphs are distinguishable: {shapes}"


# Keep this block LAST: the runner reflects over globals(), so any test defined
# below it would silently never run.
if __name__ == "__main__":
    fns = [(n, f) for n, f in sorted(globals().items())
           if n.startswith("test_") and callable(f)]
    failed = 0
    for n, f in fns:
        try:
            f()
            print(f"  PASS  {n}")
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"  FAIL  {n}: {type(e).__name__}: {e}")
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    sys.exit(1 if failed else 0)
