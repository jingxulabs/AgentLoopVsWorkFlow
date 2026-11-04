"""The five conditions.

Accuracy alone will not show you the trade-off - on clean 2-hop questions the
two architectures land in roughly the same place and you learn nothing. The
conditions are the experiment: each one perturbs the environment in a way that
a fixed graph and an adaptive loop respond to differently.

Each condition fixes (a) which questions are sampled, (b) how the tools
misbehave, and (c) what budget the arm gets. Conditions apply identically to
every arm.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Condition:
    name: str
    description: str
    hypothesis: str
    # question source: "hotpot" (the real dataset) or "chain" (synthetic
    # variable-depth chains from chain.py)
    source: str = "hotpot"
    chain_depths: tuple = (1,)
    # question sampling
    types: tuple = ("bridge",)      # HotpotQA `type` values to include
    levels: tuple = ()               # empty = any level
    # tool environment
    flaky_p: float = 0.0
    poison_first_search: bool = False
    # budget
    max_llm_calls: int | None = None
    max_steps: int = 8


CONDITIONS: dict[str, Condition] = {
    "clean": Condition(
        name="clean",
        description="2-hop bridge questions, reliable tools, no budget cap.",
        hypothesis=(
            "The workflow's fixed shape matches the task exactly, so it should "
            "match the loop on accuracy at materially lower cost and latency. "
            "This is the condition where a workflow is the right answer."
        ),
        types=("bridge",),
    ),
    "flaky_tools": Condition(
        name="flaky_tools",
        description="30% of tool calls return a transient 503.",
        hypothesis=(
            "A retry is already in the loop's vocabulary - a failed call is "
            "just another observation. The DAG has no retry edge, so a failed "
            "hop becomes a permanent hole in the synthesis context. Expect the "
            "accuracy gap to open up here."
        ),
        types=("bridge",),
        flaky_p=0.30,
    ),
    "hop_mismatch": Condition(
        name="hop_mismatch",
        description="Comparison questions mixed in with bridge questions.",
        hypothesis=(
            "Comparison questions need two independent lookups, not a bridge. "
            "The DAG's hop-2 query is conditioned on hop-1 output that is "
            "irrelevant here, so it should degrade. The loop discovers the "
            "shape; the router should recover most of the gap at the cost of "
            "one extra call."
        ),
        types=("bridge", "comparison"),
    ),
    "distractor_heavy": Condition(
        name="distractor_heavy",
        description="The first retrieval is forced to return a distractor.",
        hypothesis=(
            "Retrieval looks like it succeeded but returned the wrong "
            "paragraph. The workflow cannot tell the difference and answers "
            "confidently from a distractor. The loop can read the observation, "
            "notice the mismatch, and re-query - if the model is good enough "
            "to notice."
        ),
        types=("bridge",),
        poison_first_search=True,
    ),
    "chain_shallow": Condition(
        name="chain_shallow",
        description="Synthetic successor chains of depth 1 (2 retrievals needed).",
        hypothesis=(
            "The control for chain_deep. Depth 1 is exactly what the fixed graph "
            "provides, so all three arms should solve these. If the workflow "
            "fails here too, the chain task is simply hard and chain_deep proves "
            "nothing about depth - this condition is what makes that distinction."
        ),
        source="chain",
        chain_depths=(1,),
        max_steps=8,
    ),
    "chain_deep": Condition(
        name="chain_deep",
        description="Synthetic successor chains of depth 3-5 (4-6 retrievals needed).",
        hypothesis=(
            "The structural case. The workflow performs exactly 2 retrievals, so "
            "a depth-4 chain is unreachable for it no matter how good the model "
            "or the prompt is - it should score ~0. The loop discovers the depth "
            "by following links and should solve most. Hybrid is capped at 4 loop "
            "steps, so it should handle depth 3 and start failing by depth 5. "
            "This is the one condition where the architecture choice is forced "
            "rather than optimised."
        ),
        source="chain",
        chain_depths=(3, 4, 5),
        max_steps=10,
    ),
    "budget_cap": Condition(
        name="budget_cap",
        description="Hard ceiling of 3 LLM calls per question.",
        hypothesis=(
            "The workflow needs exactly 2 and fits comfortably. The loop's "
            "step count is a random variable, so a cap truncates its long "
            "tail and forces answers from incomplete notes. This is the cost "
            "ceiling problem stated as an experiment."
        ),
        types=("bridge",),
        max_llm_calls=3,
        max_steps=8,
    ),
}


def get(name: str) -> Condition:
    try:
        return CONDITIONS[name]
    except KeyError:
        raise SystemExit(
            f"unknown condition {name!r}. Available: {', '.join(CONDITIONS)}"
        ) from None
