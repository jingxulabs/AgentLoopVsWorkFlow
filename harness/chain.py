"""Synthetic variable-depth chain tasks - the case a fixed graph cannot solve.

Every HotpotQA question is 2-hop by construction, which is exactly the shape the
`workflow` arm hardcodes. That makes HotpotQA unable to test the *structural*
claim about agent loops: that they can handle a task whose required number of
steps is unknown until run time and varies per instance.

This module builds that task. A chain item looks like:

    Station Varnolith  -> successor Kesmirra
    Station Kesmirra   -> successor Thalquin
    Station Thalquin   -> successor Orrendel
    Station Orrendel   -> catalogue code QX-7731      <- the answer

asked as "starting from Varnolith and following each successor in turn, what is
the catalogue code of the final station?". Resolving a depth-d chain needs d+1
retrievals, and d is not visible from the question.

Two design choices that matter for validity:

  * **Invented entity names.** A model cannot answer a chain about Varnolith from
    parametric knowledge, so it is forced to actually use the tools. With real
    entities the whole comparison would leak - an arm could score well without
    retrieving anything.
  * **Decoy chains.** Each item also contains paragraphs from unrelated chains,
    so retrieval returns plausible-looking successors and the task is not
    solvable by grabbing every paragraph at once.

This condition is deliberately adversarial to the fixed workflow. That is the
point - it marks the boundary where the architecture choice is *forced* rather
than optimised. It is not evidence that loops are better in general, and the
`chain_shallow` control (depth 1, which the workflow can reach) exists to keep
that honest.
"""

from __future__ import annotations

import random

# Invented, pronounceable, and unlikely to collide with real article titles.
_SYLL_A = ["Var", "Kes", "Thal", "Orr", "Zep", "Mir", "Quen", "Dra", "Sol", "Yth",
           "Cal", "Nor", "Pell", "Ryn", "Vex", "Ulm", "Bran", "Ith", "Oga", "Tam"]
_SYLL_B = ["no", "mi", "qui", "ren", "ha", "lo", "ve", "su", "ta", "ze",
           "bri", "da", "fo", "gal", "ker", "lum"]
_SYLL_C = ["lith", "rra", "n", "del", "sk", "mir", "th", "ra", "x", "nor",
           "vik", "wen", "dor", "ys"]


def _name(rng: random.Random) -> str:
    return (rng.choice(_SYLL_A) + rng.choice(_SYLL_B) + rng.choice(_SYLL_C)).capitalize()


def _unique_names(rng: random.Random, k: int) -> list[str]:
    seen: set[str] = set()
    while len(seen) < k:
        n = _name(rng)
        # Keep names distinct enough that the bag-of-words retriever can tell
        # them apart - a near-duplicate would make this a retrieval test.
        if n not in seen and all(not (n.startswith(s[:4]) or s.startswith(n[:4]))
                                 for s in seen):
            seen.add(n)
    return list(seen)


def _code(rng: random.Random) -> str:
    return (f"{rng.choice('QXZRVK')}{rng.choice('XJLMPT')}-"
            f"{rng.randint(1000, 9999)}")


def make_chain_item(depth: int, seed: int, decoy_chains: int = 4) -> dict:
    """One chain item in the canonical HotpotQA shape.

    `depth` is the number of successor links; resolving it needs depth+1
    retrievals. Returned dict is drop-in compatible with every arm, the shared
    retriever and the official metrics - nothing else in the harness needs to
    know this item is synthetic.
    """
    rng = random.Random(seed)
    total = (depth + 1) * (1 + decoy_chains) + 2
    names = _unique_names(rng, total)

    main = names[: depth + 1]
    answer = _code(rng)

    context: list = []
    supporting: list = []

    # The real chain: each station names its successor; the last holds the code.
    for i, nm in enumerate(main):
        if i < depth:
            sents = [
                f"{nm} is an orbital research station in the Tessaly register.",
                f"The designated successor of {nm} is {main[i + 1]}.",
                f"{nm} was commissioned in the {1900 + (seed + i) % 120}s.",
            ]
        else:
            # CRITICAL: this paragraph must be surface-identical to every decoy
            # chain's end paragraph. An earlier version added "is the terminal
            # station of its register line" here, which - combined with the
            # question asking for the "terminal station's catalogue code" - let
            # the bag-of-words retriever fetch the answer directly on the words
            # 'terminal'/'catalogue'/'code'. The fixed-workflow arm then scored
            # 1.000 on depth-5 chains with 2 retrievals, which silently
            # invalidated the whole condition. Do not reintroduce any wording
            # here that the question also contains and the decoys lack.
            sents = [
                f"{nm} is an orbital research station in the Tessaly register.",
                f"The catalogue code of {nm} is {answer}.",
                f"{nm} was commissioned in the {1900 + (seed + i) % 120}s.",
            ]
        context.append([nm, sents])
        supporting.append([nm, 1])

    # Decoy chains: same surface form, wrong line. These make "retrieve
    # everything at once" fail and give a careless loop something to follow.
    cursor = depth + 1
    for _d in range(decoy_chains):
        seg = names[cursor: cursor + depth + 1]
        cursor += depth + 1
        for i, nm in enumerate(seg):
            if i < len(seg) - 1:
                sents = [
                    f"{nm} is an orbital research station in the Tessaly register.",
                    f"The designated successor of {nm} is {seg[i + 1]}.",
                    f"{nm} was commissioned in the {1900 + (seed + i) % 120}s.",
                ]
            else:
                # Same three-sentence shape as the gold end paragraph, so no
                # lexical cue distinguishes the real answer from a decoy code.
                sents = [
                    f"{nm} is an orbital research station in the Tessaly register.",
                    f"The catalogue code of {nm} is {_code(rng)}.",
                    f"{nm} was commissioned in the {1900 + (seed + i) % 120}s.",
                ]
            context.append([nm, sents])

    for nm in names[cursor:]:
        context.append([nm, [
            f"{nm} is a decommissioned relay buoy, not part of any register line.",
        ]])

    rng.shuffle(context)

    question = (
        f"Starting from the station {main[0]} and following each designated "
        f"successor in turn until you reach the terminal station, what is that "
        f"terminal station's catalogue code?"
    )
    return {
        "_id": f"chain-d{depth}-s{seed}",
        "question": question,
        "answer": answer,
        "type": "chain",
        "level": f"depth{depth}",
        "supporting_facts": supporting,
        "context": context,
        "chain_depth": depth,
    }


def make_items(n: int, depths: tuple, seed: int = 0) -> list[dict]:
    """`n` chain items with depths cycled from `depths`, deterministically."""
    rng = random.Random(seed)
    out = []
    for i in range(n):
        d = depths[i % len(depths)]
        out.append(make_chain_item(d, seed=rng.randrange(1 << 30)))
    return out
