"""The tool environment both arms share.

Every arm gets the *same* retriever over the *same* 10 paragraphs that ship
with each HotpotQA distractor item (2 gold + 8 distractors). Nothing here
knows which arm is calling it. That is what makes the comparison fair: if
`react` beats `workflow`, it is not because it had a better search engine.

Two tools, modelled on ReAct's original Wikipedia pair:
  search[query]   -> the best-matching paragraph's title + opening sentences
  lookup[keyword] -> the next sentence in the current paragraph containing it

Fault injection lives here rather than in the arms, so a condition applies
identically to whichever architecture is under test.
"""

from __future__ import annotations

import math
import random
import re
from dataclasses import dataclass, field

_WORD_RE = re.compile(r"[a-z0-9]+")
_STOP = {
    "the", "a", "an", "of", "and", "or", "in", "on", "at", "to", "for", "is",
    "are", "was", "were", "be", "been", "by", "with", "that", "this", "it",
    "as", "from", "which", "who", "what", "when", "where", "how", "did", "do",
    "does", "has", "have", "had", "than", "then", "both",
}


def _tokens(text: str) -> list[str]:
    return [w for w in _WORD_RE.findall(text.lower()) if w not in _STOP]


@dataclass
class ToolCall:
    name: str
    argument: str
    result: str
    ok: bool
    latency_s: float = 0.0
    injected_fault: str | None = None


@dataclass
class ToolEnv:
    """Per-question tool environment.

    context: the item's [[title, [sentences...]], ...] paragraph list.
    gold_titles: titles named in supporting_facts - used only for fault
        injection and diagnostics, never exposed to a model.
    """

    context: list
    gold_titles: set = field(default_factory=set)
    flaky_p: float = 0.0
    poison_first_search: bool = False
    seed: int = 0

    calls: list = field(default_factory=list, init=False)
    _rng: random.Random = field(init=False, repr=False)
    _current_para: int | None = field(default=None, init=False)
    _lookup_cursor: int = field(default=0, init=False)
    _idf: dict = field(default_factory=dict, init=False)

    def __post_init__(self):
        self._rng = random.Random(self.seed)
        n = len(self.context) or 1
        df: dict[str, int] = {}
        for _title, sents in self.context:
            for w in set(_tokens(_title + " " + " ".join(sents))):
                df[w] = df.get(w, 0) + 1
        # Smoothed IDF so a term in all 10 paragraphs carries no signal.
        self._idf = {w: math.log((n + 1) / (c + 0.5)) for w, c in df.items()}

    # -- public API -------------------------------------------------------
    @property
    def n_calls(self) -> int:
        return len(self.calls)

    @property
    def n_failed_calls(self) -> int:
        return sum(1 for c in self.calls if not c.ok)

    def call(self, name: str, argument: str) -> ToolCall:
        name = (name or "").strip().lower()
        if name == "search":
            return self._record(self._search(argument), name, argument)
        if name == "lookup":
            return self._record(self._lookup(argument), name, argument)
        return self._record(
            (
                f"Error: unknown tool {name!r}. Available tools: "
                f"search[query], lookup[keyword].",
                False,
                None,
            ),
            name,
            argument,
        )

    # -- internals --------------------------------------------------------
    def _record(self, outcome, name, argument) -> ToolCall:
        result, ok, fault = outcome
        tc = ToolCall(name=name, argument=argument, result=result, ok=ok,
                      injected_fault=fault)
        self.calls.append(tc)
        return tc

    def _maybe_flake(self) -> str | None:
        if self.flaky_p and self._rng.random() < self.flaky_p:
            return "transient_tool_error"
        return None

    def _rank(self, query: str) -> list[int]:
        q = _tokens(query)
        if not q:
            return list(range(len(self.context)))
        scored = []
        for i, (title, sents) in enumerate(self.context):
            body = _tokens(title + " " + " ".join(sents))
            counts: dict[str, int] = {}
            for w in body:
                counts[w] = counts.get(w, 0) + 1
            score = sum(
                self._idf.get(w, 0.0) * (1 + math.log(counts[w]))
                for w in set(q) if w in counts
            )
            # Title matches are what a Wikipedia search actually keys on.
            title_toks = set(_tokens(title))
            score += 2.5 * sum(self._idf.get(w, 0.0) for w in set(q) & title_toks)
            scored.append((score, i))
        scored.sort(key=lambda t: (-t[0], t[1]))
        return [i for _s, i in scored]

    def _search(self, query: str):
        fault = self._maybe_flake()
        if fault:
            return (
                "Error: search backend temporarily unavailable (503). "
                "You may retry.",
                False,
                fault,
            )

        ranked = self._rank(query)
        if not ranked:
            return ("Error: no paragraphs available.", False, None)

        injected = None
        idx = ranked[0]
        if self.poison_first_search and not any(
            c.name == "search" and c.ok for c in self.calls
        ):
            # Force the top hit to a non-gold paragraph: the retriever looks
            # like it worked, but handed back a distractor. A fixed pipeline
            # consumes this silently; a loop can notice and re-query.
            distractors = [i for i in ranked if self.context[i][0] not in self.gold_titles]
            if distractors:
                idx = distractors[0]
                injected = "poisoned_first_search"

        self._current_para = idx
        self._lookup_cursor = 0
        title, sents = self.context[idx]
        snippet = " ".join(sents[:3]).strip()
        others = ", ".join(self.context[i][0] for i in ranked[1:4])
        return (
            f"{title}: {snippet}\n(Similar results: {others})",
            True,
            injected,
        )

    def _lookup(self, keyword: str):
        fault = self._maybe_flake()
        if fault:
            return (
                "Error: lookup backend temporarily unavailable (503). "
                "You may retry.",
                False,
                fault,
            )
        if self._current_para is None:
            return (
                "Error: no paragraph selected. Call search[query] first.",
                False,
                None,
            )
        title, sents = self.context[self._current_para]
        kw = keyword.lower().strip()
        hits = [s for s in sents if kw in s.lower()]
        if not hits:
            return (f"No mention of '{keyword}' found in {title}.", True, None)
        i = min(self._lookup_cursor, len(hits) - 1)
        self._lookup_cursor = i + 1
        return (f"(Result {i + 1}/{len(hits)}) {hits[i].strip()}", True, None)

    # -- for the workflow arm, which retrieves without a model in the loop --
    def top_paragraphs(self, query: str, k: int = 2) -> list[tuple[str, str]]:
        """Deterministic top-k retrieval used by the fixed DAG. Counts as a
        tool call so the arms' tool-call totals stay comparable."""
        fault = self._maybe_flake()
        if fault:
            self._record(
                ("Error: retrieval backend temporarily unavailable (503).", False, fault),
                "retrieve",
                query,
            )
            return []
        ranked = self._rank(query)
        if self.poison_first_search and not any(
            c.name == "retrieve" and c.ok for c in self.calls
        ):
            distractors = [i for i in ranked if self.context[i][0] not in self.gold_titles]
            if distractors:
                ranked = distractors + [i for i in ranked if i not in distractors]
        picked = ranked[:k]
        out = [(self.context[i][0], " ".join(self.context[i][1]).strip()) for i in picked]
        self._record(
            (f"Retrieved: {', '.join(t for t, _ in out)}", True, None),
            "retrieve",
            query,
        )
        return out
