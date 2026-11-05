# Agent loop vs. workflow

A paired experiment that puts a **ReAct agent loop** and a **fixed workflow (DAG)**
on the same task, with the same model, the same tools and the same questions, so
the only variable is control flow — then shows where each one wins and where each
one breaks.

Built for an engineering audience: the harness produces real traces and real
token accounting, and the dashboard replays them side by side.

---

## The question this answers

Not "which is better" — that has no answer. The useful questions are:

1. **What does adaptivity cost when you don't need it?** (condition: `clean`)
2. **What does rigidity cost when the environment misbehaves?** (`flaky_tools`, `distractor_heavy`)
3. **What happens when the task's shape doesn't match the graph's shape?** (`hop_mismatch`)
4. **What happens when you put a hard cost ceiling on each?** (`budget_cap`)
5. **How forecastable is each one?** — the variance question, which accuracy numbers hide entirely.
6. **Is there a task a fixed graph structurally cannot do?** (`chain_deep`) — the
   only question here whose answer turned out to be statistically significant.

## Quick start

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python -m harness.dataset download     # ~27MB download -> 47MB JSON
.venv/bin/python tests/test_harness.py           # 33 tests, no network, no key
.venv/bin/python -m harness.run --backend mock --n 25 --out results_mock
.venv/bin/python serve.py                        # opens the dashboard

# the chain conditions need no dataset at all (items are generated)
.venv/bin/python -m harness.run --backend mock \
    --conditions chain_shallow,chain_deep --n 15
```

That full sequence runs offline with no credentials. The mock backend is a
deterministic stub, so **the control flow, the fault injection and the
accounting are real, but the answers are not** — use it to verify the mechanics
and to demo the dashboard, not to draw conclusions.

For real numbers, see [Running against a live model](#running-against-a-live-model).

## The three arms

| Arm | Control flow | LLM calls |
|---|---|---|
| `workflow` | `decompose → retrieve(hop1) → retrieve(hop2) → synthesize` | exactly 2, always |
| `react` | `Thought → Action → Observation` until `finish[...]`, capped at `max_steps` | 1 … max_steps+1 |
| `hybrid` | `classify → {single \| comparison \| bounded loop}` | 2 … max_steps+2 |

`hybrid` exists so the comparison isn't a strawman — a router in front of a
bounded loop is what most production systems actually converge on. It is also
the arm that can lose to *both* others, because a misroute means paying for the
classification and then running the wrong plan. The dashboard scores the router
against HotpotQA's own `type` label so that cost stays visible.

## The data

**HotpotQA** dev-distractor split (Yang et al., 2018), CC BY-SA 4.0 —
<https://hotpotqa.github.io/>. 7,405 multi-hop questions.

The distractor split is the right choice here because **each question ships with
its own 10 paragraphs** (2 gold + 8 distractors). The `search` / `lookup` tools
run a local retriever over exactly those 10, which makes the environment
deterministic and offline — no Wikipedia dump, no live search index drifting
between runs.

Scoring is the official answer EM/F1 from `hotpot_evaluate_v1.py`, reimplemented
in `harness/metrics.py` (including HotpotQA's yes/no/noanswer short circuit, so a
predicted `yes` scores 0 against a span instead of earning token overlap).
Supporting-fact EM/F1 is deliberately **not** scored: only the workflow arm has a
well-defined notion of "the paragraphs I committed to", so scoring it would
compare two different things.

> The canonical CMU download host (`curtis.ml.cmu.edu`) has been unreliable for
> years and was down when this was written, so `harness/dataset.py` pulls the
> HuggingFace mirror of the same release and converts it back to the canonical
> JSON shape. The item count (7,405) matches the published dev-distractor size.

## The seven conditions

Accuracy on clean 2-hop questions teaches you almost nothing — both
architectures land in roughly the same place. The conditions *are* the
experiment. Each one is declared in `harness/conditions.py` with its hypothesis
written down **before** the run, so the results can contradict it. Two of the
seven hypotheses were falsified; one was falsified by a bug in the task itself.

The first five draw from HotpotQA; the two `chain_*` conditions generate their
own items and need no download.

| Condition | Perturbation | What it's probing |
|---|---|---|
| `clean` | none | the workflow's best case — graph shape matches task shape |
| `flaky_tools` | 30% of tool calls return a transient 503 | retry is free in a loop, absent in a DAG |
| `hop_mismatch` | comparison questions mixed in 50/50 | the fixed 2-hop bet is wrong |
| `distractor_heavy` | first retrieval forced to a distractor | retrieval *looks* fine but isn't |
| `budget_cap` | hard ceiling of 3 LLM calls | the loop's long tail gets truncated |
| `chain_shallow` | synthetic depth-1 successor chains | control for `chain_deep` |
| `chain_deep` | synthetic depth-3-to-5 chains | **the structural case** — 2 retrievals cannot traverse 4 links |

## Fairness invariants

The comparison is worthless if these break, so they are enforced in `run.py`
rather than left to convention:

- **Paired questions.** Questions are sampled once per `(condition, seed)` and
  reused across every arm — not resampled per arm.
- **Paired faults.** Each `ToolEnv` is seeded from `(question_id, seed)`, so an
  injected 503 lands on the *same question* for every arm.
- **One retriever, one prompt vocabulary.** Shared in `tools.py`; nothing there
  knows which arm is calling it.
- **Metered tokens.** Token counts come from the `usage` block the API
  returns on every completion, never from an estimate. The runner's preflight
  warns if the configured endpoint omits it.

## Measured results

`gpt-5.2-2025-12-11`, 300 runs (20 questions × 5 conditions × 3 arms),
`--repeats 1`, 58 min wall clock. Rolled up across all conditions:

| option | n | EM | F1 | LLM calls | calls p95 | calls max | tokens | tokens p95 | tail | s/call | $/correct | cost CV | clean end |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **workflow** | 100 | 0.520 | 0.663 | **2.00** | **2** | **2** | **635** | **1,144** | **1.8×** | **0.83** | **0.00014** | **0.31** | **100%** |
| **react** | 100 | 0.520 | 0.673 | 3.05 | 6 | 9 | 1,382 | 4,174 | 3.0× | 1.15 | 0.00035 | 0.81 | 94% |
| **hybrid** | 100 | **0.540** | **0.677** | 3.67 | 7 | 10 | 1,365 | 4,459 | 3.3× | 1.03 | 0.00032 | 0.93 | 88% |

`tail` = tokens p95 ÷ mean (how far the expensive runs sit from the typical one).
`cost CV` = coefficient of variation; lower is more forecastable. `s/call` =
median per-call latency. Dollar figures use placeholder rates — see below.

**How to read this:** accuracy is a three-way tie (0.520 / 0.520 / 0.540) and
**none of the 15 paired head-to-head comparisons reached significance** — every
exact-McNemar p ≥ 0.25 at n=20 per cell. The accuracy column does not
discriminate between these architectures on this benchmark — and it is also
generous: scored for retrieved evidence instead of answer strings, the loop
falls to 0.357 while the workflow holds 0.480 (see [grounded EM, and why the
accuracy tie is generous](#measured-grounded-em-and-why-the-accuracy-tie-is-generous)).
The cost columns do,
and decisively: the workflow uses **2.2× fewer tokens** for the same accuracy,
and its cost distribution is a near point mass (tail 1.8×, CV 0.31) against the
loop's long right tail (tail 3.0×, CV 0.81). That is the finding that matters
operationally — one is forecastable, the other is not.

Per-condition results, hypothesis-by-hypothesis, are in `results/summary.json`
and the dashboard. Two of the five pre-registered hypotheses were **falsified**:
`distractor_heavy` (the loop did *worse*, 0.400 vs 0.500 — adaptivity cost more
and delivered less) and `hop_mismatch` (comparison questions turned out *easier*
for every arm, including the fixed graph). `budget_cap` confirmed the hybrid
risk: it fell to 0.300, the worst cell in the matrix, because the router
consumes one of only three permitted calls.

**The biggest lever is not architectural.** 39% of all wrong answers (55 of 142)
score F1 ≥ 0.5 — the right entity with a mis-scoped span (`TD Bank` vs
`TD Bank, N.A.`, `1,467` vs `1,467 rooms`). That is a prompt problem shared by
all three arms, and fixing the answer-span instruction would move more than any
architecture choice here.

## The forced case: variable-depth chains

HotpotQA is 2-hop by construction - the exact shape `workflow` hardcodes - so it
cannot test whether an agent loop can do something a fixed graph *structurally
cannot*. `harness/chain.py` builds that case: successor chains of unknown depth,
where resolving depth *d* needs *d+1* retrievals and *d* is invisible from the
question. Entity names are invented so no arm can shortcut via memorised
knowledge, and 4 decoy chains make the end paragraphs indistinguishable, so blind
guessing floors at ~20%.

15 questions per condition, live, same model:

| arm | chain_shallow (depth 1) | chain_deep (depth 3-5) | calls (deep) | tokens (deep) |
|---|---|---|---|---|
| **workflow** | 0.467 | **0.133** | 2.00 | 416 |
| **react** | 0.933 | **1.000** | 6.53 | 3,528 |
| **hybrid** | 0.933 | **1.000** | 7.60 | 3,704 |

This is the **only place in the whole experiment where accuracy differences are
statistically significant**: on `chain_deep` both loops beat the workflow 13-0 on
paired questions (exact McNemar **p = 0.0002**, 87% disagreement). The workflow's
0.133 is at the ~20% guessing floor - it retrieves end paragraphs and picks one.
No prompt or model upgrade fixes that; 2 retrievals cannot traverse 4 links.

**Caveat on how this was reached:** the first version of this task was broken. The
gold end-paragraph contained wording ("terminal station") that the question also
used and the decoys lacked, so the retriever fetched the answer directly and the
workflow scored 1.000 on depth-5 chains using 360 tokens. The fix was to make
gold and decoy end paragraphs surface-identical;
`test_chain_answer_is_not_retrievable_without_following_the_chain` now guards it
by asserting the gold end does not outrank the decoy ends.

## Known limitation: the workflow arm cannot chain its retrievals

`workflow.py` emits **both** retrieval queries from a single `decompose` call, so
the second query is written before the first result is seen. There is no data
dependency between the two retrieval nodes. This is a fixed DAG in its purest
form, but it is **weaker than a workflow a competent engineer would ship** - the
obvious improvement is `retrieve(hop1) -> extract bridge entity -> retrieve(hop2)`,
which costs one more LLM call and makes genuine 2-hop reasoning possible.

This qualifies every result here. The workflow arm's 0.520 EM on HotpotQA is not
evidence of bridging. Where it is grounded at all, it comes from both gold
paragraphs happening to land in one flat retrieval on lexical overlap; the rest
of the time the model simply knew the answer (see the next section). Its 0.467
on depth-1 chains is the same limitation showing up directly: even one link
requires reading hop 1 before querying hop 2. **A sequential-retrieval workflow
is the single most valuable fourth arm to add**, and until it exists, read
"workflow" here as "cheapest possible fixed pipeline", not "best fixed pipeline".

## Measured: grounded EM, and why the accuracy tie is generous

EM scores the answer string, not the evidence. On a benchmark built from
Wikipedia that lets the model's prior count as a win: an arm can retrieve
nothing useful, answer from memory, and score exactly like an arm that read the
sentence. `harness/grounding.py` scores the stricter thing — **grounded EM**:
exact-match correct *and* the gold string present in the paragraph text that arm
actually put in front of its own model. A title glimpsed in a
`(Similar results: …)` trailer is a pointer, not evidence, and does not count.

```bash
python -m harness.dataset download       # paragraph bodies are not in the trace
python -m harness.grounding              # the table below
python -m harness.grounding --by-condition --json
```

Over all 300 runs — 98 of the 100 paired cells, two comparison questions having
yes/no gold answers that no paragraph can contain, dropped for every arm so the
pairing survives:

| arm | EM | grounded EM | correct | of those, unevidenced |
|---|---|---|---|---|
| `workflow` | 0.510 | **0.480** | 50 | 3 |
| `react` | 0.510 | **0.357** | 50 | 15 |
| `hybrid` | 0.531 | **0.449** | 52 | 8 |

The three-way EM tie is real and it does not survive the second column. All
three arms answer about the same number of questions correctly; the loop reaches
a quarter fewer of them through its own retrieval. 15 of its 50 correct answers
came from a trailer title or from nothing retrieved at all, against 3 of the
workflow's.

**The agent loop is the arm most exposed**, which is the opposite of the
intuitive reading. It chooses when to stop — its prompt says "call `finish` as
soon as you can answer; do not keep searching to confirm an answer you already
have" — and a model that already knows the answer satisfies that after one
search that returned nothing useful. The fixed DAG has no early exit: it runs
both retrievals every time and puts four paragraphs in front of `synthesize`,
so when it is right it is usually right for the recorded reason. Part of this
number is therefore a prompt choice wearing an architecture's clothes, and it
is the one result here where the two are hard to separate.

**What this is not.** Not significant per condition: on grounded correctness
every per-condition exact-McNemar p is ≥ 0.22, the same verdict the plain-EM
tests returned. Pooled over all five conditions the workflow is grounded-correct
where the loop is not 21 times against 9 (p = 0.043), but those 100 cells reuse
only 30 distinct questions across five perturbations, so they are not
independent and that p-value is not a licence. Take the direction — the workflow
leads in four of the five conditions and ties in the fifth — and take the size
of the gap between the two columns as the reason to distrust an accuracy tie on
a benchmark made of facts a model may already know.

### Two worked pairs from `results/runs.json`

Both are `clean`, both quoted unedited, and in both the winning answer was read
rather than recalled.

**Both arms grounded; the DAG cheaper.** `5adf75e35542993344016c98`, *"What
American horror film directed by Sidney J. Furie was based off an alleged
haunting which occurred in 1974 at Culver City, California?"*, gold
`The Entity`. The question's own wording retrieves the gold paragraph, so there
is no second hop to take. `workflow` ran both retrievals, got both gold
paragraphs including the sentence naming Furie, and answered in 483 tokens.
`react` searched once, matched the haunting to a film of the same name, and
finished in 697 tokens without ever reading the director sentence. Cheaper
*and* better evidenced — most of the distractor split looks like this, which is
why the workflow's EM is competitive at all.

**Only the loop grounded.** `5ae2d8fb55429928c42394fa`, *"Who was the base where
the 413th Flight Test Group is stationed named after?"*, gold
`Brig Gen Augustine Warner Robins`. Here hop 2 is a function of hop 1's result.
`decompose` had to write it blind and produced *"What is the namesake of Edwards
Air Force Base"* — a base supplied by the model's prior — so the retrieval
returned Edwards paragraphs (one of which really does name the 413th, so the
wrong trail reads as on-topic), the string `Augustine` never entered the
context, and `synthesize` answered `Robins Air Force Base`: the bridge entity
itself, EM 0, F1 0.22. `react` searched the unit, read "stationed at Robins Air
Force Base", searched that, read "named in honor of Brig Gen Augustine Warner
Robins", and finished — 1,162 tokens against 594 for the one edge the DAG does
not have. Note which arm the model's knowledge hurt: prior knowledge does not
fail neutrally across the two architectures, which is why a comparison run on
facts the model may know is not a comparison of control flow.

A third pair, where EM scored both arms correct and **neither** was grounded,
is the reason this section exists: `5a8756f25542994775f607dc`, gold
`7.63×25mm Mauser`. The workflow's hop 2 went after the wrong pistol and it
answered correctly anyway, out of the model's knowledge of the C96 — its prompt
ends *"if the paragraphs do not contain the answer, reply with your single best
guess anyway"*. `react` saw `7.63×25mm Mauser` ranked first in a trailer — the
pointer its architecture exists to follow — and called `finish` without opening
it.

Read the HotpotQA accuracies, then, as **upper bounds on retrieval-grounded
answering** rather than as multi-hop reasoning. The variable-depth chains above
use invented entity names precisely so memorisation cannot reach them; that is
the arm comparison to trust.

## Reading latency: a measured caveat

**p95 latency in these results is gateway tail, not architecture.** Measured over
872 live calls: per-call p50 1.05s, p90 1.41s, but **p99 48.0s** — 6.7% of calls
hit a ~48s cliff, and it hits every arm equally (the `workflow` arm shows a
per-call p95 of 47.3s while making exactly 2 calls per question).

So don't read `latency_p95` as an architecture result. The architecture-attributable
part of latency is the **number of calls**, since per-call p50 is ~1s for all
three arms. Compare `llm_calls_mean` and reason from there, or re-run when the
gateway is quieter.

## What to read in the results

The headline metric is **cost per correct answer**, not accuracy — an arm that
buys 4 points of EM for 3× the tokens is a different decision at 100
questions/day than at 10M.

Then look at **`cost_cv`** (coefficient of variation) and the **distribution of
LLM calls**. This is the finding that matters operationally and that an accuracy
table cannot show you: a workflow's cost distribution is nearly a point mass, so
you can put it in an SLA and a capacity plan. A loop's has a long right tail,
because the model decides when to stop. Two systems with identical mean cost can
be very different things to run.

Also worth reading: **how runs ended**. The failure modes aren't symmetric.
A loop can fail to terminate or emit a malformed action; a DAG structurally
cannot. A DAG's characteristic failure is the quiet one — answering confidently
from a distractor it had no way to notice.

**Do not read accuracy differences as results without the paired test.** The
`pairwise` table reports, per arm pair, how many questions each arm got right
that the other got wrong, plus an exact McNemar p-value on those discordant
pairs. Across the five HotpotQA conditions every comparison came back p ≥ 0.25 —
the EM gaps there are noise at n=20. Only `chain_deep` reached significance
(13-0, p = 0.0002). The point of pairing the sampling is to make this test
available; use it.

## Running against a live model

Authentication is the standard OpenAI scheme: a secret key in `OPENAI_API_KEY`.

```bash
export OPENAI_API_KEY=sk-...
.venv/bin/python -m harness.run --backend openai --n 50 --repeats 3 --verbose
```

Or keep it in a file — `.env` is gitignored and must never be committed:

```bash
cp .env.example .env     # then paste your key into .env
.venv/bin/python -m harness.run --backend openai --n 50 --repeats 3 --verbose

# or point at a key file you already keep elsewhere, rather than copying it here
.venv/bin/python -m harness.run --backend openai --env-file ~/.config/openai/.env
```

Search order is `--env-dir` → `--env-file` → `$AGENTLOOP_ENV_FILE` → `./.env`;
already-exported variables win. Values are pushed into the process environment
by python-dotenv and are never read or logged by the harness — only the
resolved path is printed. Reusing a key file you already have beats copying
one: a secret in two places is a second thing to rotate and a second thing to
leak.

### Any OpenAI-compatible endpoint

Set `OPENAI_BASE_URL` and the same code path serves Azure OpenAI, vLLM,
LiteLLM, OpenRouter, Ollama or a corporate gateway:

```bash
export OPENAI_BASE_URL=http://localhost:11434/v1   # Ollama
export OPENAI_MODEL=llama3.1
```

One requirement: **the endpoint must return a `usage` block** on each
completion. Per-call token accounting is the entire point of this comparison,
and an endpoint that omits it silently zeroes every cost column. A
`--backend openai` run does a **preflight completion before the sweep starts**,
which both fails fast on a bad key and warns you if `usage` is missing — better
than recording the same error on all 2,250 questions.

### Model capability quirks

Some models **reject `stop`** (`400 unsupported_parameter`), and some pin
`temperature`. Since only the `react` arm sends a stop sequence, one such 400
would zero the agent arm while the workflow arm ran fine — a model-capability
detail masquerading as an architecture result. This happened during
development. `llm.py` parses the rejected parameter name out of the error,
drops it, and retries, remembering it for the rest of the run. **If you add a
parameter, make sure a rejection degrades rather than fails.**

Dropping `stop` has a consequence the loop has to handle: without it the model
can run on and write its own `Observation:` line. `react.py::_truncate_at_observation`
cuts generations there, so the loop can never hallucinate evidence and "answer"
without calling a tool.

**Use `--repeats 3` or more for any result you intend to show someone.** Variance
between identical runs is one of the main things being measured; a single pass
cannot distinguish a real difference from sampling noise.

Rough cost: a sweep is `arms × conditions × n × repeats` questions, each costing
2–9 LLM calls. The default sweep at `--n 50 --repeats 3` is 2,250 questions and
roughly 9k LLM calls. Start with `--n 10 --repeats 1 --conditions clean` to
sanity-check connectivity before committing to a full sweep.

### Corporate TLS

Python's `ssl` module doesn't read the macOS keychain, so behind a
TLS-inspecting proxy the dataset download fails with `self-signed certificate
in certificate chain` even though the URL works in a browser. `harness/dataset.py` handles this by
exporting the system trust store to `data/ca-bundle.pem` on first use
(`python -m harness.dataset bootstrap-ca` to do it explicitly). Public CA certs
only — no keys. Override with `HOTPOT_CA_BUNDLE`, or point the LLM client at a
bundle with `OPENAI_CA_BUNDLE`.

## Cost figures

`pricing.json` ships **placeholder** per-token rates. Token counts in the
results are metered and real;
every dollar figure is derived from that file. Replace the rates with your
chargeback numbers before quoting cost to anyone, or read the token columns
instead. The dashboard shows a banner whenever the rates are still placeholders.

## Layout

```
harness/
  llm.py          OpenAI-compatible client + deterministic MockLLM + CallBudget
  tools.py        shared retriever (search/lookup/retrieve) + fault injection
  metrics.py      official HotpotQA answer EM/F1
  conditions.py   the seven conditions, each with its hypothesis
  dataset.py      download, canonical-shape conversion, stratified sampling
  chain.py        synthetic variable-depth chain tasks (no download needed)
  trace.py        the one trace schema every arm emits
  aggregate.py    per-arm roll-up, per-cell stats, paired McNemar, p50/p95, CV
  run.py          the runner and the fairness invariants
  arms/
    workflow.py   fixed DAG
    react.py      Thought/Action/Observation loop
    hybrid.py     router + bounded loop
viewer/index.html dashboard: per-arm table, charts, head-to-head, trace explorer
tests/            33 tests, offline
results/          committed runs.json + summary.json (the numbers above)
results_chain/    the same, for the two chain conditions
```

## Extending it

- **A new arm** — add a module to `harness/arms/` exporting
  `run(item, llm, env, budget, max_steps, seed) -> RunResult` and register it in
  `arms/__init__.py`. Emit `Step`s and the dashboard picks it up. (The viewer's
  categorical palette seats three series safely; a fourth needs a re-validated
  palette — see the note below.)
- **A sequential-retrieval workflow** — the highest-value addition, for the
  reason in [Known limitation](#known-limitation-the-workflow-arm-cannot-chain-its-retrievals).
  `retrieve(hop1) -> extract bridge entity -> retrieve(hop2)`: one extra LLM
  call, genuine 2-hop capability, still fully deterministic.
- **A new condition** — add a `Condition` to `harness/conditions.py`. Set
  `source="chain"` to generate items instead of sampling HotpotQA. Write the
  hypothesis down before you run it — and if a result comes back flattering,
  check the task for leaks before believing it (see `chain.py`).
- **Native tool-calling** — `react.py` parses actions out of text, matching the
  original ReAct formulation and staying portable across gateway routes that
  don't expose a `tools` parameter. Swapping in native function-calling would
  make a good fourth arm: it isolates how much of the loop's malformed-action
  rate is the format's fault rather than the architecture's.

The dashboard's palette was validated for colour-vision deficiency and contrast
(3 categorical slots, light and dark, all-pairs). Light-mode slot 3 sits below
3:1 against the surface, which is why bars carry direct value labels and a table
view ships — don't remove either without re-validating.

## Secrets

The only credential this project needs is an OpenAI-format API key, read from
`OPENAI_API_KEY` in the environment. It is never written to a trace, a result
file, a log line or an error message — the harness prints the *path* of a
credentials file it loaded, never a value, and reports only the *absence* of a
key. `.env` and `*.pem` are gitignored; `.env.example` holds placeholder names
only.

No personal, customer or health data is involved anywhere in this repository.
The HotpotQA questions are public Wikipedia-derived text (CC BY-SA 4.0) and the
chain tasks are synthetic, with invented entity names.

If you fork this and point it at a private gateway, keep the key out of
`results/*.json` before you push — those files are committed here.

## License

MIT — see [LICENSE](LICENSE). The bundled HotpotQA *results* derive from
HotpotQA (Yang et al., 2018), which is CC BY-SA 4.0.
