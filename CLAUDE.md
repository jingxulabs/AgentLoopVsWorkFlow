# CLAUDE.md — AgentLoopVsWorkFlow

A paired experiment comparing a **ReAct agent loop**, a **fixed workflow (DAG)**,
and a **router+bounded-loop hybrid** on the same tasks, plus a dashboard that
replays the traces. Read `README.md` for results and rationale; this file is for
working on the code.

## Commands

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python -m harness.dataset download    # HotpotQA dev-distractor
.venv/bin/python tests/test_harness.py          # 33 tests, offline, no key
.venv/bin/python -m harness.run --backend mock --n 25
.venv/bin/python serve.py                       # dashboard at :8077
.venv/bin/python -m harness.grounding --by-condition   # grounded EM (needs the corpus)

# live: standard OpenAI key auth
export OPENAI_API_KEY=sk-...
.venv/bin/python -m harness.run --backend openai --n 50 --repeats 3 --verbose

# or point at a key file you already keep elsewhere, rather than copying it here
.venv/bin/python -m harness.run --backend openai --env-file ~/.config/openai/.env
```

There is no pytest dependency; `tests/test_harness.py` runs standalone.

## Non-negotiables

**Fairness invariants.** The whole comparison is void if these break; they live
in `run.py` and are enforced, not conventional:
- Questions are sampled once per `(condition, seed)` and **reused across arms** —
  the comparison is paired, which is what makes the McNemar test valid.
- Each `ToolEnv` is seeded from `(question_id, seed)`, so an injected fault lands
  on the *same question* for every arm.
- One retriever and one prompt vocabulary, shared in `tools.py`. Nothing there
  knows which arm is calling it. **Only control flow may differ between arms.**

**Never let a model-capability detail read as an architecture result.** This has
already bitten once: a model rejected `stop`, and since only `react` sent one,
that single 400 zeroed the agent arm while the workflow arm scored fine.
`llm.py` parses the rejected parameter from the error, drops it, and retries. If
you add a parameter, make sure a rejection degrades rather than fails.

**Token accounting requires a `usage` block on every completion.** Any
OpenAI-compatible endpoint works via `OPENAI_BASE_URL`, but one that omits
`usage` silently zeroes every cost column — which is the main metric here. The
`--backend openai` preflight warns when that happens; do not ignore it.

**Write a condition's hypothesis before running it** (`conditions.py` has a
`hypothesis` field). Two of seven were falsified — that is the format working.

**Distrust flattering results.** `chain_deep` first "proved" a 2-retrieval graph
could solve depth-5 chains; the task was leaking the answer through shared
wording between the question and the gold paragraph. Check for leaks before
believing a result that contradicts a structural argument. The guard is
`test_chain_answer_is_not_retrievable_without_following_the_chain`. The same
instinct applies to a *correct* HotpotQA answer — check it was retrieved, not
recalled; see the known limitations below.

## Known limitations (read before interpreting any HotpotQA number)

**The fixed pipeline cannot bridge.** `arms/workflow.py` emits **both** retrieval
queries from one `decompose` call, so hop 2's query cannot depend on hop 1's
result. It is the cheapest possible fixed pipeline, not a good one. Adding
`retrieve → extract → retrieve` as a fourth arm is the highest-value open task.

**EM overstates grounded answering, and the loop is the arm most exposed.** A
correct answer here does not mean the evidence was retrieved. `harness/grounding.py`
scores the stricter thing — correct *and* the gold string present in the paragraph
text that arm actually had — and over all 300 runs the EM tie does not survive it:
grounded EM is 0.480 (workflow), 0.357 (react), 0.449 (hybrid), with 3, 15 and 8 of
each arm's correct answers unevidenced. On `clean` alone the counts are 9/10, 6/10
and 9/11. The agent arm is worse than the DAG because it chooses when to stop
(`_SYS`: "call finish as soon as you can answer; do not keep searching to confirm"),
while the DAG always puts four paragraphs in front of `synthesize` — so part of this
number is a prompt choice, not an architecture.

**Count the hybrid's `single` route when scoring grounding.** It calls
`top_paragraphs` and hands whole paragraphs to `answer_single`, exactly as the
workflow does, but the trace records only titles. Reading `search` observations
alone misses those runs and undercounts the arm — that is how the hybrid figure was
once reported as 5/11 instead of 9/11. `grounding.py` reads titles back out of the
corpus for any `kind == "tool"` retrieve step, which is also why it needs the
dataset downloaded.

Three worked cases, all `clean`: `5ae2d8fb55429928c42394fa` (only the loop grounded
— `decompose` wrote hop 2 blind, guessed "Edwards Air Force Base", and answered with
the bridge entity); `5adf75e35542993344016c98` (both grounded, DAG cheaper, 483 vs
697 tokens); and `5a8756f25542994775f607dc`, gold `7.63×25mm Mauser`, where both
scored EM 1.0 and **neither** read the sentence — the workflow answered from the
model's knowledge of the C96, and react saw the gold title in a similar-results
trailer and finished without opening it.

So treat HotpotQA accuracies as **upper bounds on retrieval-grounded
answering**, never as bridging ability — the invented-entity chains in
`harness/chain.py` are the only arm comparison memorisation cannot reach.

## Gotchas

- **Corporate TLS.** Python's `ssl` ignores the macOS keychain, so HTTPS fails
  with `self-signed certificate in certificate chain` on hosts that work in curl.
  `dataset.py::bootstrap_ca` exports both keychains to `data/ca-bundle.pem`;
  `llm.py::_resolve_verify` and the downloader both fall back to it. For Node
  tooling use `NODE_EXTRA_CA_CERTS`. The bundle is gitignored (public CA certs
  only, but it is machine-specific).
- **Secrets.** The only credential is `OPENAI_API_KEY` (standard `sk-...`
  format). Never read, copy, or write `.env`. Use `--env-dir`/`--env-file`;
  python-dotenv pushes values into the environment and the harness only ever
  logs the resolved path, never a value. `.env.example` holds placeholder
  names only and is safe to edit.
- **Dataset host.** `curtis.ml.cmu.edu` is long-unreliable; `dataset.py` uses the
  HuggingFace mirror and converts back to the canonical JSON shape (HF stores
  `supporting_facts`/`context` as structs-of-arrays). Expect 7,405 dev items.
- **`latency_p95` is provider tail, not architecture** — ~7% of calls hit ~48s
  regardless of arm. Reason about `llm_calls_mean` instead.
- **Dollar figures are placeholders** (`pricing.json`). Tokens are metered; costs
  are not authoritative.
- **`viewer/index.html` is dependency-free** — hand-rolled SVG, no build step. It
  must be served over HTTP (`serve.py`); `file://` blocks the results fetch.
- **The chart palette is validated** for CVD and contrast at exactly 3 categorical
  series, light and dark. Light slot 3 is sub-3:1 against the surface, which is
  why bars carry direct value labels and a table view ships — do not remove
  either, and re-validate before adding a 4th arm's colour.
- **Keep the `__main__` block last in `tests/test_harness.py`** — it reflects over
  `globals()`, so tests defined below it silently never run. This has happened.

## Public repo

This repository is public. Before any push: no real API keys, no internal
hostnames, gateway names or team names, no personal or customer data. The only
credential is `OPENAI_API_KEY`, supplied via the environment or a gitignored
`.env`. `results/*.json` are committed — check them too.
