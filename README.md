[![RaoLM — a language model with its citations baked in. Three strands, text, code and visuals, braid into one thread.](README_Assets/og-image.png)](https://lm.rao.nyc)

# RaoLM

RaoLM is a small SmolLM2-shaped language model with citations built in. It pretrains on the
documents a [Thread](https://github.com/rao-studios/Thread) node governs, and every token it generates carries a citation
back to the exact Thread source that supports it:

```
(Thread node id, document id, partition index, token offset)
```

Training records an **entropy ledger** per optimizer step and per epoch. A **provenance
index** couples the model's logits to corpus positions (kNN-LM style). A **run manifest**
hashes the weights, the index, the Thread corpus snapshot and the tokenizer together.
Cited spans can be re-checked, token for token, against the live Thread.

It runs on MLX through [Frigate](https://github.com/rao-studios/Frigate) and speaks to Thread over
[Conduit](https://github.com/rao-studios/Conduit)'s gRPC contract.

> This is a first draft and a proof of concept. The model is a tiny from-scratch network
> that memorises a small synthetic corpus, so the citation mechanism can be measured
> precisely. It is not a useful general-purpose model.

## Quick start

Requirements: macOS 15+ on Apple Silicon, Swift 6.3+, sibling checkouts of `../Frigate`,
`../Conduit` and `../Thread` (with Thread built), and SinatraHarness at
`../../../repositories/SinatraHarness` (the path Sewn uses):

```bash
(cd ../Thread && swift build -c release && ./build-metallib.sh release)

swift build -c release
./build-metallib.sh release      # MLX needs its Metal library beside the binary
.build/release/raolm doctor      # checks everything below before you need it
.build/release/raolm demo
```

`raolm demo` runs the whole proof in about two minutes on an M4 Max:

1. It generates a deterministic fictional archive (200 documents, 622 paragraphs, 668 facts), in which every fact is stated in exactly one paragraph.
2. It launches a Thread in open mode on its own storage, with on-device embeddings, and deposits the corpus (`ThreadQuery.Index`, one partition per paragraph).
3. It exports the corpus back (`ThreadLibrary.ExportCorpus`), checks it is byte-identical, and hashes it into a snapshot.
4. It pretrains a 17.3M-parameter SmolLM2-shaped model on that snapshot, writing the ledger and a checkpoint every epoch, and builds the provenance index.
5. It generates with citations, evaluates 50 facts, and verifies every cited span against the live Thread.

Everything lands under `~/Documents/raolm-db` (or `--data-dir` / `RAOLM_DATA_DIR`).

`scripts/cli.sh` does the build steps for you. It rebuilds `raolm` only when a source in RaoLM
or a path dependency is newer than the binary, installs the Metal library the first time, and
passes every argument through. With no arguments in a terminal it opens the studio:

```bash
scripts/cli.sh                       # the studio
scripts/cli.sh doctor
scripts/cli.sh demo
RAOLM_CONFIG=release scripts/cli.sh demo
```

It never changes directory, so relative paths resolve against yours, and its own messages go to
stderr. `RAOLM_SKIP_BUILD=1` runs the existing binary, `RAOLM_BUILD=always` forces a build, and
`RAOLM_METALLIB=rebuild` reinstalls the Metal library after a Frigate update. The
[CLI guide](#cli-guide) walks through every command.

## The studio

`raolm ui` (or bare `raolm` in a terminal, or `scripts/cli.sh`) opens every tool in one
full-screen terminal UI, under the italic RaoLM wordmark and the braid:

```
    ____              __    __  ___      v0.1.0
   / __ \____ _____  / /   /  |/  /      a language model with its citations baked in
  / /_/ / __ `/ __ \/ /   / /|_/ /       ~/Documents/raolm-db  ·  thread ● 3f1c8a2e :8095  ·  4 runs
 / _, _/ /_/ / /_/ / /___/ /  / /
/_/ |_|\__,_/\____/_____/_/  /_/
∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿~~~~~~~~~~~~────────────────────────────────────────────────────────────────◉
```

| Screen | What it does |
|---|---|
| 1 Home | Every run under the data root: status, epochs, memorisation, indexed epochs, corpus hash, Thread node, evaluation. |
| 2 Tests | The doctor's checks, and the test suites (`scripts/test.sh`, with or without the MLX and Thread suites) streaming their output. |
| 3 Corpus | Dataset generation: the synthetic archive, its documents, partitions and facts; ingest into the Thread, export back as a snapshot, or snapshot offline. |
| 4 Thread | The Thread node: identity, health, what it stores, ports and log; start and stop. |
| 5 Train | Pretraining with the settings `raolm train` takes, live: progress and ETA, the loss curve, entropy, learning rate, gradient norm, throughput and the epoch table. |
| 6 Generate | Cited generation and the output-contribution debugger (below). |
| 7 Eval | The evaluation protocol: λ sweep, calibration, controls, span verification and the grounding groups; any outcome opens in the debugger. |
| 8 Ledger | The entropy ledger: epochs, a document's partitions over epochs, a fact's answer-token losses. |

[![The Generate screen with its eight regions outlined in colour and numbered (header, tabs, prompt and generation, token, the citations tables, partition, key hints, status bar), above a legend of the regions and of the marks drawn in the studio's colours: token confidence colours, the cursor, the prompt marker, span markers, the selected row, entropy bars, confidence pips, check marks, grounding glyphs and the Thread status](README_Assets/studio-legend.png)](README_Assets/studio-generate.png)

<details>
<summary>The legend as text</summary>

The screen, top to bottom:

| # | Region | What it shows |
|---|---|---|
| 1 | Header | The RaoLM mark and braid, the screen, the run, its indexed epoch and λ; on the right, the Thread (`●` up, `○` down) or the running job. |
| 2 | Tabs | The eight screens; the current one in blue. `1`–`8` switch. |
| 3 | Prompt ▸ generation | The prompt field, then the prompt (dim) and the generated tokens. The footer says whether the prompt is text or a corpus slice, and after `v`, how many spans verified. |
| 4 | Token | The token under the cursor: its four entropies with bars, p_LM, kNN agreement, p_mix, λ, confidence, how many partitions cite it, and its grounding after `G`. |
| 5 | Citations │ Neighbours │ Spans │ Grounding | Tabs over one table (`tab` switches): the partitions that cite the token, the retrieved neighbours, the verbatim spans, or the grounding attribution. |
| 6 | Partition | The selected citation's Thread node, document, partition, token offset, `raolm://` URL and text hash, and the partition's text with the cited tokens highlighted. The footer shows its verification. |
| 7 | Key hints | The keys this screen takes. `?` lists them all. |
| 8 | Status bar | The Thread, the data root or the running job; on the right, the last message or error. |

| Mark | Meaning |
|---|---|
| Token colour | Citation confidence: dim uncited, blue below 0.5, green below 0.8, white below 1.0, gold underlined inside a verified span. |
| Reverse video | The cursor. `←` `→` move it. |
| `▸` | Where the prompt ends and the generation begins. |
| `[[n]]` | The end of a verbatim span, numbered by source, the same numbers the CLI prints. |
| `▶` | The selected row. |
| `████░░` and `●●○○` | An entropy's size beside the others, and confidence in quarters. |
| `✓` `✗` `◉` | Passed, failed, verified. |
| `●` `◐` `✗` `·` | In the grounding band: grounded, unsupported, contradicted, function word. |

</details>

Above, the model is prompted with "Hello there", which appears nowhere in the corpus, and writes
about the Council of Northstow. The cursor is on " North" in the first "Northstow":

- **The token box** shows why it stands out. All four entropies are high (2.4 to 2.9 nats), only 6% of the retrieval weight predicts this token, and confidence is 0.09, so the token is coloured blue rather than white or gold.
- **The citations** show where the token came from. Its one citation is a weak match (rank 12, source loss 3.79 nats) to *The Council of Northcrest*.
- **The partition panel** shows that source: Thread node, document id, partition 0, token offset 3, its `raolm://` URL and text hash, and its live text with " North" highlighted.

The council's name is being assembled from another council's record, and the debugger shows it
token by token. The two `[[1]]` markers later in the strip close verbatim spans copied from the
generation's one source.

The debugger colours each generated token by citation confidence (dim uncited, blue, green, white,
gold underlined inside a verified span) and marks verbatim spans with the same `[[n]]` numbers the
CLI prints. For the token under the cursor it shows the model's, retrieval's, mixture's and
source's entropies, p_LM, kNN agreement and confidence. It also shows the partitions that support
the token with their weights, ranks, scores and source losses, and every retrieved neighbour.
The panel under it shows the cited partition's Thread node id, document, partition index, token
offset, `raolm://` URL, Thread partition id, text hash and memorisation epoch, with the cited
tokens highlighted in the partition's live text. `G` grounds the generation (next section).

Keys: `1`–`8` switch screens, `?` lists every key, `q` quits (asking first while training or
evaluating). `raolm ui --fixtures Tests/RaoLMStudioTests/Fixtures/demo` replays a recorded run
with no MLX and no Thread. `RAOLM_COLOR=truecolor|256|16|none`, `RAOLM_ASCII=1` and
`RAOLM_UI_THEME=terminal` adapt it to the terminal; what libraries print goes to
`<data root>/logs/studio.log`.

## CLI guide

Every tool is a `raolm` subcommand, and the studio runs the same code. The examples use
`scripts/cli.sh`, which builds `raolm` when needed; `.build/release/raolm`, or `raolm` on your
`PATH`, takes the same arguments. Everything is written under the data root, `~/Documents/raolm-db`
unless `--data-dir` or `RAOLM_DATA_DIR` says otherwise. `raolm <command> --help` lists every flag.

The steps below go from nothing to a trained, cited, verified, grounded and evaluated run.
`scripts/cli.sh demo` does steps 1 to 8 in one go, on a Thread of its own that it stops at the end.

### 1. Check the setup

```bash
scripts/cli.sh doctor
```

It checks the Metal library, the tokenizer, the Thread binary and its Metal library, the embedding
model, the ports and the disk. A failed check says what is missing, and the command exits 1.

### 2. Generate a dataset

```bash
scripts/cli.sh corpus generate --documents 200 --seed 42
scripts/cli.sh corpus show ~/Documents/raolm-db/corpora/veldmar
```

The generator is deterministic: the same seed gives the same archive, and every fact is stated
in exactly one paragraph. One paragraph is one partition, the unit a citation points to. The
corpus lands in `<data root>/corpora/<slug>` with its `facts.jsonl`. `--slug`, `--max-chars` and
`--force` change the name, the paragraph size and whether an existing corpus is overwritten.

### 3. Put it in a Thread and take a snapshot

```bash
scripts/cli.sh thread start --detach
scripts/cli.sh corpus ingest
scripts/cli.sh corpus pull --expect-corpus ~/Documents/raolm-db/corpora/veldmar
scripts/cli.sh thread status
```

- **`thread start`** runs a Thread node in open mode on its own storage (`<data root>/thread-db`, HTTP 8095, gRPC 9095), embedding on device. `--fresh` wipes that storage first.
- **`corpus ingest`** deposits the corpus with `ThreadQuery.Index` and waits until every document is indexed.
- **`corpus pull`** exports it back with `ThreadLibrary.ExportCorpus`, checks it byte for byte against the generated corpus, and saves a hashed snapshot. It prints the snapshot's path, `<data root>/snapshots/<hash>/snapshot.json`.

Training only ever reads a snapshot, and the snapshot's corpus hash goes into the run manifest.
To train without a Thread, take an offline snapshot on the studio's Corpus screen (`O`).

### 4. Pretrain

```bash
scripts/cli.sh train --corpus ~/Documents/raolm-db/snapshots/<hash>
RUN=$(ls -d ~/Documents/raolm-db/runs/* | tail -1)      # runs are named by start time
```

Training prints the first steps, then a step line every few seconds and a line per epoch. Each
epoch writes the entropy ledger and a checkpoint, and the final epoch builds the provenance
index. The main flags:

- **Model:** `--preset tiny|small|smollm2-135m`.
- **Schedule:** `--epochs`, `--batch-size`, `--seq-len` and `--lr`.
- **Eval and index epochs:** `--eval-every` and `--index-every`.
- **Stopping:** `--early-stop`, the memorised fraction at which training stops.
- **Leave-out control:** `--exclude-documents N` holds N documents out of training.

A finished run is marked `complete` in its `run.json`.

### 5. Generate with citations

```bash
scripts/cli.sh generate --run $RUN --prompt "Construction of the Hollow Lighthouse was completed in"
scripts/cli.sh generate --run $RUN --prompt-from <document id>:0:0:26 --trace --json gen.json
```

- **`--prompt`** takes any text.
- **`--prompt-from DOC:PARTITION:OFFSET:LENGTH`** takes a slice of a partition's own tokens, so the prompt is exactly what the model trained on and carries its source address.
- **`--trace`** adds a table of every generated token: H_lm, H_mix, kNN agreement, confidence and top citation.
- **`--json`** saves the full record: traces, neighbours, citations, spans and the hashes it is bound to.
- **`--lambda`** sets the retrieval weight in the logits; 0 means evidence only. `--temperature`, `--top-k` and `--max-tokens` control sampling, and `--format plain|json` changes the printed output.

### 6. Verify the citations

```bash
scripts/cli.sh verify --run $RUN --generation gen.json            # against the live Thread
scripts/cli.sh verify --run $RUN --generation gen.json --offline  # against the run's snapshot
```

Every verbatim span is re-read from its source and compared token for token. The result is
written back into `gen.json`, and the command exits 3 if any span no longer matches.

### 7. Ground the generation

```bash
scripts/cli.sh ground --run $RUN --generation gen.json --sources top
```

It scores the same output with and without its sources in front of the prompt (see
[Grounding](#grounding-the-two-fold-harness)). `--sources` picks them:

- **`top`:** each generated token's top citation.
- **`spans`:** the verbatim spans' partitions.
- **`all`:** every cited partition.
- **`fact`:** the partition the prompt was sliced from.
- **`DOC:P[,DOC:P]`:** partitions you name.

`--detail full` adds each token's rank without the source and the tokens the source pushed
toward. `--json` prints the record instead of tables. The record is saved as
`<run>/generations/<generation id>.grounding.json` (`--out` puts it elsewhere).
`generate --ground` does steps 5 and 7 together, and saves the record beside its `--json` file.

### 8. Evaluate

```bash
scripts/cli.sh eval --run $RUN --offline --grounding --facts-sample 50
```

For a sample of facts, the model is prompted with the corpus tokens before each answer. The
report covers:

- **The λ sweep** (default `0,0.25,0.5,0.75`): exact answers and citation@1 at each weight.
- **Calibration:** ECE and AUROC of citation confidence.
- **Span verification** of every answer.
- **Controls:** paraphrased prompts, fabricated entities and held-out documents; `--no-controls` skips them.
- **Grounding groups**, with `--grounding`.

The report is saved as `<run>/eval.json`, and every generation it made is saved under
`<run>/generations/`. Without `--offline`, spans are verified against the live Thread.

### 9. Read the ledger

```bash
scripts/cli.sh ledger --run $RUN                                        # one row per epoch
scripts/cli.sh ledger --run $RUN --document <document id> --partition 0  # a partition over epochs
scripts/cli.sh ledger --run $RUN --fact <fact id or suffix>             # a fact's answer-token losses
```

`--json` prints the ledger rows instead of a table.

### 10. Stop the Thread

```bash
scripts/cli.sh thread stop
```

### Reference

| Command | What it does |
|---|---|
| `raolm doctor` | Checks the Metal library, tokenizer, Thread binary and its metallib, embedding model, ports and disk. |
| `raolm corpus generate` / `show` | Generates the synthetic corpus (`--documents`, `--seed`) or summarizes one. |
| `raolm thread start` / `stop` / `status` | Hosts a Thread in open mode on `<data root>/thread-db` (HTTP 8095, gRPC 9095; `--detach` to background it). |
| `raolm corpus ingest` / `pull` | Deposits a corpus into a Thread, or exports it into a hashed snapshot. |
| `raolm train --corpus <snapshot>` | Pretrains with the ledger and provenance index (`--preset`, `--epochs`, `--exclude-documents`, …). |
| `raolm generate --run <run>` | Generates with citations (`--prompt` or `--prompt-from DOC:PARTITION:OFFSET:LEN`, `--lambda`, `--format markers\|plain\|json`, `--trace`, `--ground`). |
| `raolm verify --run <run> --generation <json>` | Re-checks every cited span against the live Thread (`--offline`: the snapshot). Exits with 3 if any span fails. |
| `raolm ground --run <run> --generation <json>` | Measures a generation against its sources with and without them (SinatraHarness): ι, drift, risk, attribution beside the citations. |
| `raolm eval --run <run>` | Runs the evaluation protocol: λ ablation, calibration, and paraphrase, fabricated-entity and leave-out controls (`--grounding` adds the two-fold harness). |
| `raolm ledger --run <run>` | Shows per-epoch summaries, a partition's trajectory (`--document`) or a fact's answer losses (`--fact`). |
| `raolm demo` | Runs everything above end to end (`--no-grounding`, `--no-controls`, `--keep-thread`, `--reuse-thread`). |
| `raolm ui` | The full-screen studio; bare `raolm` in a terminal opens it (`RAOLM_NO_UI=1` prints help instead). |

A run directory contains `run.json` (manifest and hashes), `ledger/`, `checkpoints/epoch-N/`,
`provenance/epoch-N/`, `generations/` (with any `.grounding.json` records), `eval.json` and
`thread.log`.

Exit codes follow sysexits, so scripts can branch on them:

| Code | Meaning |
|---|---|
| 0 | Success. |
| 1 | `doctor` found a problem. |
| 3 | `verify` found a span that no longer matches its source. |
| 64 | Usage: a bad flag or value. |
| 65 | Data: a hash or text mismatch, a rejected index request, a corrupt index. |
| 66 | Missing input: no run, corpus, snapshot, checkpoint or index. |
| 69 | Unavailable: usually the Thread is not reachable. |
| 70 | Internal error. |
| 73 | Cannot create: the output already exists (`--force` overwrites). |
| 75 | Temporary: a timeout, or a grounding measurement over its budget. |
| 78 | Configuration: the Metal library or the tokenizer is missing. |
| 130 | Interrupted. |

Environment variables:

| Variable | Effect |
|---|---|
| `RAOLM_DATA_DIR` | The data root (default `~/Documents/raolm-db`). |
| `RAOLM_THREAD_BINARY` | The Thread binary (default `../Thread/.build/release/thread`). |
| `RAOLM_TOKENIZER_DIR` | A tokenizer folder to use instead of the vendored SmolLM2 one. |
| `RAOLM_NO_UI` | Bare `raolm` prints help instead of opening the studio. |
| `RAOLM_UI_FIXTURES` | The studio replays these fixtures (as `raolm ui --fixtures`). |
| `RAOLM_COLOR`, `RAOLM_ASCII`, `RAOLM_UI_THEME`, `NO_COLOR` | The studio's colour depth, glyphs and background. |
| `RAOLM_CONFIG`, `RAOLM_BUILD`, `RAOLM_SKIP_BUILD`, `RAOLM_METALLIB` | How `scripts/cli.sh` builds (see Quick start). |
| `RAOLM_MLX_TESTS`, `RAOLM_THREAD_TESTS` | Which gated test suites run. |

## Grounding: the two-fold harness

A citation says which corpus positions support a token. It cannot say whether the source *moved*
the model: a memorised token is cited whether or not its source is in front of the model.
[SinatraHarness](https://github.com/riteshpakala/SinatraHarness) answers that second question by scoring the
same output twice, once with the cited source partitions in front of the prompt and once without
them. For each token that gives:

- **ι**, the log-probability the source added;
- **KL** between the two distributions;
- **drift**, how much less the output took than the source gave (KL − ι);
- **risk**, 1 − p on tokens the source did not back;
- per partition, **A_p**, the source's share of the influence, beside RaoLM's citation weight.

The bare side is exactly the prompt the generation saw, so its log-probabilities and entropies
reproduce the trace's p_LM and H_lm; the record reports that gap, which is about 1e-6.

```bash
raolm ground --run <run> --generation <run>/generations/<id>.json --sources top   # or spans, all, fact, DOC:P
raolm generate --run <run> --prompt-from DOC:0:0:12 --ground --sources fact
raolm eval --run <run> --offline --grounding          # grounding per fact and per control group
```

`raolm eval --grounding` measures every answer against its fact's true source, and does the
same for the paraphrase, fabricated-entity and held-out controls. It reports each group's
grounding, drift, context dependence and hallucination risk. It also reports how well the risk
predicts a wrong answer (AUROC), and the correlation between an answer's ι and RaoLM's citation
confidence. `raolm demo` includes the grounding section (`--no-grounding` skips it). Records are
saved as `<run>/generations/<generation id>.grounding.json`.

## What a run shows

These are the numbers from a recorded `raolm demo` run: 1,080 optimizer steps, and 97% of
corpus positions memorised (eval loss 0.059 nat) by epoch 60.

| λ (retrieval weight) | exact answer | citation@1 | on correct answers | answer inside a verified verbatim span |
|---|---|---|---|---|
| 0 (evidence only) | 74% | 86% | 97% | 70% |
| 0.5 (logits coupled to the corpus) | 86% | 91% | 97% | 82% |

- **Every cited span checked out.** All 86 verbatim spans were re-read from the live Thread and matched token for token.
- **Coupling helps.** Mixing retrieval into the logits (λ = 0.5) raises exact answers from 74% to 86%.
- **Confidence ranks well but is not a probability.** It separates right from wrong citations with an AUROC of 0.946, but its calibration error is 0.23. At confidence ≥ 0.9, 100% of citations were right.
- **Control: invented entities.** Prompts about entities that do not exist get mean confidence 0.43, against 0.83 on correct answers.
- **Control: paraphrases.** Paraphrased prompts are never answered, which is expected for a model that memorised rather than generalised.
- **Control: leave-out.** A separate run held out 20 documents (`--exclude-documents 20`). Facts from those documents were answered 0% of the time, and no answer token cited them. That is the causal check that answers come from training on the source.

### What a citation catches

Here the model is prompted with a sentence typed by hand rather than taken from the corpus:

```
$ raolm generate --run <run> --prompt "Construction of the Hollow Lighthouse was completed in"
Construction of the Hollow Lighthouse was completed in▸ 21[[1]]54. The Hollow Lighthouse was designed by Mabel Corrow. Later clerks[[2]] …

  [[1]] The Hollow Lighthouse — raolm-veldmar-d74ac13a… p0 tokens 26..<29 · verified
  [[2]] The Hollow Library    — raolm-veldmar-276d0e60… p0 tokens 41..<51 · verified
```

The Lighthouse was completed in 2174, and its architect is Pador Halrow. The model blended
it with a similarly named document, and the citations show exactly where:

- **"21"** is cited verbatim to the Lighthouse's own record.
- **"5"** is the token with the highest model entropy (1.67 nats) and the lowest retrieval agreement (0.32). Its citation jumps to another document, at confidence 0.38.
- **"designed by Mabel Corrow"** is cited, verbatim and verified, to **The Hollow Library**. That is where the name really comes from.

## How it works

```
SyntheticCorpus ─Index─▶ Thread node ─ExportCorpus─▶ CorpusSnapshot (corpus hash)
                                                          │
                                                          ▼
                 Pretrainer (MLX, AdamW, per-step + per-epoch entropy ledger, checkpoints)
                                                          │ deterministic eval pass
                                                          ▼
     ProvenanceIndex: key = [mid-layer ⊕ final hidden], value = next token, address, loss, entropy
                                                          │
prompt ─▶ CitedGenerator: p = λ·p_knn + (1−λ)·p_lm ─▶ tokens + traces + citations + verbatim spans
                                                          │
                                                          ▼
                        CitationVerifier: re-read cited partitions from the live Thread
```

[Docs/PROVENANCE.md](Docs/PROVENANCE.md) explains the citation address, the hash chain, the
ledger, the key design, the confidence formula and the evaluation protocol. It also says
precisely what "generative proof" means here and what it does not.

## The model

`RaoTransformer` is SmolLM2's architecture: pre-norm blocks of grouped-query attention with
RoPE, a SwiGLU MLP, RMSNorm and a tied output head. It uses SmolLM2's 49,152-token tokenizer,
vendored in `Sources/RaoLMModel/Resources/Tokenizer` with its license notice.

| Preset | Hidden | Layers | Heads (KV) | Parameters |
|---|---|---|---|---|
| `tiny` (default) | 256 | 6 | 4 (2) | 17.3M |
| `small` | 384 | 8 | 6 (2) | ≈29M |
| `smollm2-135m` | 576 | 30 | 9 (3) | 135M |

Weights use the Hugging Face Llama names, and a checkpoint directory is a plain Llama model
folder (`config.json`, `model.safetensors`, tokenizer files):

- **Frigate** loads it as `LlamaModel`, and its logits match (checked by a test).
- **Python `transformers`** loads it as `LlamaForCausalLM`, and on the prompt checked its greedy output matched RaoLM's token for token.

## Package layout

| Target | Role |
|---|---|
| `RaoLMCore` | Corpus model, hashing, synthetic corpus, snapshots, manifests, ledger rows, citation schemas and span logic. Foundation only. |
| `RaoLMModel` | The transformer, provenance key, tokenizer and checkpoint I/O (MLX via Frigate). |
| `RaoLMTraining` | Tokenized corpus, sampler, loss, pretraining loop, eval pass, ledger recorder and indexer. |
| `RaoLMProvenance` | Index query, logit mixing, cited generation, verifier, run loader and fact evaluator. |
| `RaoLMThread` | gRPC client for a Thread node and a host that runs one (Conduit). |
| `RaoLMGrounding` | The two-fold harness: SinatraHarness's with/without-source scoring joined to RaoLM's traces and citations. |
| `RaoLMTerminal` | The terminal toolkit: raw mode, a diffing cell renderer, colour degradation, widgets, the banner, the app loop. Foundation only. |
| `RaoLMWorkflows` | What the CLI and the studio share: the training driver, doctor checks, tables, failure mapping. |
| `RaoLMStudio` | `raolm ui`: studio state, screens, the MLX worker thread, the fixture backend. |
| `RaoLM` / `raolm` | The umbrella module and the CLI. |

## Tests

```bash
scripts/test.sh                                              # every suite, MLX included
scripts/test.sh --filter RaoLMCoreTests                      # just the pure suites
scripts/test.sh --filter RaoLMTerminalTests                  # the terminal toolkit, no TTY needed
scripts/test.sh --filter RaoLMStudioTests                    # every screen rendered from the fixtures
RAOLM_THREAD_TESTS=1 scripts/test.sh --filter RaoLMThreadTests   # launches a real Thread
```

`scripts/test.sh` puts MLX's Metal library inside the test bundles and runs `swift test
--skip-build`. It exists because those copies break the bundles' ad-hoc re-sign on the next
build. The MLX suites are gated by `RAOLM_MLX_TESTS=1` (or the family's
`FRIGATE_MLX_TESTS=1`), and the script sets it for you.

## Known limits and next steps

- **Calibration.** Confidence ranks citations well (AUROC 0.95) but overstates mid-range certainty (ECE 0.23). A post-hoc calibration fitted on held-out facts is the obvious next step.
- **Scale.** Retrieval is exact search over every corpus position. A larger Thread needs an approximate index, or a key projection (`index.json` already has a field for one).
- **Resumability.** Optimizer state is not checkpointed (MLXOptimizers keeps it internal), so a checkpoint restores weights only.
- **Reproducibility.** Metal training is not bit-reproducible. Hashes name the weights that exist.
- **Real corpora.** The key layer and mix are defaults, not tuned. Personal Thread corpora will need a warm start from SmolLM2 weights rather than from-scratch memorisation. The checkpoint loader is written to accept HF SmolLM2 snapshots, but that path is untested against the real weights.

## License

Apache License 2.0 (see [LICENSE](LICENSE)). The vendored SmolLM2 tokenizer files are also Apache-2.0; see
`Sources/RaoLMModel/Resources/Tokenizer/NOTICE`. RaoLM links Frigate (MIT) and Conduit
(Apache-2.0). Thread is only ever run as a separate process, never linked.
