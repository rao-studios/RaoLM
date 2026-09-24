# How RaoLM cites its sources

This document explains what a RaoLM citation is, how it is produced, what it proves and
what it does not. It is written for someone who has not read the code.

## The citation address

A Thread stores each document as an ordered list of **partitions** (RaoLM deposits one
paragraph per partition). Thread's own partition ids hash the embedding vector, so they
change whenever text is re-embedded. RaoLM therefore never uses them as the address. A
citation points at:

```
(Thread node id, document id, partition index, token offset)
```

- **Thread node id**: the UUID the node was launched with (`--node-id`), recorded in every snapshot and run.
- **Document id**: `raolm-<slug>-<first 24 hex of SHA-256(canonical text)>`, chosen by RaoLM and stored by Thread as given.
- **Partition index**: the partition's position in the document, which is its position in `ExportCorpus` output.
- **Token offset**: the position inside that partition's *own* tokenization. Each partition is tokenized on its own, so the offset can be recomputed from the partition's text alone.

RaoLM also hands Thread a per-partition url, `raolm://<slug>/<document id>/p/<index>`.
Thread keeps it and returns it with every exported partition, so the address survives inside
the Thread.

## From Thread to weights: the hash chain

1. **Ingest.** `ThreadQuery.Index` deposits each document with its id, name, partition texts and partition urls.
2. **Snapshot.** `ThreadLibrary.ExportCorpus` returns the corpus exactly as Thread stores it. RaoLM hashes it:

   ```
   corpusHash = SHA-256( for each (document id, partition index) in sorted order:
                           "<document id>\t<partition index>\t<SHA-256(partition text)>\n" )
   ```

   `raolm demo` checks that the export is byte-identical to what was generated before training starts.
3. **Training.** Training reads only the snapshot, and `run.json` records its `corpusHash`, the tokenizer's SHA-256, the hyperparameters, and for every epoch the SHA-256 of the checkpoint, of that epoch's ledger file and of the provenance index.
4. **Index.** `index.json` records the checkpoint hash it was built from. Generation refuses to run when the checkpoint on disk hashes differently.
5. **Generation.** Every cited generation carries a `ManifestRef` (run id, epoch, checkpoint, index, corpus and tokenizer hashes, Thread node id).

Each hash can be recomputed by hand with `shasum -a 256`.

## The entropy ledger

The trainer computes, from the same pass over the logits, each token's cross-entropy
(`logSumExp(z) − z[y]`) and the model's predictive entropy (`logSumExp(z) − Σ softmax(z)·z`).
It writes them under `<run>/ledger/`:

| File | One row per | What it records |
|---|---|---|
| `steps.jsonl` | optimizer step | loss, entropy summary (mean, p10, p50, p90, max), loss − entropy, gradient norm before clipping, learning rate, throughput |
| `partitions-epoch-N.jsonl` | partition, every epoch | training loss and entropy from the steps that touched it; on eval epochs the deterministic eval loss, entropy, p90 and max loss, memorised fraction, and the first epoch at which ≥ 95% of it was memorised |
| `facts-epoch-N.jsonl` | fact, eval epochs | the loss of every answer token: when the model learned the fact |
| `epochs.jsonl` | epoch | summary plus checkpoint, ledger and index hashes |

Positions whose *input* is `<|endoftext|>` are excluded everywhere: under document packing
they predict the first token of an unrelated document. A token counts as **memorised** when
its loss is below 0.1 nat, which means the model gives the true next token at least 90%
probability.

The **eval pass** is deterministic. It runs per document, never across documents, in sliding
windows that give every position at least half a window of left context. It produces the
ledger's eval columns and the provenance index.

## The provenance index

For every corpus transition (context → next token) the index stores:

- a **key**: `[√α · tap/‖tap‖, √(1−α) · final/‖final‖]`, where `tap` is the residual stream after block `tapLayer` (the middle block by default) and `final` is the normed hidden state that feeds the tied output head.
- the **value**: the token that followed.
- where the context sits and where the value sits: `(row, offset)` for each.
- that transition's eval **loss** and **entropy** under the indexed epoch's weights.

Why two layers? With tied embeddings the final hidden state at every position that predicts
token *y* is pulled toward the same embedding row. Final-layer neighbours therefore cluster
by *next token*, for example every "…in 1" position predicting "8". The mid-layer half keeps
the context (entity names, local phrasing) that tells sources apart. The final half keeps
next-token agreement.

## Generating with citations

At every position the generator:

1. Builds the key for the current position and retrieves the `k` nearest index entries (exact cosine search, one matmul).
2. Turns their scores into weights, `w = softmax(score / τ)`, and forms `p_knn(y) = Σ w_i · [value_i = y]`.
3. Mixes: `p = λ · p_knn + (1 − λ) · p_lm`. With `λ > 0` the emitted distribution is a function of the retrieved corpus positions, so the citation is part of how the token was chosen, not a lookup afterwards. `λ = 0` is evidence-only mode.
4. Chooses the token, then records a trace: model entropy, retrieval entropy, mixture entropy, *source entropy* (how spread the retrieval weight is across partitions; high means generic phrasing), the neighbours, and citations grouped by partition.

The **citation address is the value position**: the emitted token is the corpus token that
*followed* the retrieved context, one position after the key.

**Citation confidence** for a token and a partition *c* is the geometric mean

```
( simTerm(best cosine in c) · support(c) · exp(−loss of that corpus transition) )^(1/3)
simTerm(s) = clamp((s − 0.5) / 0.5, 0, 1)
support(c) = total retrieval weight of the matching neighbours in c
```

It is high only when the context is close to the cited one, retrieval agrees on this token
from this partition, and the model had memorised that exact transition.

A token no retrieved context predicted is reported as **uncited**. RaoLM does not invent a
source for it.

### Verbatim spans

A **verbatim span** is at least three consecutive emitted tokens that walk one partition
position by position: token *t* has a matching neighbour of rank ≤ 3 whose value position is
(row *r*, offset *o + t*). Overlapping candidates are resolved greedily (longest, then
heaviest), and the losers are kept as `alternatives`. Each span also carries a
**distinctiveness**: the fraction of its token 3-grams that occur in only one document.
Zero means boilerplate that many documents share. A span that lies entirely inside the prompt
is marked `promptVerbatim` and shown, but it never counts as evidence.

In `markers` output each verbatim span is followed by a Sewn-style `[[n]]` marker, with a
numbered source list, so the output drops into the family's attribution UI.

## Verification

`raolm verify` re-reads each cited partition from the **live Thread** (`ThreadLibrary.Documents`),
re-tokenizes it, and compares the tokens at the recorded offset:

| Status | Meaning |
|---|---|
| `verified` | The live partition's tokens at the offset equal the span's tokens. |
| `stale` | The partition's text changed since the index was built. |
| `mismatch` | The text is unchanged but the tokens at the offset differ, which is a pipeline bug. |
| `tokenizerDrift` | The text at the offset matches but the token ids differ. |
| `missing` | The document or partition is gone from the Thread. |

The verifier refuses to run with a tokenizer whose hash differs from the run's.

Verification is an **integrity check** of the chain from Thread text to index to weights. It
is not evidence about the model: spans are built from corpus positions whose next token *is*
the emitted token, so a span can only fail if the corpus or the pipeline changed.

## What "generative proof" means here, precisely

For a verbatim span, RaoLM can show all of the following:

1. The exact weights, index, corpus snapshot and tokenizer, named by SHA-256.
2. That the span occurs verbatim at the recorded offset of the recorded partition in the live Thread.
3. How strongly those weights had memorised each of its tokens at that position (the ledger and `exp(−loss)`).
4. That the emitted distribution was a function of those corpus positions (λ > 0).

It does **not** show that the cited document *caused* the output. No counterfactual is run:
another document, or generalisation, might produce the same tokens, and `alternatives` lists
the other sources that fit. Confidence is evidence of memorisation and retrieval support, not
an influence certificate. The evaluation's leave-out control (`raolm train
--exclude-documents N`) is the only causal evidence in this draft: facts from documents the
model never saw should neither be answered nor cited.

## Evaluation protocol

`raolm eval` samples facts from the synthetic corpus. Each fact is stated in exactly one
partition. The prompt is the corpus's own token slice leading up to the answer, and the run
uses greedy decoding.

| Metric | Definition |
|---|---|
| exact | the first answer-length generated tokens equal the answer's tokens |
| citation@1 | share of answer tokens whose top-ranked neighbour's value lies in the fact's partition (τ-independent) |
| offset@1 | same, at the exact expected token offset |
| cited@1 | share of answer tokens whose top *citation* is the fact's partition |
| covered | the whole answer lies inside one verbatim span of the fact's partition that verified against the live Thread |
| ECE / AUROC | calibration of citation confidence against "top citation is the right partition", pooled over answer tokens, paraphrased prompts and prompts about fabricated entities |

The controls are **paraphrased prompts**, whose stems never occur in the corpus, and
**fabricated entities**: the same template about something that does not exist, where any
citation is wrong by construction.

## Limitations of this draft

- Training on Metal is not bit-reproducible (scatter-add atomics). Hashes name the weights that exist rather than promising re-derivation.
- Optimizer state is not checkpointed, so a checkpoint restores weights only.
- Retrieval is exact search over every corpus position. That is fine for this corpus but linear in its size, and a larger corpus needs an approximate index or a projection.
- The key layer (`--tap-layer`) and mix (`--alpha`) are defaults, not tuned. The evaluation is how to tune them.
- The model is a tiny from-scratch SmolLM2 shape that memorises a small corpus. It is a proof of the citation mechanism, not a useful language model.
