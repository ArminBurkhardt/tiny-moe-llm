# IR table reshape — cost, neutrality, and the probe budget (2026-08-28)

The IR expert's table grew from `8192 x 128` to `65536 x 256` and its read changed from an exact
full-table softmax to a two stage centroid probe. This file records the three things that had to be
measured before either sharpening arm was worth running: what the reshape costs in compute, whether
the migrated checkpoint is behaviourally the same model, and how many clusters have to be probed for
the two stage read to see what the exact read would have seen.

The sharpening finetune's own results (Gate G2, arm A vs. arm B) are not here — see
[ir_sharpening.md](ir_sharpening.md) once those runs land.

## 1. Compute: 8x the entries, 2x the dim, less forward compute

`TinyMoETransformer.__init__`'s own print, at `seq_len=4096`:

| | old (`8192 x 128`, exact) | new (`65536 x 256`, two stage) |
|---|---|---|
| total params | 332.3M | **364.0M** |
| active params | 173.1M | 204.8M |
| active excl. embeddings | 103.9M | 135.6M |
| forward FLOP/token | **490M** | **484M** |
| — body | 264M | 258M |
| — heads (`lm_head` per loop + MTP) | 100M | 100M |
| — attention @ 4096 | 126M | 126M |
| IR module alone, per token per loop | 4.194M | **1.720M** |

The table is 8x taller and twice as wide and the block got *cheaper*, because the exact read scores
every entry (`2 * 65536 * (256 + 256)` would have been 67M FLOP/token/loop, 40x the old cost) while
the two stage read scores 256 centroids, then `probe_clusters * capacity = 8 * 256 = 2048`
candidates at a 1.5x capacity allowance, then gathers 32 values. That is the whole argument for the
two stage path: it is what makes a table this size affordable at all, and it is deliberately the
same shape the external evidence store will need later.

**The FLOP accounting had to be fixed to see this.** `loop_flops_per_token` billed the MoE at
`2 * active_params`, which is right for a matmul and wrong for a lookup table — 33.5M of table
parameters would have been charged 67M FLOP/token/loop whether they were read exactly or not.
`transformer.py` now subtracts the table parameters out of the matmul term and adds each IR module's
own `flops_per_token`, which knows which read path it is on.

## 2. Neutrality: the migrated checkpoint is its own read-zeroed ablation

`scripts/migrate_ir_reshape.py` zero-inits `y_values`, so every retrieval returns the zero vector
regardless of what the keys are. The IR expert therefore contributes exactly `g_proj(0)` — a bias —
and the migrated checkpoint is, by construction, the read-zeroed ablation of its source. That is a
claim worth checking rather than asserting, because a reshape touches five tensors and a silent
error in the carry-over of the other ~250 would not announce itself.

`scripts/eval_calibration.py`, same held-out slice and identical flags on both
(`--start-doc-idx 0 --max-batches 40 --batch-size 4`):

| | `checkpoint_phase2_final_phase0` | `..._phase0_irrandom` (migrated) | Δ |
|---|---|---|---|
| per-loop CE | 3.5152 / 3.3922 / **3.3843** | 3.5153 / 3.3924 / **3.3845** | +0.0002 |
| top-1 accuracy | 0.3963 | 0.3964 | +0.0001 |
| ECE(`p_max`) | 0.0089 | 0.0092 | +0.0003 |
| AUROC(`p_max` → correct) | 0.8376 | 0.8374 | −0.0002 |

**+0.0002 nats is the number Stage 0 already measured** for zeroing the IR read on this checkpoint
(0.0002–0.0004 nats, the finding that failed Gate G1 and motivated this whole phase). The reshape
costs exactly what deleting the read costs, i.e. it costs what the read was worth, i.e. nothing. The
old table was a bias term and the new one starts as the same bias term.

This also means the migrated checkpoint is a legitimate *seed*, not a *resume point* — like the
earlier head-removal migration it drops optimizer and scheduler state, since two tensors changed
shape and Adam's moments are indexed by position.

## 3. Probe budget: how many clusters the read has to look in

The two stage read is only faithful if the clusters it probes contain the entries an exact read
would have picked. `InformationRetrievalModule._candidate_recall` measures exactly that — the
fraction of the exact top-32 that survives into the candidate set — and it runs at every cluster
refresh, on the query reservoir, so the number is always read on real `down_proj` outputs.

**Synthetic queries are useless here.** On isotropic random queries against a 2048-entry table,
recall@16 by probed clusters read 1→0.021, 2→0.045, … 32→0.331: near the chance rate `probe/C`,
because random queries have no reason to concentrate anywhere. Real queries are strongly
anisotropic, and on the same table with structured queries the numbers were 1→0.493, 2→0.607,
4→0.730, 8→0.851, 32→1.000. Recall has to be measured on the reservoir, and
`tests/test_ir_two_stage.py` asserts the *shape* of the curve (monotone in probe count, above the
chance rate) rather than an absolute bar, for the same reason.

**On the real 65536-entry table, 4 probed clusters straddled the bar.** A shakedown finetune's seven
refreshes measured recall@32 of 0.9519, 0.8871, 0.8851, 0.8851, 0.9063, 0.8992, 0.8581 — around and
mostly below the 0.9 that `scripts/sft.py`'s `IR_MIN_CANDIDATE_RECALL` warns beneath, and drifting
*down* as the temperature anneal sharpened the read. `ir_probe_clusters` is therefore **8**, not the
4 the plan guessed. It costs 0.79M FLOP/token/loop (484M vs. 481M per token overall, ~0.6%), which
is a cheap way to stop measuring a retrieval the model would not have made.

The 0.9 bar is a warning, not an abort: recall below it means the sharpening is outrunning the
clustering, and the response is more probed clusters or a shorter refresh interval, neither of which
is worth a runtime decision.

## 4. Throughput

The shakedown ran at **52k tokens/sec** and **22.29 GB peak** at micro batch `4 x 4096` on the 5090
in BF16 — within the resident budget, next to the 21.4 GB the same shape cost before the reshape.
The 33.5M new parameters are a lookup table, so they cost memory and almost no bandwidth per token.
