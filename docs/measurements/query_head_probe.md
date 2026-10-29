# Query head probe (2026-10-21): does the trunk's state carry the question for search

Run under `TINY_LLM_CONFIG=config_micro.yaml` (`scripts/probe_query_head.py`).

Arm PI was stopped at 87M on the language bar: the retriever's InfoNCE term, with its gradient
into the trunk, cost 0.30 nats of held-out CE at pilot size and bought no search. Before a
cheaper arm, this probe asks whether the state the search head reads already carries the
question in a form a small map can turn into a usable query. If it does, a head trained alone
(no trunk gradient, no language cost) can search; if it does not, neither a head nor a tower on
detached states can, and the trunk would have to change. Answer: it does not, at micro scale. No
head on the frozen state, at the last token or pooled over the prompt, gets past about 7% of
bge's recall@5. The term's gain split between the head and the trunk state: a head on the
untouched trunk (`w0e3`) reaches about 0.025 at k5, a head on the term-trained trunk (`w25e3`)
about 0.055, the model's own trained query 0.059. The trunk's share cost 0.30 nats at pilot
shape (arm PI).

## Method

`scripts/probe_query_head.py` (eval only, no module change) loads a checkpoint and runs the 8,715
`openqa` store questions (`data/index/openqa`) through the model with no evidence attached,
capturing the IR expert's normed input at loop 1 (the state its search head sees, 256-d at
micro) at the last prompt token; `--pool mean` averages it over the prompt's real tokens (chat
control tokens dropped) and `--pool meanlast` concatenates mean and last (512-d). On these states
it fits, under 5-fold cross validation, a ridge map (alphas 0.01, 0.1, 1, 10) and a two-layer MLP
(hidden 1024, GELU, cosine loss) onto bge's question vector, and scores recall@5 and recall@20
of the gold chunk over the 274,212-chunk store for each held-out fold. The same scoring runs on
the model's own query (its trained search head, in the selector's space against
`key_adapter(bge key)`), on the training folds' mean bge vector (a floor that knows nothing of
the question) and on bge itself (the ceiling). Capture took 22 to 43 s per read, every fit under
10 s. Checkpoints: the finals of the three-epoch InfoNCE pair
([infonce_micro.md](infonce_micro.md)), `w0e3` (term off) and `w25e3` (term at 0.25, every
loop, trunk gradient). Outputs `ckpts/inject/probe_{w0e3,w25e3}{,_b16,_mean,_meanlast}.{json,log}`.

The pooled reads and `w25e3` last ran at batch 16; `w0e3` last ran at batch 64 only. `w25e3`
last at batch 64 and 16 agree to 0.001 (padding changes the reduction order, nothing else), so
the batch does not matter at this precision.

Cosine to bge, also in the logs, is not the column to read: the mean-vector floor sits at 0.60,
and the MLP's cosine falls under it while its recall rises (the cosine loss fits the shared
direction every question has, which ranks nothing).

## Reads

Recall of the gold chunk at loop 1, held-out folds, 8,715 questions over 274,212 chunks.

`w0e3` (no contrastive term):

| predictor | last @5 | last @20 | mean @5 | mean @20 | meanlast @5 | meanlast @20 |
|---|---|---|---|---|---|---|
| model query | 0.000 | 0.002 | 0.000 | 0.002 | 0.000 | 0.002 |
| ridge 0.01 | 0.006 | 0.017 | 0.009 | 0.025 | 0.014 | 0.038 |
| ridge 0.1 | 0.005 | 0.016 | 0.009 | 0.025 | 0.014 | 0.039 |
| ridge 1 | 0.005 | 0.014 | 0.008 | 0.024 | 0.014 | 0.036 |
| ridge 10 | 0.004 | 0.011 | 0.007 | 0.018 | 0.010 | 0.028 |
| mlp | 0.007 | 0.021 | 0.022 | 0.051 | 0.025 | 0.055 |
| mean bge (floor) | 0.000 | 0.001 | 0.000 | 0.001 | 0.000 | 0.001 |
| bge (ceiling) | 0.829 | 0.897 | 0.829 | 0.897 | 0.829 | 0.897 |

`w25e3` (term at 0.25, trunk gradient, every loop):

| predictor | last @5 | last @20 | mean @5 | mean @20 | meanlast @5 | meanlast @20 |
|---|---|---|---|---|---|---|
| model query | 0.059 | 0.133 | 0.059 | 0.133 | 0.059 | 0.133 |
| ridge 0.01 | 0.032 | 0.080 | 0.018 | 0.047 | 0.043 | 0.109 |
| ridge 0.1 | 0.032 | 0.080 | 0.018 | 0.047 | 0.043 | 0.108 |
| ridge 1 | 0.031 | 0.080 | 0.017 | 0.047 | 0.043 | 0.107 |
| ridge 10 | 0.028 | 0.075 | 0.015 | 0.041 | 0.040 | 0.099 |
| mlp | 0.040 | 0.098 | 0.037 | 0.082 | 0.055 | 0.122 |
| mean bge (floor) | 0.000 | 0.001 | 0.000 | 0.001 | 0.000 | 0.001 |
| bge (ceiling) | 0.829 | 0.897 | 0.829 | 0.897 | 0.829 | 0.897 |

## Reading

- No head on the frozen state, at any position or pooled, gets past about 7% of bge at k5 (best
  0.055 / 0.122, `w25e3` meanlast MLP; 0.025 / 0.055 on `w0e3`). The question is not represented
  in the trunk's states at this size in a form a small map reaches. The heads do learn something
  (every one is above the floor of 0.000 / 0.001), so the state is not empty, only far from a
  retriever's.
- On `w0e3` pooling helps the MLP threefold (0.007 to 0.022 at k5) and concatenating adds a
  little (0.025): the last token alone carries less of the question than the prompt's mean.
- On `w25e3` the best head (0.055 / 0.122) nearly matches the model's own trained query
  (0.059 / 0.133), and the last-token heads (0.040 / 0.098) sit below it. The term moved the
  trunk's last-token state toward search (last-token MLP 0.007 on `w0e3` against 0.040 here),
  and pooled the gap stays: the best head reaches 0.025 on `w0e3` against 0.055 on `w25e3`. The
  term's gain therefore split between the head and the trunk state (a head on the untouched
  trunk reaches about 0.025 at k5, on the term-trained trunk about 0.055, the model's own query
  0.059), and the trunk's share cost 0.07 nats of held-out language at micro and 0.30 at pilot
  shape (arm PI).
- Ridge saturates at the low alphas (0.01 and 0.1 equal); the MLP is ahead on every read, by 0.001
  to 0.019 at k5, so a nonlinear head is worth a little, not a factor.

## Consequence

- A "train only the search head" arm reads what the head alone gets from an untouched trunk,
  about 3% of bge at k5 at micro scale, but it costs no language, since its gradient cannot
  reach the trunk. One such arm runs: `bash
  ckpts/inject/launch_infonce_head.sh` (`config_micro_infonce_head_e3.yaml`, the `w25e3` recipe
  with `evidence_contrastive_loops: [0]` and `evidence_contrastive_trunk_grad: 0.0`; run
  directory `ckpts/evidence_infonce_head_e3`, log `infonce_head_e3.log`, then `eval_store.py
  recall --loop 1` into `infonce_head_e3_store_recall.{json,log}`), launched and read 2026-10-21. Bar: held-out language within noise of `w0e3` (4.38) and recall above `w0e3`; the
  probe predicts about 0.02 to 0.03 at k5 (the `w0e3` heads), not the 0.055 a head reaches on
  the term-trained trunk. Its result is in
  [infonce_micro.md](infonce_micro.md). The head-only arm confirmed the cap from below:
  recall@5 0.003 at zero language cost, at the probe's ridge row on the last token (0.004 to
  0.006), not its pooled MLP ([infonce_micro.md](infonce_micro.md)).
- A separate search tower that reads the trunk's detached states is capped the same way, since
  it sees the same states the probe's heads see. A working tower would be a small bge-style
  encoder over the prompt text, a retriever beside the model rather than the trunk searching.
  Decided 2026-10-21 (the user): the tower is held, bge keys the store and embeds the question
  at serving, A7's recall half stays untested, and the model's own retrieval is revisited at the real run's
  size (the two keys stay as the knob).
- Not tested: a larger model (the cap may lift with width), later loops (loop 1 is the pass the
  store read uses), and heads wider than one hidden layer.
