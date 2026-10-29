# The model's own query against the store: options without a language cost (2026-10-21)

A literature round on one question, with each option judged against this codebase. Nothing here
is decided; the arms at the end are proposals. Papers are cited by arXiv number.

## The question

How can a small language model (35M to 99M parameters, trained from scratch) learn to retrieve
from a store with its own query, without paying in language quality?

The gate that failed is A4: the model's query at recall@5 0.004 over the 274,212-chunk `openqa`
store against 0.829 for bge's own question embedding. The fix tried, in-batch InfoNCE on
Co-LMLM's recipe ([infonce_micro.md](measurements/infonce_micro.md)), moved recall@5 from 0.000
to 0.059 and recall@20 from 0.002 to 0.133 over three epochs (104M tokens), at 0.07 nats of
held-out answer CE (4.45 against 4.38), a gap that widened with dose (0.05 at 20M, 0.08 at 100M).

## The mechanism as built

The store is keyed by an external embedder, bge-small-en-v1.5 (`modules/data/store.py`):
`keys.npy` holds one unit 384-d key per chunk, and `query_keys_bge.npy` holds bge's embedding of
each eval question with the instruction prefix "Represent this sentence for searching relevant
passages: " (`scripts/build_store.py`). The model's query lives in the IR expert
(`modules/model/experts.py`, `modules/model/information_retrieval.py`): the expert's input (the
residual at the top of the loop) goes through the expert's own norm, `down_proj` (256 to 384 at
micro width, so a rank-256 linear map; 512 to 384 at pilot width), plus `loop_query_bias` (a
zero-init linear map of a sinusoidal loop encoding), then unit normalization. External chunk keys
pass through `key_adapter`, a square orthogonal-init 384 x 384 map, are normalized, and score
against the query in one softmax with the learned 256-entry table, masked to the token's own
document's visible chunks. The selection loss (BCE against the gold flags, over the 4 to 20
visible chunks) trains this read and works within a document (chunk AUROC 0.78 at pilot shape).
The same `down_proj`, bias and adapter also produce the IR read that reaches the residual through
`up_proj` and `direct_gate`, so they are not a pure retrieval head: changing them changes what
the IR expert adds to the residual.

The contrastive term (`evidence_contrastive_loss`, `LoopMixtureOfExperts.evidence_contrastive_term`,
gated in `pretrain.train_step`) rescores the stashed unit query at every answer-start position
against every adapted chunk key in the batch (about 250 at batch 32 x 1024; the held-out loss
starts at ln 250, about 5.5), positives the position's own document's gold chunks, temperature
0.07, symmetric, averaged over every (loop, IR expert). Its gradient reaches the trunk through
the expert's input. Two details matter for the options below. First, `eval_store.py recall`
extracts the query at loop 1 with `evidence=None` (retrieval happens before any evidence exists),
while the term trains loops 1 to 3, and loops 2 and 3 have already read the evidence through the
key/value reader; recall at loops 2 and 3 stays at 0.005 and 0.004 at k 20, so two thirds of
the term trains a query the eval never uses, one that can lean on the gold chunk it has already
read. Second, the store keys are fixed bge vectors, not the model's own states as in Co-LMLM, so
the model's query has to land in bge's geometry. That is the asymmetric setup of query-encoder
distillation (a new query encoder against a fixed document index), not the symmetric dual-encoder
setup in-batch InfoNCE was designed for.

## What the literature says, by direction

1. **Stop the gradient at the query head.** RA-DIT (2310.01352) and REPLUG (2301.12652) train
   only the retriever's query encoder against a frozen LM (LM-supervised retrieval, a KL to the
   LM's answer likelihood per chunk); REPLUG's tuned retriever adds 6.3% on GPT-3 language
   modelling. Atlas (2208.03299, Table 4) finds query-side-only fine-tuning at 45.7 against 44.1
   for full fine-tuning at 64 shots and 51.0 against 51.7 at 1024 shots, and a frozen retriever
   worse (41.1, 48.3). "One Model Is Enough" (2603.08429) is the closest relative: a 25M head
   (input linear, two transformer layers over the prompt's tokens, mean pool) on the cached,
   untouched hidden states of Qwen3-8B, trained to land in Qwen3-Embedding-8B's space, reaches
   R@10 0.607 against the teacher's 0.637. All of these sit on large pretrained trunks; none
   says whether a linear head on a 35M from-scratch trunk can retrieve. Raw LM states are poor
   embeddings without a trained map (SBERT 1908.10084, BERT-flow 2011.05864), but here the map is
   trained.
2. **Distill bge instead of contrasting from scratch.** Query-encoder distillation by embedding
   alignment (2306.11550): a student query encoder regressed onto the teacher's query embeddings
   (Euclidean loss), the document index untouched; a 2-layer BERT student keeps 92.5% of the
   teacher on BEIR, a 4-layer one 96.2%. EmbedDistill (2301.12005): matching the teacher's
   embedding geometry beats score-only distillation, 1/10th-size students keep 95 to 97% on
   MSMARCO. In 2603.08429's loss ablation on frozen states, alignment alone gives R@10 0.567,
   contrastive alone 0.498, rank distillation alone collapses (0.001), alignment plus contrastive
   0.582, all three 0.595. Joint generation plus embedding: GritLM (2402.09906) at 7B reads 66.8
   MTEB and 55.5 generative average joint, against 66.8 embedding-only and 55.2 generative-only,
   "no performance loss"; its loss-ratio ablation (2.4 to 6.0) moves both by under 1 point; but
   it finetunes a pretrained Mistral 7B with a separate bidirectional, mean-pooled embedding
   mode, so it says little about a 35M trunk sharing one causal state. OneGen (2409.05152)
   reports generation preserved with contrastive retrieval tokens on a 7B finetune. RepLLaMA
   (2310.08319), E5-Mistral (2401.00368), LLM2Vec (2404.05961) and NV-Embed (2405.17428) turn
   LMs into embedders and give up generation; they do not measure the trade-off.
3. **A separate query tower.** RETRO (2112.04426) uses a frozen BERT for both keys and queries:
   the LM never produces a query and pays nothing. Dual encoders (DPR 2004.04906) and late
   interaction (ColBERT 2004.12832) keep the query encoder's parameters separate from any
   generator. 2603.08429 is a separate tower reading LM states, so it is the evidence that the
   tower can be small (25M on an 8B trunk; a two-layer MLP or one attention layer is the analogue
   at 35M).
4. **Weight, schedule, scaled gradient.** No paper found ablates the contrastive weight against
   the LM loss at small scale. Co-LMLM (2607.07707) uses weight 0.25, temperature 0.07, batch
   128 at 135M and 360M and reports no ablation of the weight, of a stopped gradient, or of the
   perplexity cost of the term (it reports perplexity against baselines only). GritLM's ratio
   ablation is the only weight curve, at 7B. A forward-identical, backward-scaled leak
   (`h.detach() + a * (h - h.detach())`) is the gradient-reversal layer's identity with a
   positive factor (1409.7495); PCGrad (2001.06782) projects a conflicting task gradient away
   but needs a second backward through the trunk.
5. **Negatives and store size.** DPR: in-batch negatives beat 7 fixed negatives clearly, and one
   BM25 hard negative per question adds more (NQ top-20 78.4 against BM25's 59.1 in the headline
   setting, batch 128). ANCE (2007.00808): in-batch negatives are "not representative of the
   irrelevant documents in testing"; negatives from an ANN index over the corpus fix it.
   Contriever (2112.09118): MoCo queue of 131,072; more negatives (2,048 to 131,072) helps,
   especially unsupervised; at equal 4,096 negatives MoCo and in-batch are close (30.1 against
   31.9 nDCG@10). RocketQA (2010.08191) cross-batch negatives, GradCache (2101.06983) large
   effective batches in small memory, XBM (1912.06798) a cross-batch memory for small batches.
   The usual objection to a memory bank, stale keys from an encoder that moved, does not apply
   here: the store keys are fixed bge vectors and only the 384 x 384 adapter moves, so a bank of
   raw keys stays exact and ANN-mined hard negatives never need a refresh.
6. **From reranker scale to retriever scale.** CL-DRD (2204.13679) distils a reranker into a
   dense retriever with a curriculum that moves from coarse to fine-grained candidate sets;
   RocketQA denoises hard negatives with a cross-encoder. Here the selection loss is the
   reranker-scale signal, the batch is the next scale, a fixed-key bank the next, bge-mined hard
   neighbours the last.
7. **Memory-augmented LMs.** LMLM (2505.15962) writes the query as text, an (entity, relation)
   pair, matched by an external MiniLM at cosine 0.6; no gradient reaches the trunk for
   retrieval beyond next-token CE on the query tokens. Co-LMLM (2607.07707) reads its own
   last-layer state at a `<FACT>` token as the query against keys that are its own states over
   the corpus (no external embedder; Faiss index up to 2.2B items), trained by the in-batch
   InfoNCE copied here. Memorizing Transformers (2203.08913) query a kNN memory with an
   attention layer's own keys, no retrieval loss, non-differentiable memory. Memory layers at
   scale (2412.09764) train product keys end to end by the LM loss alone: the analogue of this
   repo's learned IR table, which measured as storing nothing the trunk lacks. M+ (2502.00592)
   co-trains a retriever over hidden-state memory. Larimar (2403.11901) and MemoryLLM
   (2402.04624) read latent memory without store-scale retrieval. Pattern: where the query and
   key share the model's own space, no paper reports a separate retrieval loss cost; where keys
   come from an external embedder (RETRO, LMLM), the query is external too (text or a frozen
   encoder).

## Options

Language cost is held-out answer CE against the weight-0 twin (`w0e3`, 4.38 at 104M); recall is
`eval_store.py recall` at loop 1. Hours are build hours before the first launch.

| Option | What changes in this repo | Expected language cost | Expected recall effect | Evidence | Build h |
|---|---|---|---|---|---|
| 0. Loop-1 term | the contrastive term only at loop 1 (the read the eval uses); one key | same or lower than 0.07 (one third of the gradient) | likely up: trains the evidence-free query the eval reads | repo: loops 2 and 3 read 0.005 / 0.004 at k 20 | 0.5 |
| 1. Stop-grad head | recompute the term's query at the P answer-start rows from the stashed expert input, detached (`a = 0`), through the expert norm, `down_proj`, loop bias; adapter as now | near 0; only through the IR read's shared head (the read itself is small, external mass 0.04 to 0.12) | bounded by what the frozen-by-gradient trunk state carries; unknown at 35M | RA-DIT 2310.01352, REPLUG 2301.12652, Atlas 2208.03299 Table 4, 2603.08429 | 1 |
| 2a. Align to gold key | cosine of the query to `normalize(A.detach() k_gold)` at answer-start rows, with 1 | near 0 with 1; unknown without | likely above InfoNCE alone at small batch (no negatives needed, fixed targets cannot collapse) | 2306.11550, EmbedDistill 2301.12005, 2603.08429 (alignment 0.567 vs contrastive 0.498) | 1 |
| 2b. Distill bge question | a `{split}.qkey` sidecar (bge of each conversation's question, with bge's instruction), cosine of the query to `normalize(A.detach() e_q)`; covers rows with no gold (none, distractors) and bge's exact geometry; can add NQ and TriviaQA train questions to close the domain gap (HotpotQA 0.237 vs NQ 0.019 at k 20) | near 0 with 1 | ceiling bge 0.829 at k 5; the best single lever if the state carries the question | 2306.11550 (92.5% with a 2-layer student), 2603.08429, GritLM 2402.09906 (joint at no loss, 7B) | 4 |
| 3. Separate query tower | new module (2-layer MLP 256 to 1024 to 384, or one attention layer pooling the prompt's states) on detached states, used only for store retrieval in `eval_store.py` and the session's first retrieval; trained by 2b plus InfoNCE | exactly 0 | at least option 1's (nonlinear, may pool over the prompt) | RETRO 2112.04426, DPR, ColBERT 2004.12832, 2603.08429 (25M tower on frozen 8B, R@10 0.607 vs 0.637) | 5 |
| 4. Weight, schedule, leak | `evidence_contrastive_weight` 0.05 to 0.1; term only after a token mark; leak factor `a` in (0, 1) on the trunk gradient | between 0 and 0.07, presumably monotone in weight times `a` | presumably monotone the other way | no small-scale ablation exists; GritLM's ratio ablation at 7B; 1409.7495; PCGrad 2001.06782 | 0.5 |
| 5. Bank and hard negatives | a fixed bank of 16k to 64k raw keys sampled from `{split}.evkey` (fp16 memmap), adapted on the fly, as extra negatives (a `[P, 16k]` matmul); later 1 to 4 bge-nearest neighbours of each gold key mined offline | same as the base term (near 0 with 1) | targets the measured failure: in-batch loss 0.98 nats at 104M but 274k-scale recall 0.059 | DPR 2004.04906, ANCE 2007.00808, Contriever 2112.09118, RocketQA 2010.08191, XBM 1912.06798 | 2 (bank), +3 (mined) |
| 6. Candidate curriculum | grow the candidate set from visible chunks to batch to bank to mined neighbours over the run | as 5 | small over 5 | CL-DRD 2204.13679 | 2 on top of 5 |
| 7a. External query (RETRO) | retrieve with bge on the prompt text; drop the model query from retrieval | 0 | 0.829 by definition | RETRO 2112.04426 | 0 |
| 7b. Text query (LMLM) | the model writes a query in tokens, bge embeds it | CE on query tokens only | depends on data annotation | LMLM 2505.15962 | 20+ |

Options 0, 1, 2a, 4 and the bank half of 5 are config keys or a few lines next to
`evidence_contrastive_term`; they compose. Option 3 is the only one with a strictly zero language
cost, and it is also architecturally natural: the store lookup already happens before the
forward (`eval_store.py` reads with `evidence=None`), so the lookup query need not be the
in-forward selector's. Option 7a gives up the A4 gate rather than passing it; it is listed
because RETRO shows the LM does not need its own query to gain from retrieval.

## Recommended order

First, before any training arm, an offline probe (about 3 hours of build, minutes to run) that
decides whether options 1 to 3 can work at all: on the existing `w0e3` and `w25e3` finals,
capture the IR expert's normed input at the last prompt token at loop 1 (a pre-hook beside
`QueryReadout` in `eval_store.py`) for the 8,715 store questions, fit a ridge map and a two-layer
MLP onto `query_keys_bge.npy` with 5-fold cross-validation over questions, and score recall@5 and
@20 of the mapped query directly against `keys.npy` (bge space, no adapter). This is the
ceiling of a head on that state, in domain. If the linear probe reads near 0.06 (the contrastive
arm) the state carries little and no head option is worth an arm; if it reads well above, the
arms below are worth running.

The arms run on the three-epoch QA recipe (`config_micro_infonce_e3.yaml`, 104M tokens, about 41
minutes) so `w0e3` and `w25e3` are the ready-made baselines; at 300M tokens the QA slice would
be 8.6 epochs and overfit, so 300M only makes sense on a merged corpus. Every arm reads
held-out answer CE against `w0e3` (cost), `eval_store.py recall --store data/index/openqa
--buffer 4 --loop 1` against `w25e3` (0.059 / 0.133) and bge, and `selection` on held out.

1. **Head-only InfoNCE, loop 1** (options 0 + 1, about 1.5 hours of build): new keys
   `evidence_contrastive_loops: [0]` and `evidence_contrastive_trunk_grad: 0.0`, the term's query
   recomputed at the P rows from the detached expert input; weight 0.25. Read: CE within 0.01 of
   `w0e3` (cost gone) and recall@5 against 0.059 (does a head on an unshaped trunk retrieve).
2. **Head-only distillation** (option 2b plus arm 1's detach, about 4 hours): the `.qkey`
   sidecar for `evidence_nomany_train`, `evidence_query_distill_weight: 1.0` on
   `1 - cos(q, normalize(A.detach() e_q))` at every answer-start row, InfoNCE kept at 0.25 on
   loop 1. If the probe says the state is linear-readable, this is the arm that should approach
   it. Option 2a (gold-key target, no sidecar) is the one-hour fallback if the sidecar slips.
3. **Fixed-key bank** (option 5, about 2 hours, on the better of arms 1 and 2):
   `evidence_contrastive_bank: 16384` raw keys resampled from `evidence_nomany_train.evkey` each
   step, excluding the batch's gold chunks. Read: recall@20 against the parent arm at equal
   language cost; the in-batch loss already reads 0.98, so if recall moves here the negatives,
   not the query's capacity, were the limit.

If arm 1 or 2 holds CE but recall stays near the probe's ceiling and well under bge, option 3
(a separate tower over the prompt's states) is the next build, since it can pool over positions
a single-position head cannot.

## What no option fixes

Every head option, linear or tower, reads the trunk's state; distillation copies bge's geometry
and cannot pass bge (0.829 at k 5) without an LM-likelihood signal (REPLUG, Atlas). If the
hidden state at the answer-start position, at loop 1 with no evidence attached, does not carry a
question-shaped representation (the entity and relation the question asks about), no head
recovers it, and the only routes left are a trunk gradient (the 0.07-nat cost, or some of it
under a leak factor) or a tower that pools over the prompt's tokens, which at the limit is a
small encoder trained from scratch beside the LM: a smaller bge, not the model's own query. The
probe above measures exactly this before any arm is spent.

## References added this round

RA-DIT 2310.01352; REPLUG 2301.12652; Atlas 2208.03299; One Model Is Enough 2603.08429; query
encoder distillation 2306.11550; EmbedDistill 2301.12005; GritLM 2402.09906; OneGen 2409.05152;
RepLLaMA 2310.08319; E5-Mistral 2401.00368; LLM2Vec 2404.05961; NV-Embed 2405.17428; RETRO
2112.04426; DPR 2004.04906; ColBERT 2004.12832; ANCE 2007.00808; Contriever 2112.09118; RocketQA
2010.08191; GradCache 2101.06983; XBM 1912.06798; CL-DRD 2204.13679; gradient reversal
1409.7495; PCGrad 2001.06782; SBERT 1908.10084; BERT-flow 2011.05864; Memorizing Transformers
2203.08913; Memory layers at scale 2412.09764; M+ 2502.00592; Larimar 2403.11901; MemoryLLM
2402.04624. Already cited: LMLM 2505.15962; Co-LMLM 2607.07707.
