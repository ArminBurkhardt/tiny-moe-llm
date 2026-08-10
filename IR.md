# IR.md

I read `CONCLUSION.md` (it's at [ckpts/trained/temp-train/CONCLUSION.md](ckpts/trained/temp-train/CONCLUSION.md), gitignored), PLAN.md Step 15, and the IR/loop/attention code. Here's the brainstorm.

## Four facts from the run that constrain everything

1. **There is no retriever to preserve.** Cosine logits ∈ [−1,1] at `temperature=1.0` over 8192 entries → entropy near maximal → retrieved ≈ `mean(y_values)`. The table stores nothing addressable. That's liberating: replacing its contents, its size, or `ir_dim` costs you nothing, because nothing is in there. Don't design around "migrating" the existing table.
2. **The router likes the slot anyway** (7–9% vs 5.7% uniform). So the *pathway* — per-token side channel consumed via cross-attention — is wanted even when the side channel is a constant. That's mild evidence the plumbing is right, and a warning that it may be being used as a bias term. Ablation settles it.
3. **Loop 3 buys ~0 nats today** (per-loop CE 3.109 / 2.969 / 2.969). ">3 loops" is not a config change; later loops need a *reason* to differ. RAG supplies exactly that reason — but only if retrieval is re-executed per loop with a query that actually moves.
4. **Depth override already works mechanically.** `max_enc_loops=64` ([moe.py:183](modules/model/moe.py#L183)), sinusoidal loop encoding, `loop_scale` clamps to the last entry. `forward(n_loops=8)` runs today. Nothing structural blocks deep loops; only training does.

## The reframe

Two decisions get conflated and shouldn't be:

- **(a) Where memory *content* comes from** — learned parameters (today) vs. external swappable data (RAG).
- **(b) How the read is *refined* across loops** — one-shot vs. iterative/multi-hop.

You need (a) for RAG to exist at all. (b) is the only thing that makes >3 loops pay. PLAN Step 15's items 1–6 are all (a)-flavoured and all still parametric — a bigger, sharper learned table is a *knowledge-capacity* project, not a RAG project. A parametric table can't be updated without training; the entire point of RAG is that the corpus is data.

## (a) Where evidence enters — three ports

**Option A — IR module becomes a reader over externally supplied memory.** Add `memory=(K_ext, V_ext)` to [information_retrieval.py](modules/model/information_retrieval.py); concatenate: `K = [z_keys ; K_ext]`, `V = [y_values ; V_ext]`. Two properties fall out for free:

- **Graceful degradation** — no corpus attached → bit-identical to today. One checkpoint serves both modes.
- **Softmax mass on external vs. parametric entries is a groundedness signal.** "I retrieved nothing relevant" becomes *measurable* rather than guessed. That is directly the feature Step 12b-i's probe is missing, and it makes abstention grounded instead of memorized. This is the single biggest cross-workstream win available.

Limitation: one 128-d vector per entry is a topic vector, not a passage. Fine for "which fact", weak for "copy this span".

**Option B — evidence as token-level KV through the CrossAttention expert.** `other` in [transformer.py:341](modules/model/transformer.py#L341) is already a per-call injection port that is **re-read at every loop** — structurally already "re-consult the evidence at each refinement step". Swap `_moe_ple(input_ids)` for embedded retrieved chunks and you have RETRO/FiD-lite. Strongest for extractive/grounded answering, which is what this model can actually learn at 332M.

One concrete blocker: [attention.py:109-116](modules/model/attention.py#L109-L116) passes the *same* `cu_seqlens` for q and k, so `o_len` must equal `S`. An evidence set of length M needs `cu_seqlens_k` / `max_seqlen_k` plumbed (flash supports it natively) and `causal=False`. Plus a decision on RoPE for evidence keys — [gemma4.py:79-81](modules/model/gemma4.py#L79-L81) currently rotates q and k with the same `cos/sin`, which is meaningless for evidence positions. Probably: no RoPE on evidence keys, or a separate short position basis per chunk.

**Option C — both, with a division of labour. This is what I'd build.** IR expert = *selector* (which of the k candidates matter; also long-tail entity memory), CrossAttention expert = *reader* over the actual evidence tokens. That maps the two experts you already have onto the standard retriever/reader decomposition, and both already sit in the pool.

**Caveat on C:** don't let the router decide whether to consult evidence. The run shows the aux loss pinned at its balanced value from step 0 and mean routed weight flat across all 35 experts — the router never specialized. Making "does this token need a fact?" depend on the weakest measured component is a bad bet. When a corpus is attached, make the evidence read **always-on**, alongside `shared_mlp`/`shared_attn`, and let the *content* be gated by retrieval scores instead.

## (b) Getting external keys into the query space

The query is `down_proj(RMSNorm(h)) ∈ R^128`. Keys must live there.

- **B1 self-encoding** — encode chunks with the model itself, pool, `down_proj`. No second model, spaces match by construction, but a 128-d pooled state from a model never trained for retrieval is a weak retriever, and any trunk change forces a re-index.
- **B2 external embedder + adapter** (bge-small / e5-small, 33M, 384-d) — **the pragmatic choice.** Map the *query into the embedder's space* so the ANN index is standard and reusable, project retrieved vectors into IR space for the read. Removes retriever quality from the critical path entirely; chunk embeddings are precomputed, so inference cost is one small forward per query.
- **B3 contrastive warmup** — regardless of B1/B2, train the query head with InfoNCE on (context → the chunk containing the continuation) pairs mined from `phase1.bin`. This is what makes retrieval *work* rather than hoping CE discovers it. Freeze the document side and train only the query side, which also sidesteps index staleness.

## Granularity: where the cost actually lands

Per-token ANN over a real corpus during generation is infeasible; per-sequence-at-prefill kills multi-hop. The design that works:

> **ANN retrieves k≈32–64 candidates per sequence per loop. The IR module's soft, differentiable read over those candidates stays per-token.**

End-to-end differentiable, ANN cost is loops × sequences (not tokens), and it's exactly the two-stage retrieve/read structure the module already implements.

KV-cache note: if the evidence set mutates mid-generation, the IR/cross-attn cache for past tokens goes stale. Make evidence **append-only** — new retrievals extend the KV set, never rewrite it. Cache-friendly, and it hands you the accumulating buffer that multi-hop needs anyway.

## Making >3 loops actually pay

Cheapest first:

1. **Loop-conditioned query.** Zero-init per-loop bias on the IR query, mirroring `loop_router_bias` exactly (sinusoidal in absolute loop index, clamped). Guarantees loop 3 doesn't re-issue loop 1's query. No-op at init → checkpoint loads unchanged.
2. **Append-only evidence buffer.** Loop L reads the union of retrievals from loops 1..L. This *is* the "refinement and reprocessing" the goal names, made concrete, and it makes depth monotonically informative.
3. **Novelty pressure.** Mask already-retrieved ids from the next loop's ANN result (or an MMR term). Without it, three loops fetch the same top-1 three times.
4. **Depth curriculum.** `sample_n_loops` / `loop_ce_weights_for` in [pretrain.py:239-257](scripts/pretrain.py#L239-L257) already truncate-and-rescale so the deepest loop run carries weight 1.0. Extend sampling *upward* (max 6–8) on retrieval-augmented batches, with a **back-loaded** weight vector for those batches (e.g. `[0,0,0.1,0.2,0.3,1.0]`) while plain-LM batches keep `[0.2,0.3,1.0]`. This resolves PLAN 12b-iv's "loop 1 reads out well vs. later loops do a lot" tension *per-task* instead of globally — the loop-index conditioning is exactly the capacity that lets the model behave differently at depth.
5. **A real job for the halt head.** CONCLUSION's diagnosis is that halting gates the loop's *output*, not its *compute*, so λ was never a control knob. Under RAG it can become "stop retrieving and stop looping" — an actual early exit with an actual compute payoff, and "keep looping while new evidence is still arriving" is a well-posed, learnable criterion (buffer stopped growing / delta norm below τ). That's the first version of that head with genuine authority.
6. **Retrieval-utility diagnostic:** per-loop CE-with-evidence minus CE-with-evidence-zeroed. At minimum it tells you whether depth is buying grounding or just churn.

## Training recipe

**Stage 0 — measure, no training, hours.** Non-negotiable. Retrieval entropy today (expect ≈ ln 8192 = 9.01 nats). IR ablation: zero the expert's output, ΔCE on held-out. Query drift: `cos(down_proj(h_loop1), down_proj(h_loop3))` — if ≈1, item 1 above is mandatory rather than optional.

**Stage 1 — sharpen.** Learned log-temperature, plus top-k read. **The trap PLAN Step 15 item 1 understates:** `y_values` have only ever been used as a near-uniform mixture, so dropping temperature at inference reads out vectors that were never individually trained. You get a loss spike, not a signal. Temperature must be *annealed during a finetune* (1.0 → ~0.05), and the "free look at inference" is not actually informative.

**Stage 2 — oracle evidence. This is the key trick and the main training spend.** Before any index exists, hand the model evidence you already know is relevant: the gold passage for QA, a held-out span from the same document for web text. Three conditions mixed roughly evenly:

| condition | what it teaches |
|---|---|
| gold evidence | read the buffer (large, immediate CE gradient) |
| distractor evidence | don't blindly trust the buffer |
| no evidence | **abstain — grounded in retrieval, not memorized as a string** |

That third row is why this stage is worth doing even if the RAG project stalls: it's a principled fix for the 78.4% false-abstention collapse that doesn't depend on rebalancing SQuAD v2 ratios.

**Stage 3 — align the retriever** (InfoNCE, query side only, document encoder frozen; one hard-negative mining round).

**Stage 4 — end-to-end with the real ANN index.**

**Stage 5 — depth curriculum** (items 1–4 above). Only here does ">3 loops" get trained.

**Stage 6 — RAG SFT.** [chat.py](modules/data/chat.py) has system/user/assistant only, so evidence needs either a new segment or the system turn.

## Gates, including the one that decides whether this was worth it

- **G1**: IR ablation ΔCE > ~0.02 nats. If ~0 today, the expert is a bias term — fix that before building an index.
- **G2**: post-anneal, entropy well below ln N *and* held-out CE not regressed.
- **G3**: gold-vs-no-evidence CE gap ≥ ~0.3 nats on the answer span, and abstention rate under no-evidence ≫ under gold.
- **G4**: recall@k beats BM25. If it doesn't, take B2 and stop training the retriever.
- **G5**: EM/F1 on NQ-open / TriviaQA / **PopQA** (long-tail is where this should shine) with corpus attached vs. not. Then the depth ablation: EM at `n_loops` = 2, 3, 4, 6, 8 with corpus attached. **Flat past 3 → the depth story is dead, ship 3, don't rationalize it.**
- **G6 — the honest baseline: put the retrieved passages in the prompt as text.** If side-channel RAG only matches that, the architecture claim is unproven. The claim worth aiming at is the one in-context evidence *can't* make: **attach far more evidence than 4096 tokens can hold, at a cost that doesn't grow quadratically with evidence.** That's the actual reason to do this in the architecture.

## Risks worth naming now

- **128 dims is thin for passage content.** Since the table holds no information, raising `ir_dim` to 256 for the RAG variant is nearly free — do it at the same time as anything else that re-inits the table.
- **332M / 16B tokens is a weak reader.** But grounded *extraction* is the easiest thing to teach at this scale — copying beats recalling. That's a real argument that RAG is the right direction for this specific model: it converts a knowledge problem it cannot solve into a copying problem it can.
- **Retriever/reader co-training staleness** — mitigated by freezing the document side throughout.
- **Sequencing vs. Step 12b:** Stage 2's no-evidence condition overlaps the abstention repair. Doing 12b-iii's rebalance *first* and Stage 2 *second* risks training the same behaviour twice with different mechanisms. Worth deciding deliberately which one owns abstention — my vote is retrieval-grounded (Stage 2), with 12b-iii reduced to just fixing the SQuAD-v2 conversation-count imbalance.

Want me to write this up as an expanded Step 15 (or a new Step 18) in [PLAN.md](PLAN.md), or start with the Stage 0 diagnostics — the entropy/ablation/query-drift measurements are a single script against the existing SFT checkpoint and would settle several of the assumptions above before any design gets committed to?
