import math

import torch
from torch import nn
import torch.nn.functional as F


class RetrievalEntropyTracking():
    """per loop EMA of the retrieval softmax entropy, normalized as ``E / ln(width)``.

    ``width`` is the number of entries the softmax the module actually takes spans: the whole table
    on the exact path, and ``read_top_k`` once two stage scoring is on. Those are different units
    and the trainer's log line says which one it is printing, because a top-32 read cannot exceed
    ``ln 32`` no matter how flat it is -- normalizing it by ``ln 65536`` would report a sharp table
    that is in fact perfectly uniform over everything it looked at.

    Same quantity ``scripts/eval_stage0.py`` reports as its diagnostic 1. 1.0 means a token's read
    touches every candidate uniformly, i.e. the read carries no selection; both 16B checkpoints
    measured 0.995 of the full-table maximum, which is what Phase 3 exists to move.

    Two guards, for the same reasons ``_ExpertTracking`` in moe.py has them: the reduction only runs
    on every ``sample_interval``-th forward, because it is a full pass over the [T, width] weight
    matrix and is pure instrumentation; and it is capped at the number of updates one real forward
    makes, so an activation checkpoint recompute cannot count the same loop twice. Values
    stay on device -- ``get_stats()`` is the only host sync, and the trainer calls it at its log
    cadence.
    """

    def __init__(self, num_entries: int, n_loops: int):
        self._n_loops = n_loops
        self.num_entries = num_entries
        self._ln_entries = math.log(num_entries)
        self.entropy = None        # [n_loops], lazily placed on device on the first update
        self._warm = False
        # shorter window than _ExpertTracking's 256: sampling every 8th forward already smooths
        # this, and an anneal that sharpens the table is something you want to see move.
        self.window = 64
        self.sample_interval = 8
        # rows per fp32 chunk. the bf16 weights are already [B*S, width]; upcasting that whole
        # tensor at once is another ~0.5GB at B*S=16384/width=8192, which is real peak memory.
        # 1024 measured fastest at that shape (1.35ms vs 1.83 at 2048 and 1.90 at 512): small enough
        # that both live tensors stay in cache-friendly territory, large enough not to be launch bound
        self.chunk_rows = 1024
        self._expected_updates = None
        self._seen_updates = 0
        self._forward_counter = 0
        self._active = True

    def begin_forward(self, expected_updates: int):
        self._expected_updates = expected_updates
        self._seen_updates = 0
        self._active = (self._forward_counter % self.sample_interval) == 0
        self._forward_counter += 1

    @torch.no_grad()
    def update(self, weights: torch.Tensor, loop_idx: int = 0):
        """fold one loop's retrieval weights [.., width] into that loop's EMA."""
        if not self._active:
            return  # throttled forward, skip the reduction
        if self._expected_updates is not None:
            if self._seen_updates >= self._expected_updates:
                return  # checkpoint recompute pass, already counted
            self._seen_updates += 1

        w = weights.detach().reshape(-1, weights.shape[-1])
        total = torch.zeros((), device=w.device, dtype=torch.float32)
        for start in range(0, w.shape[0], self.chunk_rows):
            # torch.special.entr is -x*log(x) in ONE kernel (and defines entr(0) = 0, so no clamp
            # is needed). The written-out -(c * c.clamp_min().log()).sum() form costs three extra
            # full-size intermediates and measured 3x slower at the real [16384, 8192] shape.
            total += torch.special.entr(w[start:start + self.chunk_rows].float()).sum()
        frac = total / max(w.shape[0], 1) / self._ln_entries

        if self.entropy is None or self.entropy.device != w.device:
            self.entropy = torch.zeros(self._n_loops, device=w.device)
            self._warm = False
        # loop indices past the trained count fold into the last slot, matching how loop_scale and
        # the router bias treat an inference time depth override
        idx = min(int(loop_idx), self._n_loops - 1)
        if not self._warm:
            # seed every slot, so the first log line reads a real entropy rather than an EMA that
            # is still climbing out of zero
            self.entropy.fill_(0.0)
            self.entropy += frac
            self._warm = True
        else:
            decay = 1.0 - 1.0 / self.window
            self.entropy[idx] = self.entropy[idx] * decay + frac * (1.0 - decay)

    def get_stats(self):
        """the per loop ``E / ln(width)`` fractions, as a plain list (one host sync)."""
        if self.entropy is None:
            return []
        return self.entropy.cpu().tolist()

    def reset_stats(self):
        if self.entropy is not None:
            self.entropy.zero_()
            self._warm = False


def is_rebuilt_ir_param(name: str) -> bool:
    """True for a parameter the 65536 x 256 reshape rebuilds from scratch.

    One definition, two callers that must agree: ``scripts/migrate_ir_reshape.py`` decides what to
    re-initialize, and ``scripts/sft.py`` decides what gets the fresh-parameter learning rate. If
    those two lists ever disagreed, some tensor would be reset and then trained at a converged
    trunk's LR, or carried over and trained at a from-scratch LR -- both silent.

    The IR expert's own norm and attention are NOT in here: they were trained with the model and
    only their input changes.
    """
    if ".ir_module." in name:
        return True
    # down_proj / up_proj live on the expert rather than the module, and their widths moved with
    # ir_dim. Scoped to `experts.` so the dense decoder's identically-named MLP projections and the
    # routed experts' grouped GEMMs are untouched.
    return "experts." in name and (name.endswith("down_proj.weight") or name.endswith("up_proj.weight"))


def _group_starts(group_ids: torch.Tensor, num_groups: int) -> torch.Tensor:
    """exclusive prefix sum of the per group counts of an already sorted ``group_ids``.

    Deliberately not ``torch.bincount``: bincount's output length is ``max(minlength, max(x)+1)``,
    so CUDA has to read the maximum back to the host to size the result. This path runs every
    forward and must not sync (see CLAUDE.md).
    """
    counts = torch.zeros(num_groups, dtype=torch.long, device=group_ids.device)
    counts.scatter_add_(0, group_ids, torch.ones_like(group_ids))
    return torch.cumsum(counts, 0) - counts


@torch.no_grad()
def balanced_spherical_kmeans(
    keys: torch.Tensor,
    num_clusters: int,
    iters: int = 12,
    rounds: int = 8,
    centroids: torch.Tensor = None,
):
    """cluster L2-normalized keys into EXACTLY equal sized groups on the unit sphere.

    Spherical, not Euclidean: the module scores cosine similarity against normalized keys, so a
    centroid is the renormalized mean of its members and Lloyd's plain Euclidean update optimizes
    the wrong metric.

    Exactly equal sized rather than merely capped, because equal sizes are what let the forward
    path view the table as ``[num_clusters, capacity, dim]`` and score a probed cluster as a slice
    instead of a ragged gather. It also removes the failure the cap was there to prevent: a giant
    cluster makes the per token probe cost variable, and the entries in it that never make the
    exact stage are unreachable, i.e. untrainable.

    The assignment is a capacity constrained proposal round (each still-unassigned entry proposes
    to its best cluster that still has room; each cluster accepts its best proposers up to its
    remaining room), which is the deferred-acceptance shape and needs no per entry Python loop.
    Whatever is left after ``rounds`` is packed into the remaining holes in similarity order.

    Args:
        keys: ``[num_entries, dim]``, any dtype; normalized internally in fp32.
        num_clusters: must divide ``num_entries``.
        iters: k-means iterations.
        rounds: proposal rounds per assignment.
        centroids: optional ``[num_clusters, dim]`` warm start, e.g. the previous refresh's, so a
            refresh perturbs the partition instead of rebuilding it from a fresh random draw.

    Returns:
        ``(members, centroids)`` -- ``members`` is ``[num_clusters, capacity]`` int64 holding each
        cluster's entry ids, ``centroids`` is ``[num_clusters, dim]`` fp32 and normalized.
    """
    n, dim = keys.shape
    assert n % num_clusters == 0, f"num_entries {n} must divide by num_clusters {num_clusters}"
    capacity = n // num_clusters
    z = F.normalize(keys.detach().float(), p=2, dim=-1)

    if centroids is None:
        pick = torch.randperm(n, device=z.device)[:num_clusters]
        c = z[pick].clone()
    else:
        c = F.normalize(centroids.detach().float(), p=2, dim=-1).clone()

    entry_ids = torch.arange(n, device=z.device)
    assign = torch.zeros(n, dtype=torch.long, device=z.device)
    for _ in range(iters):
        sim = z @ c.t()                                        # [n, num_clusters]
        assign = _capacity_assign(sim, capacity, rounds)
        # centroid = renormalized mean of its members
        c = torch.zeros_like(c).index_add_(0, assign, z)
        c = F.normalize(c, p=2, dim=-1)

    # members, grouped: sorting by cluster id makes each cluster's ids contiguous, and every count
    # is exactly `capacity`, so the reshape is exact rather than padded
    order = torch.argsort(assign, stable=True)
    members = entry_ids[order].view(num_clusters, capacity)
    return members, c


@torch.no_grad()
def _capacity_assign(sim: torch.Tensor, capacity: int, rounds: int) -> torch.Tensor:
    """assign every row of ``sim`` [n, c] to a column, with exactly ``capacity`` rows per column."""
    n, num_clusters = sim.shape
    device = sim.device
    assign = torch.full((n,), -1, dtype=torch.long, device=device)
    remaining = torch.full((num_clusters,), capacity, dtype=torch.long, device=device)

    for _ in range(rounds):
        free = assign < 0
        if not bool(free.any()):
            break
        # a full cluster stops receiving proposals; -inf keeps argmax away from it
        full_mask = (remaining <= 0).unsqueeze(0)
        proposals = sim.masked_fill(full_mask, float("-inf")).argmax(dim=1)   # [n]
        proposals = torch.where(free, proposals, torch.full_like(proposals, num_clusters))

        # rank each cluster's proposers by similarity and accept the top `remaining` of them.
        # sorting by (cluster, -sim) puts every cluster's proposers contiguous and in order, so a
        # within-group position beats a per cluster topk with a variable k.
        prop_sim = torch.where(free, sim.gather(1, proposals.clamp(max=num_clusters - 1).unsqueeze(1)).squeeze(1),
                               torch.full((n,), float("-inf"), device=device, dtype=sim.dtype))
        # sort by similarity first, then stably by cluster: the second sort preserves the first's
        # order inside each group, which is what makes "position < remaining" mean "best proposers"
        by_sim = torch.argsort(prop_sim, descending=True, stable=True)
        cl_sorted_idx = by_sim[torch.argsort(proposals[by_sim], stable=True)]
        cl_sorted = proposals[cl_sorted_idx]

        starts = _group_starts(cl_sorted, num_clusters + 1)
        pos = torch.arange(n, device=device) - starts[cl_sorted]
        accept = pos < remaining.clamp(min=0)[cl_sorted.clamp(max=num_clusters - 1)]
        accept = accept & (cl_sorted < num_clusters)

        accepted_cl = torch.where(accept, cl_sorted, torch.zeros_like(cl_sorted))
        # scatter only the accepted ones. rejected proposals all write to one trash row that is
        # sliced off again, rather than writing their old value back to a real row -- duplicate
        # indices in a scatter are order-undefined, and every rejected row would be a duplicate.
        ext = torch.cat([assign, assign.new_zeros(1)])
        rows = torch.where(accept, cl_sorted_idx, torch.full_like(cl_sorted_idx, n))
        ext.scatter_(0, rows, accepted_cl)
        assign = ext[:n]
        taken = torch.zeros(num_clusters, dtype=torch.long, device=device)
        taken.scatter_add_(0, accepted_cl, accept.long())
        remaining = remaining - taken

    # anything still unassigned goes into whatever holes are left, in cluster order. only reached
    # when the proposal rounds run out, which the recall diagnostic would show up as a bad refresh
    leftover = (assign < 0).nonzero(as_tuple=True)[0]
    if leftover.numel() > 0:
        holes = torch.repeat_interleave(
            torch.arange(num_clusters, device=device), remaining.clamp(min=0)
        )
        assign[leftover] = holes[:leftover.numel()]
    return assign


class InformationRetrievalModule(nn.Module):
    """learned key/value table read by a temperature-controlled softmax over cosine similarity.

    Exact path (``num_clusters == 0``):
        1. normalize: ``x_hat = x / ||x||``, ``z_hat = z / ||z||``
        2. similarity: ``s = (x_hat @ z_hat^T) / temperature``
        3. softmax over the WHOLE table, read ``y = w @ Y``
        4. project back with ``g``

    Two stage path (``num_clusters > 0``), which is what the 65536-entry table runs:
        1. score ``num_clusters`` centroids, keep the best ``probe_clusters``
        2. score those clusters' members exactly, keep the best ``read_top_k``
        3. softmax over those and read their values

    The two stage path exists because top-k sparsifies the *read* but not the *scoring*: a full
    65536x256 score is ~275 GFLOP per loop at 8192 tokens and materializes a [tokens, 65536]
    softmax, which is several GB of retained activation per loop. Probing 4 of 256 clusters costs
    ~2% of that. It is also deliberately the same retrieve-then-read structure the external corpus
    will use (docs/plans/NEXT.md's Option A), so there is one retrieval mechanism with two backing
    stores rather than two mechanisms.

    Differentiability is preserved through the selected entries only -- an entry that is never in a
    probed cluster's top-k gets no gradient, which is why ``entry_usage`` is tracked and dead
    entries are recycled at refresh.
    """

    def __init__(
        self,
        num_entries,
        latent_dim,
        output_dim,
        temperature=1.0,
        temperature_scale: float = 1.0,
        use_min_dist=False,
        residual=False,
        num_clusters: int = 0,
        probe_clusters: int = 4,
        read_top_k: int = 32,
        query_capacity_factor: float = 1.5,
        reservoir_size: int = 1024,
    ):
        """
        Args:
            num_entries: Number of trainable z and y pairs.
            latent_dim: Dimension of the input vector `x`.
            output_dim: Dimension of the returned vector y.
            temperature: initial retrieval temperature. Now the init of a LEARNED
                ``log_temperature`` rather than a constant, multiplied by an externally driven
                anneal scale (see ``set_temperature_scale``).
            temperature_scale: initial value of that anneal multiplier. It is a persistent buffer,
                so a checkpoint carries the sharpness it was trained at; 1.0 is "no anneal yet".
            use_min_dist: If True, retrieves the vector with the **minimum** dot product. Exact
                path only.
            residual: If True, considers `x` during output projection (output = g(y_ret, x)).
            num_clusters: 0 keeps the exact full-table path (what small tables and the unit tests
                use). Otherwise the number of centroids for two stage scoring; must divide
                ``num_entries``.
            probe_clusters: how many centroids a token's exact stage scores the members of.
            read_top_k: how many entries the read softmax spans.
            query_capacity_factor: per cluster query buffer size as a multiple of the mean, for the
                dispatch that scores tokens against their probed clusters. Overflowing
                (token, cluster) pairs are dropped rather than growing the buffer, exactly like a
                capacity-limited MoE dispatch -- a token keeps its other probes.
            reservoir_size: how many recent queries are kept for the refresh diagnostics (candidate
                recall) and for re-seeding dead entries.
        """
        super().__init__()
        self.num_entries = num_entries
        self.latent_dim = latent_dim
        self.output_dim = output_dim
        self.use_min_dist = use_min_dist
        self.residual = residual
        self.num_clusters = int(num_clusters)
        self.probe_clusters = int(probe_clusters)
        self.read_top_k = int(read_top_k)
        self.query_capacity_factor = float(query_capacity_factor)

        # retrieval keys z
        self.z_keys = nn.Parameter(torch.empty(num_entries, latent_dim))

        # learned, and annealed from outside. a constant temperature cannot be lowered after
        # training either: y_values that have only ever been read as a near-uniform mixture are not
        # individually meaningful, so sharpening at inference reads out vectors nothing ever
        # trained. exp() keeps it positive without a clamp; init log(temperature) so a fresh module
        # starts exactly where the old constant was.
        self.log_temperature = nn.Parameter(torch.tensor(float(math.log(temperature))))
        # externally driven anneal multiplier. a PERSISTENT BUFFER, not a plain float: the sharpness
        # the anneal ends at is part of what the trained table means, and the values were only ever
        # supervised at that sharpness. Kept as a float attribute it is absent from the state dict,
        # so a finished checkpoint silently reloads at scale 1.0 and reads ~20x flatter than it
        # trained -- which reads as "the anneal did nothing" on every eval. It is 0-dim and only
        # ever divides, so it costs no sync and no graph node.
        self.register_buffer("temperature_scale", torch.tensor(float(temperature_scale)))

        # optional instrumentation, attached from outside (LoopMixtureOfExperts shares one tracker
        # across every IR expert). None means the module is a plain retrieval module again
        self.tracker: RetrievalEntropyTracking = None

        # y vectors: trainable information vectors (the 'values')
        self.y_values = nn.Parameter(torch.randn(num_entries, output_dim) * 0.02)

        # g(x) = final transformation layer
        # project the retrieved information back into the latent space
        import transformer_engine.pytorch as te
        self.g_proj = te.Linear(
            in_features=output_dim + latent_dim if residual else output_dim,
            out_features=output_dim,
            bias=False
        )

        if self.num_clusters > 0:
            assert num_entries % self.num_clusters == 0, (
                f"num_entries {num_entries} must divide by num_clusters {self.num_clusters}"
            )
            assert not use_min_dist, "use_min_dist is exact-path only"
            capacity = num_entries // self.num_clusters
            assert self.read_top_k <= self.probe_clusters * capacity, (
                f"read_top_k {self.read_top_k} exceeds the {self.probe_clusters * capacity} "
                f"candidates {self.probe_clusters} probed clusters can supply"
            )
            # persistent: the assignment is part of what the checkpoint has to restore. loading a
            # checkpoint whose keys were clustered differently and re-deriving the partition from
            # scratch would move every token's candidate set on the first step after a resume.
            self.register_buffer(
                "cluster_members",
                torch.arange(num_entries, dtype=torch.long).view(self.num_clusters, capacity),
            )
            self.register_buffer("centroids", torch.zeros(self.num_clusters, latent_dim))
            # per entry selection EMA (for dead entry recycling) and a ring buffer of recent
            # normalized queries (for the refresh's recall measurement and for re-seeding dead keys
            # somewhere useful). Plain attributes rather than buffers, and fp32: ``model.to(BF16)``
            # casts every floating point buffer, and an EMA whose increments are ~1% of its own
            # value does not survive bf16's ulp -- the same arithmetic that forces fp32 masters in
            # the optimizer. Neither is state worth checkpointing; both are rolling samples.
            self.reservoir_size = int(reservoir_size)
            self.entry_usage = None
            self.query_reservoir = None
            self.reservoir_ptr = None
            self._usage_decay = 0.99
            self._observe_interval = 8
            self._observe_counter = 0

        self.reset_keys()

    def reset_keys(self):
        """Ensures all z vectors are correctly initialized."""
        nn.init.normal_(self.z_keys, mean=0.0, std=0.02)
        if self.num_clusters > 0:
            with torch.no_grad():
                self.centroids.copy_(
                    F.normalize(self.z_keys[self.cluster_members].float().mean(dim=1), p=2, dim=-1)
                    .to(self.centroids.dtype)
                )

    @property
    def flops_per_token(self) -> int:
        """forward FLOPs one token's retrieval costs, per application.

        The table is the one place in this model where the "2 x params" approximation is simply
        wrong: ``z_keys`` and ``y_values`` are 33.5M parameters that a two stage read touches ~1.5%
        of. Billing them densely overstates the model's compute by ~67 MFLOP/token, which is more
        than the entire dense decoder -- and every throughput and MFU number is derived from this.
        """
        if self.num_clusters == 0:
            # score every key, then read a dense [num_entries] x [num_entries, out] combination
            return 2 * self.num_entries * (self.latent_dim + self.output_dim)
        capacity = self.num_entries // self.num_clusters
        centroid_scoring = 2 * self.num_clusters * self.latent_dim
        # the exact stage pays for the dispatch buffer's padding, not just the probed candidates
        candidate_scoring = int(2 * self.probe_clusters * capacity * self.latent_dim
                                * self.query_capacity_factor)
        read = 2 * self.read_top_k * self.output_dim
        return centroid_scoring + candidate_scoring + read

    @property
    def temperature(self):
        """the effective retrieval temperature: learned, times the external anneal scale."""
        return self.log_temperature.exp() * self.temperature_scale

    @torch.no_grad()
    def set_temperature_scale(self, scale: float):
        """set the anneal multiplier on the learned temperature (1.0 = no anneal).

        Writes THROUGH the buffer rather than rebinding the attribute, so the value stays in the
        state dict and a checkpoint keeps the sharpness it was trained at.
        """
        self.temperature_scale.fill_(float(scale))

    def forward(self, x: torch.Tensor, return_weights=False, loop_idx: int = 0, **kwargs) -> (torch.Tensor | tuple[torch.Tensor, torch.Tensor]):
        """
        x: input tensor of shape (batch, seq_len, latent_dim) or (batch, latent_dim)
        return_weights: if True, also returns the retrieval weights. On the two stage path these
            are the [tokens, read_top_k] read weights, NOT a full-table distribution -- there is no
            full-table distribution to return.
        loop_idx: which loop of the MoE recurrence this call belongs to. Only used to bucket the
            entropy instrumentation per loop; the retrieval itself does not depend on it
        """
        # handle 2D or 3D inputs
        original_shape = x.shape
        if x.dim() == 3:
            x_flat = x.reshape(-1, self.latent_dim)
        else:
            x_flat = x

        # 1. normalization
        # for Cosine Similarity (via dot product)
        x_norm = F.normalize(x_flat, p=2, dim=-1)

        if self.num_clusters > 0:
            retrieved_y, weights = self._two_stage_read(x_norm)
        else:
            retrieved_y, weights = self._exact_read(x_norm)

        # instrumentation only: no grad, throttled, and reading the weights that already exist here
        # rather than recomputing them (which is what eval_stage0.py has to do from outside)
        if self.tracker is not None:
            self.tracker.update(weights, loop_idx)

        # restore original shape (batch, seq_len, output_dim)
        if len(original_shape) == 3:
            retrieved_y = retrieved_y.view(original_shape[0], original_shape[1], -1)

        # 5. final projection g
        if self.residual:
            out = self.g_proj(torch.cat([retrieved_y, x], dim=-1))
        else:
            out = self.g_proj(retrieved_y)

        if return_weights:
            return out, weights
        return out

    def _exact_read(self, x_norm: torch.Tensor):
        """full-table softmax read. [T, D] -> ([T, out], [T, num_entries])."""
        z_norm = F.normalize(self.z_keys, p=2, dim=-1)

        # 2. similarity <x, z>: [T, latent_dim] @ [latent_dim, num_entries] -> [T, num_entries]
        logits = torch.matmul(x_norm, z_norm.t())
        if self.use_min_dist:
            logits = -logits  # flip to find the minimum dot product
        logits = logits / self.temperature

        # 3. retrieval process
        weights = F.softmax(logits, dim=-1)

        # 4. retrieve the information vector y
        # [T, num_entries] @ [num_entries, output_dim] -> [T, output_dim]
        return torch.matmul(weights, self.y_values), weights

    def _two_stage_read(self, q: torch.Tensor):
        """centroid probe then exact candidate scoring. [T, D] -> ([T, out], [T, read_top_k])."""
        num_tokens = q.shape[0]
        num_clusters, capacity = self.cluster_members.shape
        probe_k = min(self.probe_clusters, num_clusters)

        # stage 1: which clusters to open. centroids are a derived quantity (the renormalized mean
        # of each cluster's members), not a parameter -- a centroid trained independently of the
        # assignment it indexes would start selecting clusters whose members no longer match it.
        centroids = F.normalize(self.centroids.to(q.dtype), p=2, dim=-1)
        probe = torch.topk(q @ centroids.t(), probe_k, dim=-1).indices        # [T, probe_k]

        # stage 2: score the probed clusters' members exactly. done by grouping tokens BY cluster
        # rather than gathering each token's candidate keys: the gather would materialize
        # [T, probe_k, capacity, D] (tens of GB at the real shape), while grouping turns it into one
        # bmm against the table viewed as [num_clusters, capacity, D].
        pairs = probe.reshape(-1)                                            # [T * probe_k]
        order = torch.argsort(pairs, stable=True)   # stable: the checkpoint recompute must agree
        cl_sorted = pairs[order]
        starts = _group_starts(cl_sorted, num_clusters)
        pos = torch.arange(pairs.numel(), device=q.device) - starts[cl_sorted]

        # fixed per cluster capacity, so the buffer size never depends on a device-side count (a
        # variable size would mean a host sync every forward). overflowing pairs are dropped.
        query_capacity = max(1, int(math.ceil(pairs.numel() / num_clusters * self.query_capacity_factor)))
        keep = pos < query_capacity
        trash = num_clusters * query_capacity          # one extra row every dropped pair writes to
        slot = cl_sorted * query_capacity + pos.clamp(max=query_capacity - 1)
        slot = torch.where(keep, slot, torch.full_like(slot, trash))

        token_of_pair = (order // probe_k)                                    # [T * probe_k]
        qbuf = q.new_zeros(trash + 1, q.shape[-1])
        qbuf.index_copy_(0, slot, q.index_select(0, token_of_pair))

        keys = F.normalize(self.z_keys, p=2, dim=-1)[self.cluster_members]     # [C, capacity, D]
        scored = torch.bmm(qbuf[:trash].view(num_clusters, query_capacity, -1), keys.transpose(1, 2))
        scored = scored.reshape(trash, capacity)

        # back to (token, probe) order. dropped pairs read the trash row, so they are masked to a
        # large negative FINITE value: -inf would make a token whose every probe overflowed produce
        # a NaN softmax instead of a uniform one.
        cand = torch.cat([scored, scored.new_full((1, capacity), -1e4)], dim=0).index_select(0, slot)
        cand = torch.where(keep.unsqueeze(-1), cand, torch.full_like(cand, -1e4))
        inv = torch.empty_like(order)
        inv.scatter_(0, order, torch.arange(order.numel(), device=q.device))
        cand = cand.index_select(0, inv).view(num_tokens, probe_k * capacity)

        top_scores, top_local = torch.topk(cand, self.read_top_k, dim=-1)      # [T, read_top_k]
        # local index -> global entry id, through the probed cluster and its member table
        probe_slot = top_local // capacity
        member_slot = top_local % capacity
        entry_ids = self.cluster_members[probe.gather(1, probe_slot), member_slot]  # [T, read_top_k]

        # fp32 softmax: post-anneal the scores are divided by ~0.05, and the masked-out candidates
        # sit at -1e4/T, which underflows cleanly in fp32 and not always in bf16
        weights = F.softmax(top_scores.float() / self.temperature.float(), dim=-1).to(q.dtype)

        self._track_usage(q, entry_ids, weights)

        values = F.embedding(entry_ids, self.y_values)                         # [T, read_top_k, out]
        read = torch.einsum("tk,tko->to", weights, values)
        return read, weights

    @torch.no_grad()
    def _track_usage(self, q: torch.Tensor, entry_ids: torch.Tensor, weights: torch.Tensor):
        """EMA of how much read mass each entry receives, plus a rolling query sample.

        Both feed ``refresh_clusters``: usage identifies entries a sharpened softmax has stopped
        selecting (and therefore stopped training), the reservoir gives the refresh real queries to
        measure candidate recall with and to re-seed those dead entries toward. Training only --
        an eval pass must not move what the next refresh acts on.
        """
        if not self.training:
            return
        if self.entry_usage is None or self.entry_usage.device != q.device:
            self.entry_usage = torch.zeros(self.num_entries, device=q.device)
            self.query_reservoir = torch.zeros(self.reservoir_size, self.latent_dim, device=q.device)
            self.reservoir_ptr = 0

        mass = torch.zeros_like(self.entry_usage)
        mass.scatter_add_(0, entry_ids.reshape(-1), weights.detach().float().reshape(-1))
        mass /= max(entry_ids.shape[0], 1)
        self.entry_usage.mul_(self._usage_decay).add_(mass, alpha=1.0 - self._usage_decay)

        # the reservoir only has to be recent and diverse, so it is sampled on the same throttle
        # the entropy tracker uses rather than written every forward
        self._observe_counter += 1
        if self._observe_counter % self._observe_interval:
            return
        flat = q.detach()
        take = min(self.reservoir_size, flat.shape[0])
        stride = max(1, flat.shape[0] // take)
        sample = flat[::stride][:take].float()
        idx = (torch.arange(sample.shape[0], device=q.device) + self.reservoir_ptr) % self.reservoir_size
        self.query_reservoir.index_copy_(0, idx, sample)
        self.reservoir_ptr = (self.reservoir_ptr + sample.shape[0]) % self.reservoir_size

    @torch.no_grad()
    def refresh_clusters(self, recycle: bool = True, dead_quantile: float = 0.0):
        """re-cluster the keys, measure the two stage path's recall, recycle dead entries.

        Cluster assignments go stale as the keys train, and a stale partition makes the exact stage
        miss entries the full-table top-k would have found -- which turns the temperature anneal
        into noise rather than sharpening. So this runs on a token cadence and its recall number is
        the check on whether ``probe_clusters`` is high enough.

        Args:
            recycle: re-seed entries a sharpened softmax has stopped selecting toward a recent
                underserved query, and zero their value. Zeroing the value is the same neutrality
                trick used everywhere else here: a recycled entry contributes nothing until it has
                learned something.
            dead_quantile: CAP on the fraction of entries recycled per refresh, not a target -- see
                ``_recycle_dead``. 0.0 disables recycling.

        Returns:
            dict of plain floats, plus ``recycled_ids`` (a device tensor) when entries were
            recycled -- the caller has to zero those rows' optimizer moments and re-seed their fp32
            masters, or the next optimizer step writes the old key straight back over the new one.
        """
        assert self.num_clusters > 0, "refresh_clusters is two-stage only"
        stats = {}
        # no reservoir yet means no training forward has run (a refresh scheduled at step 0, or a
        # pure eval process); re-cluster anyway, skip everything that needs real queries
        has_queries = self.query_reservoir is not None and bool(self.query_reservoir.abs().sum() > 0)
        queries = (
            F.normalize(self.query_reservoir.float(), p=2, dim=-1) if has_queries else None
        )

        if self.entry_usage is not None:
            # measured BEFORE recycling, and with the same "below 1% of the mean" test the recycler
            # uses: this is the population the cap is applied to. Read after the recycler has
            # re-seeded those entries' usage to the mean it would only ever report the overflow the
            # cap refused, i.e. it would read ~0 exactly when the recycler is working hardest
            threshold = self.entry_usage.mean() * 0.01
            stats["dead_frac"] = float((self.entry_usage < threshold).float().mean())

        if recycle and dead_quantile > 0.0 and has_queries:
            stats.update(self._recycle_dead(queries, dead_quantile))

        members, centroids = balanced_spherical_kmeans(
            self.z_keys, self.num_clusters,
            centroids=self.centroids if float(self.centroids.abs().sum()) > 0 else None,
        )
        self.cluster_members.copy_(members)
        self.centroids.copy_(centroids.to(self.centroids.dtype))

        if has_queries:
            stats["recall"] = self._candidate_recall(queries)
        return stats

    @torch.no_grad()
    def _recycle_dead(self, queries: torch.Tensor, dead_quantile: float, dead_ratio: float = 0.01):
        """re-seed genuinely unselected entries onto queries the table currently serves worst.

        ``dead_quantile`` is a CAP, not a target: an entry is only recycled if its usage EMA is also
        below ``dead_ratio`` of the table's mean, so a healthy table recycles nothing. Taking a
        fixed bottom quantile every refresh instead would churn 2% of the keys on every cadence
        whether or not anything had died -- 20% of the table over ten refreshes, each one thrown
        back to a zero value, which is a slow leak of trained capacity dressed up as maintenance.
        """
        cap = int(self.num_entries * dead_quantile)
        if cap == 0:
            return {"recycled": 0}
        threshold = self.entry_usage.mean() * dead_ratio
        num_dead = int(min(cap, int((self.entry_usage < threshold).sum())))
        if num_dead == 0:
            return {"recycled": 0}
        dead = torch.topk(self.entry_usage, num_dead, largest=False).indices

        # "underserved" = the queries whose best key is furthest away. a dead key parked next to a
        # well-covered query would just be dead again by the next refresh.
        z = F.normalize(self.z_keys.float(), p=2, dim=-1)
        best = (queries @ z.t()).max(dim=-1).values                 # [reservoir]
        worst = torch.argsort(best)[:num_dead]
        seeds = queries.index_select(0, worst % queries.shape[0])
        if seeds.shape[0] < num_dead:
            seeds = seeds.repeat((num_dead + seeds.shape[0] - 1) // seeds.shape[0], 1)[:num_dead]
        # a little noise so several dead entries seeded from the same query do not collapse onto
        # one point and immediately compete for the same reads
        seeds = F.normalize(seeds + 0.01 * torch.randn_like(seeds), p=2, dim=-1)

        self.z_keys.index_copy_(0, dead, (seeds * self.z_keys.float().norm(dim=-1).mean()).to(self.z_keys.dtype))
        self.y_values.index_copy_(0, dead, torch.zeros_like(self.y_values[dead]))
        # seeded with the mean rather than 0, so a freshly recycled entry is not immediately the
        # least-used one again at the next refresh before it has had a chance to be selected
        self.entry_usage.index_copy_(0, dead, torch.full_like(self.entry_usage[dead], float(self.entry_usage.mean())))
        return {"recycled": num_dead, "recycled_ids": dead}

    @torch.no_grad()
    def _candidate_recall(self, queries: torch.Tensor) -> float:
        """fraction of the exact full-table top-k that the two stage candidate set contains.

        One matmul over the reservoir (~1024 queries), i.e. cheap enough to run at every refresh.
        Below ~0.9 the centroid path is silently dropping the entries the read wanted, and the fix
        is more probed clusters, not more training.
        """
        z = F.normalize(self.z_keys.float(), p=2, dim=-1)
        exact = torch.topk(queries @ z.t(), self.read_top_k, dim=-1).indices        # [Q, k]
        centroids = F.normalize(self.centroids.float(), p=2, dim=-1)
        probe = torch.topk(queries @ centroids.t(), min(self.probe_clusters, self.num_clusters), dim=-1).indices
        cand = self.cluster_members[probe].reshape(queries.shape[0], -1)            # [Q, probe*cap]
        hit = (exact.unsqueeze(-1) == cand.unsqueeze(1)).any(dim=-1).float()
        return float(hit.mean())
