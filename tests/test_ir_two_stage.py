"""Two stage IR scoring: the candidate path must agree with exact scoring where it can.

Three checks, in order of how much they would cost to get wrong:

1. **Balanced clustering partitions the table.** Every entry belongs to exactly one cluster and
   every cluster is exactly the same size. The forward path views the key table as
   ``[clusters, capacity, dim]`` and indexes it by cluster id -- an entry that appeared twice would
   be double-read, one that appeared zero times would be silently untrainable.
2. **Probing every cluster reproduces exact scoring.** With ``probe_clusters == num_clusters`` the
   candidate set IS the whole table, so the read weights and the retrieved vector must match a
   plain full-table top-k softmax read to bf16 tolerance. This is the check that the dispatch,
   capacity masking, and the local-index-to-entry-id remap are all consistent: any of them being
   off permutes the candidates and the comparison fails.
3. **The read lands on a usable scale.** Unit value rows, so the read's magnitude is bounded by the
   softmax rather than by whatever norm the rows drifted to -- the failure that left the read at
   ~0.003 of a unit residual for a whole pretraining run.
4. **Partial probing recovers most of the exact top-k.** Not exact by construction -- that is the
   trade the two stage path makes -- but the refresh's recall diagnostic is what decides whether
   ``probe_clusters`` is high enough, so the same number is asserted here on clustered data.

Requires a GPU (Transformer Engine).
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F

from modules.model.information_retrieval import (
    InformationRetrievalModule,
    balanced_spherical_kmeans,
)

torch.manual_seed(0)
DEV = "cuda"
N, D, C, CAP = 2048, 64, 32, 64
assert N == C * CAP


# --- 1. the partition ---------------------------------------------------------------------
keys = torch.randn(N, D, device=DEV)
members, centroids = balanced_spherical_kmeans(keys, C, iters=8)
assert members.shape == (C, CAP), members.shape
assert members.unique().numel() == N, "clustering lost or duplicated entries"
z = F.normalize(keys.float(), p=2, dim=-1)
own = (z[members] * centroids.unsqueeze(1)).sum(-1).mean()
other = (z @ centroids.t()).mean()
assert own > other + 0.1, f"clusters no better than random: own={own:.3f} any={other:.3f}"
print(f"[ok] balanced k-means: {C} x {CAP} exactly, member-to-own-centroid cos {own:.3f} "
      f"vs {other:.3f} to an arbitrary one")


# --- 2. probing everything == exact -------------------------------------------------------
mod = InformationRetrievalModule(
    num_entries=N, latent_dim=D, output_dim=D,
    num_clusters=C, probe_clusters=C, read_top_k=16,
).to(DEV).to(torch.bfloat16)
with torch.no_grad():
    mod.refresh_clusters(recycle=False)
    # values are seeded orthonormal PER CLUSTER, so they have to be re-drawn after a re-clustering
    # moves the membership -- the same ordering scripts/migrate_ir_reshape.py has to observe
    mod.reset_values()

q = F.normalize(torch.randn(512, D, device=DEV, dtype=torch.bfloat16), p=2, dim=-1)
with torch.no_grad():
    read, weights = mod._two_stage_read(q)

    # the reference: full-table cosine scores, top-k, softmax over those, gather
    scores = q.float() @ F.normalize(mod.z_keys.float(), p=2, dim=-1).t()
    top_s, top_i = torch.topk(scores, mod.read_top_k, dim=-1)
    ref_w = F.softmax(top_s / mod.temperature.float(), dim=-1)
    # the values as the read sees them: unit rows, so the reference has to normalize too
    ref_values = F.normalize(mod.y_values.float(), p=2, dim=-1)
    ref_read = torch.einsum("tk,tko->to", ref_w, F.embedding(top_i, ref_values))

w_err = (weights.float() - ref_w).abs().max()
r_err = (read.float() - ref_read).abs().max()
# bf16 scoring: 3 decimal digits, and a near-tie inside the top-k can swap two entries with
# essentially equal weight, so this is a tolerance and not an equality
assert w_err < 5e-2, f"read weights diverge from exact: max |dw| = {w_err:.4f}"
assert r_err < 5e-2, f"retrieved vector diverges from exact: max |dy| = {r_err:.4f}"
print(f"[ok] probe_clusters == num_clusters reproduces exact top-k scoring "
      f"(max |dw| {w_err:.2e}, max |dy| {r_err:.2e})")


# --- 2b. the read lands on a scale the residual can feel -----------------------------------
# The whole 16B token run read this table at RMS ~0.003 against a unit RMS residual, because
# nothing normalized the value side. With unit rows the read's norm is bounded by the softmax
# alone: 1/sqrt(k) for a perfectly uniform read, 1.0 for a fully peaked one. Asserting the band is
# what would catch the normalization being dropped again -- a read below it cannot move the model
# whatever it retrieves, and one above it means the rows are not unit after all.
row_norms = mod.y_values.float().norm(dim=-1)
assert (row_norms - 1.0).abs().max() < 1e-2, f"value rows are not unit: {row_norms.min():.3f}-{row_norms.max():.3f}"
read_norm = read.float().norm(dim=-1)
lo = 1.0 / (mod.read_top_k ** 0.5)
# the mean, not the min: candidates are drawn from several clusters, and only WITHIN a cluster are
# the rows exactly orthogonal, so an individual token's read can fall below 1/sqrt(k) on the cross
# terms. The quantity that matters is the scale, and the failure this guards against is three
# orders of magnitude away from either bound.
assert lo * 0.5 < read_norm.mean() and read_norm.max() < 1.05, (
    f"read norm mean {read_norm.mean():.3f}, max {read_norm.max():.3f}, expected ~[{lo:.3f}, 1.0]"
)
# and the orthonormal-within-a-cluster init actually holds where the read looks
cl = mod.cluster_members[0]
gram = F.normalize(mod.y_values[cl].float(), p=2, dim=-1)
gram = gram @ gram.t()
off = (gram - torch.eye(cl.numel(), device=DEV)).abs().max()
assert off < 5e-2, f"values inside a cluster are not orthonormal: max |off-diagonal| {off:.3f}"
print(f"[ok] unit value rows, mean read norm {read_norm.mean():.3f} against 1/sqrt(k) = {lo:.3f}, "
      f"within-cluster max |off-diagonal| {off:.2e}")


# --- 3. partial probing, and the recall diagnostic ----------------------------------------
mod4 = InformationRetrievalModule(
    num_entries=N, latent_dim=D, output_dim=D,
    num_clusters=C, probe_clusters=4, read_top_k=16,
).to(DEV).to(torch.bfloat16)
# clustered keys, i.e. what the warm start and a trained table both look like -- on isotropic
# random keys no partition can beat chance and the recall number would be meaningless
seeds = F.normalize(torch.randn(C, D, device=DEV), p=2, dim=-1)
clustered = (seeds.repeat_interleave(CAP, 0) + 0.25 * torch.randn(N, D, device=DEV))
with torch.no_grad():
    mod4.z_keys.copy_(clustered.to(torch.bfloat16))
    mod4.refresh_clusters(recycle=False)
    # queries drawn near the same seeds, so they have a real nearest cluster to find
    qq = F.normalize(
        seeds[torch.randint(C, (256,), device=DEV)] + 0.25 * torch.randn(256, D, device=DEV),
        p=2, dim=-1,
    ).float()
    curve = {}
    for p in (1, 2, 4, 8, C):
        mod4.probe_clusters = p
        curve[p] = mod4._candidate_recall(qq)
    mod4.probe_clusters = 4

# no absolute bar here: recall depends on how clustered the keys actually are, which is exactly
# what the refresh measures on the real table and what the probe count is then tuned against. What
# must hold is that the centroid stage is doing the selecting -- recall far above the fraction of
# the table the candidates cover -- and that opening more clusters monotonically helps.
for p, r in curve.items():
    if p < C:
        assert r > 1.5 * (p / C), f"probing {p}/{C} clusters recalls {r:.3f}, no better than chance"
assert curve[C] > 0.999, f"probing every cluster must recall everything, got {curve[C]:.4f}"
assert curve[8] > curve[4] > curve[2] > curve[1], f"recall not monotone in probes: {curve}"
print("[ok] candidate recall@%d by probed clusters: %s" % (
    mod4.read_top_k, ", ".join(f"{p}->{r:.3f}" for p, r in curve.items())))


# --- 4. gradients reach the selected entries and the temperature --------------------------
mod4.train()
x = torch.randn(64, D, device=DEV, dtype=torch.bfloat16, requires_grad=True)
out = mod4(x)
out.float().pow(2).mean().backward()
assert mod4.y_values.grad is not None and mod4.y_values.grad.abs().sum() > 0
assert mod4.z_keys.grad is not None and mod4.z_keys.grad.abs().sum() > 0
assert mod4.log_temperature.grad is not None and mod4.log_temperature.grad.abs() > 0
touched = (mod4.y_values.grad.abs().sum(-1) > 0).sum().item()
assert touched <= 64 * mod4.read_top_k, "more entries got gradient than were ever selected"
assert mod4.entry_usage is not None and float(mod4.entry_usage.sum()) > 0
print(f"[ok] gradients: {touched} of {N} entries received one (<= 64 x {mod4.read_top_k} "
      f"selected), keys/values/log_temperature all wet, usage EMA live")

print("\nall two stage IR checks passed")
