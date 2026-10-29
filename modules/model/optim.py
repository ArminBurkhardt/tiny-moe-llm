"""Muon on the hidden matrices and AdamW on everything else, as one optimizer.

The real run's optimizer rule, as one object so the trainers keep taking one optimizer and one
``LambdaLR``. Muon (Jordan et al., 2024; Moonlight, arXiv 2502.16982) orthogonalizes each 2D
weight's momentum with five Newton-Schulz iterations in bf16 and scales the result to AdamW's
update RMS (``0.2 * sqrt(max(m, n))``), so the learning rate and the decoupled weight decay of an
AdamW run carry over unchanged. AdamW steps the rest with ``torch.optim.AdamW``'s arithmetic and
state keys (``exp_avg``, ``exp_avg_sq``, ``step``), so code that edits Adam moments by name
(``sft.apply_ir_refresh``) sees the same state either way.

The partition is a rule over parameter names, not a list (``muon_kind``). On Muon: the hidden 2D
projections of every attention (q, k, v, o), the dense and shared MLPs, the routed experts one
matrix at a time (TE ``GroupedLinear`` keeps one 2D ``weight{i}`` per expert), the selector's
query and output projections, the store adapters and the per-layer-embedding projections. Fused
``gate_up`` weights hold the gate rows over the up rows, so their two halves are orthogonalized
as two matrices. On AdamW: every table (the token embeddings, the per-layer tables, the IR keys and
values, the loop-conditioned biases over the sinusoidal loop encoding), every head (the LM head
and its sub-heads and projection, the MTP head), the router and its noise projection, every tensor
of ndim 1 or less, and the zero-init neutrality matrices (``loop_inject``, ``direct_gate``, the IR
``g_proj``), which a normalized update would move to full RMS on the first step.
"""
import math
import re

import torch

MUON_MOMENTUM = 0.95
MUON_NS_STEPS = 5
MUON_RMS_MATCH = 0.2
NS_COEFFICIENTS = (3.4445, -4.7750, 2.0315)

_ADAMW_PATTERNS = (
    r"embed_tokens\.weight$", r"\.ple\.weight$", r"moe_embeddings\.weight$",
    r"\.z_keys$", r"\.y_values$",
    r"loop_query_bias\.weight$", r"loop_router_bias\.weight$", r"evidence_query_bias\.weight$",
    r"^lm_head\.", r"^mtp_head\.", r"^coda_head\.", r"^groundedness_head\.",
    r"\.router\.", r"noise_proj\.weight$",
    # the zero-init neutrality matrices: moe.inject (loop_inject), the IR expert's direct_gate and
    # g_proj, and the cross reader's o_proj
    r"(^|\.)inject\.weight$", r"direct_gate\.weight$", r"\.g_proj\.weight$",
    r"shared_evidence\.attn\.o_proj\.weight$",
)
_FUSED_PATTERN = re.compile(r"gate_up\.weight\d*$")


def muon_kind(name: str, param: torch.Tensor) -> str:
    """Which update a parameter takes: ``"adamw"``, ``"muon"`` or ``"muon_halves"``.

    Args:
        name: the parameter's name from ``named_parameters``.
        param: the parameter (only ``ndim`` is read).

    Returns:
        ``"adamw"`` for tables, heads, the router, scalars, vectors and the zero-init neutrality
        matrices; ``"muon_halves"`` for a fused ``gate_up`` weight (rows ``[:I]`` and ``[I:]``
        orthogonalized separately); ``"muon"`` for every other 2D weight.
    """
    if param.ndim < 2:
        return "adamw"
    for pattern in _ADAMW_PATTERNS:
        if re.search(pattern, name):
            return "adamw"
    if _FUSED_PATTERN.search(name):
        return "muon_halves"
    return "muon"


def newton_schulz(grad: torch.Tensor, steps: int = MUON_NS_STEPS, eps: float = 1e-7) -> torch.Tensor:
    """An approximate orthogonalization of a 2D matrix, in bf16.

    The quintic iteration from the Muon reference with its coefficients tuned to converge the
    singular values into about (0.68, 1.13) in five steps rather than to exactly 1. A tall matrix
    is transposed first so the square product is the smaller one.

    Args:
        grad: ``[m, n]`` matrix (any float dtype).
        steps: iterations.
        eps: added to the Frobenius norm before the first scaling.

    Returns:
        ``[m, n]`` bf16 matrix with near-unit singular values.
    """
    assert grad.ndim == 2, grad.shape
    a, b, c = NS_COEFFICIENTS
    x = grad.to(torch.bfloat16)
    transposed = x.shape[0] > x.shape[1]
    if transposed:
        x = x.mT
    x = x / (x.norm() + eps)
    for _ in range(steps):
        gram = x @ x.mT
        x = a * x + (b * gram + c * (gram @ gram)) @ x
    if transposed:
        x = x.mT
    return x


def muon_update(momentum: torch.Tensor, halves: bool, steps: int = MUON_NS_STEPS) -> torch.Tensor:
    """The orthogonalized, RMS-matched update for one weight from its (Nesterov) momentum.

    Args:
        momentum: ``[m, n]`` fp32 momentum (after the Nesterov blend).
        halves: orthogonalize rows ``[:m // 2]`` and ``[m // 2:]`` as two matrices.
        steps: Newton-Schulz iterations.

    Returns:
        ``[m, n]`` fp32 update with RMS about ``MUON_RMS_MATCH``.
    """
    if halves:
        half = momentum.shape[0] // 2
        return torch.cat([muon_update(momentum[:half], False, steps),
                          muon_update(momentum[half:], False, steps)], dim=0)
    m, n = momentum.shape
    return newton_schulz(momentum, steps).float() * (MUON_RMS_MATCH * math.sqrt(max(m, n)))


class MuonAdamW(torch.optim.Optimizer):
    """One optimizer: Muon on the groups tagged ``muon``, AdamW's arithmetic on the rest.

    A param group with ``"muon": "muon"`` or ``"muon": "muon_halves"`` steps every parameter in
    it with Muon (Nesterov momentum ``momentum``, the orthogonalized update scaled to AdamW's RMS,
    decoupled weight decay ``weight_decay`` at the group's ``lr``; the momentum buffer is kept in
    fp32 whatever the parameter's dtype, also after a resume). Every other group steps with AdamW
    (``betas``, ``eps``, decoupled ``weight_decay``): the arithmetic of ``torch.optim.AdamW``'s
    single-tensor path without ``amsgrad`` or ``maximize``, with the same state keys
    (``exp_avg``, ``exp_avg_sq``, ``step``); its foreach path sums in another order, so the two
    agree to rounding, not bit for bit.

    Args:
        params: param groups (dicts) or an iterable of parameters.
        lr: learning rate (the group's own ``lr`` wins).
        betas: AdamW moments.
        eps: AdamW epsilon.
        weight_decay: decoupled weight decay (the group's own wins); ``torch.optim.AdamW``'s
            default, so a group without the key decays as it would there.
        momentum: Muon's momentum.
        nesterov: blend the gradient into the momentum before orthogonalizing.
        ns_steps: Newton-Schulz iterations.
    """

    def __init__(self, params, lr: float = 1e-3, betas=(0.9, 0.999), eps: float = 1e-8,
                 weight_decay: float = 1e-2, momentum: float = MUON_MOMENTUM, nesterov: bool = True,
                 ns_steps: int = MUON_NS_STEPS):
        assert 0.0 <= momentum < 1.0, momentum
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, momentum=momentum,
                        nesterov=nesterov, ns_steps=ns_steps, muon=None)
        super().__init__(params, defaults)
        for group in self.param_groups:
            assert group["muon"] in (None, "adamw", "muon", "muon_halves"), group["muon"]
            if group["muon"] in ("muon", "muon_halves"):
                for p in group["params"]:
                    assert p.ndim == 2, f"a Muon group holds a {p.ndim}D tensor of shape {tuple(p.shape)}"
                    if group["muon"] == "muon_halves":
                        assert p.shape[0] % 2 == 0, f"a fused weight with an odd row count {tuple(p.shape)}"

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            if group["muon"] in ("muon", "muon_halves"):
                self._muon_step(group)
            else:
                self._adamw_step(group)
        return loss

    def _muon_step(self, group):
        lr, wd = group["lr"], group["weight_decay"]
        momentum, nesterov, steps = group["momentum"], group["nesterov"], group["ns_steps"]
        halves = group["muon"] == "muon_halves"
        for p in group["params"]:
            if p.grad is None:
                continue
            grad = p.grad.float()
            state = self.state[p]
            if "momentum_buffer" not in state:
                state["momentum_buffer"] = torch.zeros_like(grad)
                state["step"] = torch.zeros((), dtype=torch.float32)
            buf = state["momentum_buffer"]
            if buf.dtype != torch.float32:
                # load_state_dict casts the buffer to the parameter's dtype; pretraining steps bf16
                # parameters directly, so a resumed buffer comes back bf16 and the momentum would
                # run in bf16 from then on
                buf = state["momentum_buffer"] = buf.float()
            state["step"] += 1
            buf.lerp_(grad, 1.0 - momentum)
            blended = grad.lerp(buf, momentum) if nesterov else buf
            update = muon_update(blended, halves, steps)
            if wd != 0.0:
                p.mul_(1.0 - lr * wd)
            p.add_(update.to(p.dtype), alpha=-lr)

    def _adamw_step(self, group):
        lr, wd, eps = group["lr"], group["weight_decay"], group["eps"]
        beta1, beta2 = group["betas"]
        for p in group["params"]:
            if p.grad is None:
                continue
            grad = p.grad
            state = self.state[p]
            if "exp_avg" not in state:
                state["step"] = torch.zeros((), dtype=torch.float32)
                state["exp_avg"] = torch.zeros_like(p)
                state["exp_avg_sq"] = torch.zeros_like(p)
            exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
            state["step"] += 1
            step = float(state["step"])
            if wd != 0.0:
                p.mul_(1.0 - lr * wd)
            exp_avg.lerp_(grad, 1.0 - beta1)
            exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)
            bias_correction1 = 1.0 - beta1 ** step
            bias_correction2 = 1.0 - beta2 ** step
            step_size = lr / bias_correction1
            denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(eps)
            p.addcdiv_(exp_avg, denom, value=-step_size)


def tag_param_groups(groups, names_by_param: dict) -> list:
    """Split each param group by ``muon_kind`` into tagged groups.

    Args:
        groups: param groups whose ``params`` are the tensors the optimizer steps; when a group
            steps fp32 masters, ``names_by_param`` maps the master to the model parameter's name.
        names_by_param: ``{id(stepped tensor): name}``.

    Returns:
        The same groups with ``"muon"`` set, one group per (original group, kind), kinds in the
        order ``adamw``, ``muon``, ``muon_halves``; empty ones dropped.
    """
    out = []
    for group in groups:
        by_kind = {"adamw": [], "muon": [], "muon_halves": []}
        for p in group["params"]:
            by_kind[muon_kind(names_by_param[id(p)], p)].append(p)
        for kind, params in by_kind.items():
            if params:
                tagged = {k: v for k, v in group.items() if k != "params"}
                tagged["params"] = params
                tagged["muon"] = kind
                out.append(tagged)
    return out


def build_optimizer(kind: str, groups, names_by_param: dict = None, **kwargs):
    """``torch.optim.AdamW`` for ``"adamw"``; ``MuonAdamW`` over ``tag_param_groups`` for ``"muon"``.

    Args:
        kind: ``"adamw"`` or ``"muon"``.
        groups: param groups as the trainers build them.
        names_by_param: required for ``"muon"``, see ``tag_param_groups``.
        **kwargs: ``lr`` and the rest, passed to the optimizer.

    Returns:
        The optimizer.
    """
    if kind == "adamw":
        return torch.optim.AdamW(groups, **kwargs)
    assert kind == "muon", kind
    assert names_by_param is not None, "the Muon partition is a rule over parameter names"
    return MuonAdamW(tag_param_groups(groups, names_by_param), **kwargs)


def optimizer_kind(optimizer) -> str:
    """``"muon"`` when any param group steps with Muon, else ``"adamw"``."""
    return "muon" if any(g.get("muon") in ("muon", "muon_halves") for g in optimizer.param_groups) else "adamw"


def saved_optimizer_kind(optimizer_state_dict: dict) -> str:
    """The kind a saved optimizer state was written under, read off its param groups.

    A ``MuonAdamW`` state carries the ``muon`` tag on every group; a ``torch.optim.AdamW`` state
    carries none. The two lay their groups out differently, so a resume across kinds fails in
    ``load_state_dict`` with a group-count or size error that names neither; the trainers check
    this first and name both kinds.

    Args:
        optimizer_state_dict: the ``optimizer_state_dict`` of a checkpoint.
    """
    groups = optimizer_state_dict.get("param_groups", [])
    return "muon" if any(g.get("muon") in ("muon", "muon_halves") for g in groups) else "adamw"


def check_resume_kind(optimizer, optimizer_state_dict: dict, path: str = "") -> None:
    """Raise before ``load_state_dict`` when the checkpoint and the live optimizer differ in kind.

    Args:
        optimizer: the live optimizer.
        optimizer_state_dict: the checkpoint's.
        path: the checkpoint's path, for the message.
    """
    saved, live = saved_optimizer_kind(optimizer_state_dict), optimizer_kind(optimizer)
    if saved != live:
        raise ValueError(
            f"{path or 'the checkpoint'} was saved under optimizer {saved!r} and this run is "
            f"configured for {live!r} (training.optimizer): the group layouts differ, so it cannot "
            f"resume; relaunch under {saved!r}, or pass it with -c to initialize from its weights"
        )


def partition_summary(optimizer) -> str:
    """One line: how many tensors and elements each kind steps."""
    counts = {}
    for group in getattr(optimizer, "param_groups", []):
        kind = group.get("muon") or "adamw"
        n, numel = counts.get(kind, (0, 0))
        counts[kind] = (n + len(group["params"]), numel + sum(p.numel() for p in group["params"]))
    return ", ".join(f"{kind}: {n} tensors, {numel / 1e6:.1f}M" for kind, (n, numel) in sorted(counts.items()))
