"""Loss-free MoE balancing: the selection bias on the router, in one GPU pass.

1. a zero bias is an exact no-op: the selection and the weights equal a plain topk over the
   routing probabilities, and the buffer is in the state dict;
2. a nonzero bias moves the selection but not the weights, which stay the unbiased probabilities;
3. the hard counts follow the token mask and the update moves every MLP slot's bias by exactly
   one rate step toward the under-loaded experts, leaves the dense slots at zero, skips a pass
   that ran nothing, and resets the counts;
4. the bias survives a state dict round trip, a checkpoint without the buffer loads as the
   unbiased router, and the load statistic reads 1.0 on an even load.

GPU + TE required.
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torch.nn import functional as F

from modules.model.transformer import TinyMoETransformer
from utils import load_model_state

torch.manual_seed(0)
dev = "cuda"
N_LOOPS = 3
P = dict(vocab_size=512, max_seq_len=128, hidden_size=256, intermediate_size=512,
         head_dim=32, num_layers=2, num_heads=8, num_mlp_experts=8, num_attn_experts=1,
         top_k=2, n_loops=N_LOOPS, num_ir_experts=1, num_ir_entries=256, ir_dim=64,
         dropout=0.0, ple_embeddings_size=32, mtp_num_extra_tokens=2, lm_head_factor=4)


def main():
    model = TinyMoETransformer(**P).to(dev).to(torch.bfloat16)
    moe = model.moe
    E, first = moe.num_experts, moe.first_mlp_index
    B, S = 2, 64
    x = torch.randn(B, S, P["hidden_size"], device=dev, dtype=torch.bfloat16)

    # 1. zero bias: the plain topk
    model.eval()
    assert moe.balance_bias.shape == (N_LOOPS, E) and "moe.balance_bias" in model.state_dict()
    with torch.no_grad():
        probs = F.softmax((moe.router(x) + moe.loop_bias(1)) / moe.temperature, dim=-1)
        ref_scores, ref_idx = torch.topk(probs, moe.top_k, dim=-1)
        ref_scores = ref_scores / ref_scores.sum(-1, keepdim=True)
        scores, idx, _ = moe.route(x, temperature=moe.temperature, loop_idx=1)
    assert torch.equal(idx, ref_idx) and torch.equal(scores, ref_scores)
    print("1. a zero bias routes exactly as the plain topk                                PASS")

    # 2. a nonzero bias moves the selection only
    target = first + 3
    with torch.no_grad():
        moe.balance_bias[1, target] = 1.0
        scores2, idx2, _ = moe.route(x, temperature=moe.temperature, loop_idx=1)
        picked = (idx2 == target).any(dim=-1)
        assert bool(picked.all()), "a bias of 1.0 on a probability must put that expert in every token's top k"
        gathered = probs.gather(-1, idx2)
        assert torch.equal(scores2, gathered / gathered.sum(-1, keepdim=True)), "weights must be the unbiased probabilities"
        # the other passes' rows are still zero, so pass 2 routes as the plain topk
        probs2 = F.softmax((moe.router(x) + moe.loop_bias(2)) / moe.temperature, dim=-1)
        _, idx_other, _ = moe.route(x, temperature=moe.temperature, loop_idx=2)
        assert torch.equal(idx_other, torch.topk(probs2, moe.top_k, dim=-1).indices)
        moe.balance_bias.zero_()
    print("2. a nonzero bias moves the selection and leaves the weights unbiased         PASS")

    # 3. counts, the update and the mask
    model.train()
    moe.set_router_noise(0.0)
    moe.set_balance_tracking(True)
    mask = torch.ones(B, S, dtype=torch.bool, device=dev)
    mask[:, S // 2:] = False
    with torch.no_grad():
        _, idx3, _ = moe.route(x, temperature=moe.temperature, loop_idx=0, token_mask=mask)
    bias, counts, load = moe._balance_state(x.device)
    expect = torch.zeros(E, device=dev)
    expect.scatter_add_(0, idx3[:, : S // 2].reshape(-1), torch.ones(B * (S // 2) * moe.top_k, device=dev))
    assert torch.equal(counts[0], expect), "counts must follow the token mask"
    assert counts[1].sum() == 0 and counts[2].sum() == 0
    mlp_counts = expect[first:]
    rate = 0.001
    moe.update_balance_bias(rate)
    sign = torch.sign(mlp_counts.mean() - mlp_counts)
    centred = sign - sign.mean()
    assert torch.allclose(bias[0, first:], rate * centred), (bias[0], centred)
    assert abs(bias[0, first:].sum().item()) < 1e-6, "the update is zero mean over the MLP slots"
    assert torch.all(bias[0, :first] == 0), "dense slots keep a zero bias"
    assert torch.all(bias[1] == 0) and torch.all(bias[2] == 0), "a pass that ran nothing is untouched"
    assert torch.allclose(moe.balance_bias[0].float(), bias[0], atol=1e-3)
    assert counts.sum() == 0, "counts reset after the update"
    assert torch.allclose(load[0], 0.1 * expect / expect.sum()) and load[1].sum() == 0
    ratio, dense = moe.balance_load_stats()
    assert len(ratio) == N_LOOPS and ratio[1] == 0.0 and ratio[0] >= 1.0
    assert abs(dense[0] - 0.1 * expect[:first].sum().item() / expect.sum().item()) < 1e-6 and dense[1] == 0.0
    # an even load reads 1.0 on the statistic once the EMA is the only thing in it, and moves
    # nothing; a skewed load with more cold than hot slots still sums to zero
    with torch.no_grad():
        counts.zero_()
        counts[2, first:] = 10.0
        counts[1, first:] = 1.0
        counts[1, first] = 50.0
        load.zero_()
    moe.update_balance_bias(rate)
    assert abs(moe.balance_load_stats()[0][2] - 1.0) < 1e-6 and torch.all(bias[2] == 0)
    assert bias[1, first] < 0 and torch.all(bias[1, first + 1:] > 0) and abs(bias[1].sum().item()) < 1e-6
    # a zero rate (the aux mode's call) reads the load into the EMA and resets the counts
    # without moving the bias by a single bit
    with torch.no_grad():
        bias.zero_()
        moe.balance_bias.zero_()
        counts.zero_()
        counts[0, first:] = 1.0
        counts[0, first] = 50.0
    before = moe.balance_bias.clone()
    moe.update_balance_bias(0.0)
    assert torch.equal(moe.balance_bias, before) and torch.all(bias == 0), "a zero rate moved the bias"
    assert counts.sum() == 0 and moe.balance_load_stats()[0][0] > 1.0, "a zero rate must still read the load"
    print("3. masked counts, one centred rate step toward the under loaded MLP slots, reset  PASS")

    # 4. the round trip, the absent buffer and the fp32 shadow
    with torch.no_grad():
        moe.balance_bias[0, first] = 0.25
    state = model.state_dict()
    fresh = TinyMoETransformer(**P).to(dev).to(torch.bfloat16)
    load_model_state(fresh, state)
    assert torch.equal(fresh.moe.balance_bias, moe.balance_bias)
    without = {k: v for k, v in state.items() if k != "moe.balance_bias"}
    older = TinyMoETransformer(**P).to(dev).to(torch.bfloat16)
    load_model_state(older, without)
    assert torch.all(older.moe.balance_bias == 0), "an absent buffer is the unbiased router"
    shadow, _, _ = fresh.moe._balance_state(dev)
    assert shadow.dtype == torch.float32 and torch.allclose(shadow[0, first], torch.tensor(0.25, device=dev))
    print("4. state dict round trip, absent buffer loads unbiased, fp32 shadow            PASS")

    # 5. the selection scores are formed in fp32
    model.eval()
    with torch.no_grad():
        moe.balance_bias.zero_()
        big = torch.randn(8, 512, P["hidden_size"], device=dev, dtype=torch.bfloat16)
        probs = F.softmax((moe.router(big) + moe.loop_bias(0)) / moe.temperature, dim=-1)
        old_idx = torch.topk(probs + moe.balance_bias[0].to(probs.dtype), moe.top_k, dim=-1).indices
        w, idx, _ = moe.route(big, temperature=moe.temperature, loop_idx=0)
        assert idx[..., 0].numel() >= 1000 and torch.equal(idx, old_idx), "zero bias moved the selection"
        g = probs.gather(-1, old_idx)
        assert torch.equal(w, g / g.sum(-1, keepdim=True)), "zero bias moved the weights"
        # ties: coarse probabilities so most rows hold repeated values, bf16 topk against fp32 topk
        tied = (probs.float() * 8).round().div(8).to(torch.bfloat16)
        assert torch.equal(torch.topk(tied, moe.top_k, dim=-1).indices,
                           torch.topk(tied.float(), moe.top_k, dim=-1).indices), "ties broke differently in fp32"
        # a bias of a few tenths: the picks follow the fp32 sum, which the bf16 sum gets wrong
        moe.balance_bias[0, first:] = torch.linspace(0.30, 0.31, E - first, device=dev).to(torch.bfloat16)
        _, idx_b, _ = moe.route(big, temperature=moe.temperature, loop_idx=0)
        bias32 = moe.balance_bias[0].float()
        ref = torch.topk(probs.float() + bias32, moe.top_k, dim=-1).indices
        coarse = torch.topk(probs + moe.balance_bias[0], moe.top_k, dim=-1).indices
        assert torch.equal(idx_b, ref), "selection is not the fp32 sum"
        print(f"   bf16 sum picks differ from fp32 on {(coarse != ref).any(-1).float().mean().item():.3f} of tokens")
        moe.balance_bias.zero_()
    print("5. selection scores in fp32: zero bias identical (with ties), a large bias exact  PASS")
    print("all loss free balancing checks passed")


if __name__ == "__main__":
    main()
