"""The fixed-target readings ``scripts/sft.py``'s ``evaluate`` returns, on a tiny model and one
hand-built evidence batch, plus the kill decision's persistence in the checkpoint payload.

1. **Per-loop answer CE comes off the same call as the headline.** ``per_condition_ce_by_loop``
   has one entry per loop and every condition, and its final loop is exactly (``==``, not close)
   ``per_condition_ce``.
2. **The selector readings are all present and in range**: raw and per chunk mass by condition,
   the gold-present against distractors AUROC on mass per chunk, and the per token chunk AUROC.
3. **Grounded AUROC is read twice**: over every answer start and over those whose buffer holds a
   chunk (the fixture has a ``none`` conversation, so the two sets differ).
4. **``kill_checked`` survives a save and a load**, and a payload written before the field existed
   loads as False.

GPU required (flash varlen, TE).
"""
import os, sys, tempfile
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from torch import optim

from modules.model.transformer import TinyMoETransformer
from scripts.prepare_evidence_data import CONDITIONS
from scripts.sft import EVIDENCE_PHASE, evaluate, load_sft_checkpoint, save_sft_checkpoint

DEVICE = "cuda"
BF16 = torch.bfloat16
PAD = 0

P = dict(
    vocab_size=512, max_seq_len=128, hidden_size=256, intermediate_size=512,
    head_dim=32, num_layers=2, num_heads=8, num_mlp_experts=8, num_attn_experts=1,
    top_k=2, n_loops=3, num_ir_experts=1, num_ir_entries=256, ir_dim=64,
    dropout=0.0, ple_embeddings_size=32, mtp_num_extra_tokens=2, lm_head_factor=4,
    evidence_port=True, groundedness_head=True,
)


def _batch():
    """Two rows of two conversations each plus a trailing pad segment.

    Row 0: a ``gold`` conversation (one gold chunk) and a ``none`` one (no evidence at all).
    Row 1: a ``mixed`` conversation (gold plus two distractors) and a ``distractors`` one (two).
    """
    torch.manual_seed(0)
    B, S, chunk_len = 2, 32, 6
    spans = [(0, 16, 12), (16, 30, 26)]                     # (start, end, answer start) per slot
    conds = [("gold", "none"), ("mixed", "distractors")]
    chunks = [[(0, 1)], [(0, 1), (0, 0), (0, 0), (1, 0), (1, 0)]]  # (slot, gold) per chunk

    input_ids = torch.randint(1, P["vocab_size"], (B, S))
    labels = torch.full((B, S), -100, dtype=torch.long)
    document_ids = torch.full((B, S), 2, dtype=torch.long)
    condition_ids = torch.full((B, S), -1, dtype=torch.long)
    answerable_ids = torch.full((B, S), -1, dtype=torch.long)
    input_ids[:, 30:] = PAD
    for r in range(B):
        for slot, (start, end, answer) in enumerate(spans):
            document_ids[r, start:end] = slot
            labels[r, answer:end] = input_ids[r, answer:end]
            condition_ids[r, start:end] = CONDITIONS.index(conds[r][slot])
            answerable_ids[r, start:end] = 1

    C = max(len(c) for c in chunks)
    S_ev = max(len(c) for c in chunks) * chunk_len
    evidence_ids = torch.zeros(B, S_ev, dtype=torch.long)
    evidence_chunk_ids = torch.full((B, S_ev), -1, dtype=torch.long)
    evidence_doc_slot = torch.full((B, S_ev), -1, dtype=torch.long)
    chunk_slot = torch.full((B, C), -1, dtype=torch.long)
    chunk_gold = torch.zeros(B, C, dtype=torch.uint8)
    for r, row_chunks in enumerate(chunks):
        for k, (slot, gold) in enumerate(row_chunks):
            at = slice(k * chunk_len, (k + 1) * chunk_len)
            evidence_ids[r, at] = torch.randint(1, P["vocab_size"], (chunk_len,))
            evidence_chunk_ids[r, at] = k
            evidence_doc_slot[r, at] = slot
            chunk_slot[r, k] = slot
            chunk_gold[r, k] = gold
    return {
        "input_ids": input_ids, "labels": labels, "document_ids": document_ids,
        "loss_weights": (labels != -100).float(),
        "condition_ids": condition_ids, "answerable_ids": answerable_ids,
        "evidence_ids": evidence_ids, "evidence_chunk_ids": evidence_chunk_ids,
        "evidence_doc_slot": evidence_doc_slot, "chunk_slot": chunk_slot,
        "chunk_keys": torch.randn(B, C, P["ir_dim"]), "chunk_gold": chunk_gold,
    }


def test_readings(model):
    batch = _batch()
    stats = evaluate(model, [batch, batch], DEVICE, PAD, max_batches=2, selector_by_loop=True)
    assert stats is not None, "evaluate read no supervised tokens"

    headline = stats["per_condition_ce"]
    by_loop = stats["per_condition_ce_by_loop"]
    expected_conds = {"gold", "mixed", "distractors", "none"}
    assert set(headline) == expected_conds, f"conditions read: {sorted(headline)}"
    assert sorted(by_loop) == list(range(P["n_loops"])), f"loops read: {sorted(by_loop)}"
    for loop_idx, ce in by_loop.items():
        assert set(ce) == expected_conds, f"loop {loop_idx} read {sorted(ce)}"
    final = by_loop[P["n_loops"] - 1]
    assert final == headline, f"final loop {final} is not the headline {headline}"
    print(f"1. per loop CE for {P['n_loops']} loops, final loop == headline exactly   PASS")

    selector = stats["selector_by_loop"]
    assert sorted(selector) == list(range(P["n_loops"])), f"selector loops: {sorted(selector)}"
    for loop_idx, reading in selector.items():
        for key in ("mass_by_condition", "mass_per_chunk_by_condition", "mass_auroc",
                    "chunk_auroc", "gold_share"):
            assert key in reading, f"loop {loop_idx} has no {key}: {sorted(reading)}"
        assert 0.0 <= reading["chunk_auroc"] <= 1.0 and 0.0 <= reading["mass_auroc"] <= 1.0
        # the none conversation's buffer is empty, so both its masses are exactly zero
        assert reading["mass_by_condition"]["none"] == 0.0
        assert reading["mass_per_chunk_by_condition"]["none"] == 0.0
        # mixed holds three chunks, gold one: per chunk mass divides mixed by three
        assert reading["mass_per_chunk_by_condition"]["mixed"] < reading["mass_by_condition"]["mixed"]
    print("2. selector readings present per loop (chunk AUROC "
          + ", ".join(f"{r['chunk_auroc']:.3f}" for r in selector.values()) + ")   PASS")

    assert "grounded_auroc" in stats and "grounded_auroc_evidence" in stats, (
        f"grounded readings missing: {sorted(k for k in stats if k.startswith('grounded'))}"
    )
    print(f"3. grounded AUROC all rows {stats['grounded_auroc']:.3f} / evidence rows "
          f"{stats['grounded_auroc_evidence']:.3f}   PASS")


def test_kill_checked_roundtrip(model):
    optimizer = optim.AdamW(model.parameters(), lr=1e-5)
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "checkpoint_evidence_test.pt")
        common = dict(epoch=0, step=3, token_count=10, start_token_count=0, global_offset=0,
                      losses=[], seed=0, phase=EVIDENCE_PHASE)
        save_sft_checkpoint(model, optimizer, scheduler, path, kill_checked=True, **common)
        state = load_sft_checkpoint(model, optimizer, scheduler, path, EVIDENCE_PHASE, tmp)
        assert state["kill_checked"] is True, f"kill_checked did not survive: {state['kill_checked']}"

        payload = torch.load(path, map_location="cpu")
        del payload["sft"]["kill_checked"]
        torch.save(payload, path)
        state = load_sft_checkpoint(model, optimizer, scheduler, path, EVIDENCE_PHASE, tmp)
        assert state["kill_checked"] is False, "an older payload did not load as kill_checked False"
    print("4. kill_checked survives save/load, an older payload reads False   PASS")


def main():
    torch.manual_seed(0)
    model = TinyMoETransformer(**P).to(DEVICE).to(BF16).eval()
    model.set_checkpointing(False, False)
    test_readings(model)
    test_kill_checked_roundtrip(model)


if __name__ == "__main__":
    main()
