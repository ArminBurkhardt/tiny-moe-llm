"""``scripts/sft.py``'s ``evaluate`` split by source, on a tiny model and hand-built evidence batches.

Reuses the fixture of ``test_eval_fixed_readings.py`` (four conversations, one per condition) and
adds ``source_ids`` to it, then a second batch with different content and the two sources swapped
between conversations, so every condition has tokens from both sources.

1. **Per source CE is present**: ``per_source_condition_ce`` holds both source names with the
   conditions each one has, ``per_source_tokens`` counts each source's supervised tokens.
2. **The split is a partition**: the token weighted average of the per source CE equals the pooled
   ``per_condition_ce`` for every condition to 1e-5, with the weights counted independently from the
   batch tensors.
3. **Without ``source_ids`` nothing changes**: the result has exactly the keys ``evaluate`` returned
   before, none of the two new ones, and the same pooled numbers as the call with sources.

GPU required (flash varlen, TE).
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch

from modules.model.transformer import TinyMoETransformer
from scripts.prepare_evidence_data import CONDITIONS, SOURCE_KEYS
from scripts.sft import evaluate
from test_eval_fixed_readings import BF16, DEVICE, P, PAD, _batch

NEW_KEYS = {"per_source_condition_ce", "per_source_tokens"}


def with_sources(batch, sources_by_slot):
    """``source_ids`` over the query axis: each conversation's slot value, -1 on row padding."""
    out = dict(batch)
    ids = torch.full_like(batch["document_ids"], -1)
    for r in range(ids.shape[0]):
        for slot, source in enumerate(sources_by_slot[r]):
            ids[r][batch["document_ids"][r] == slot] = source
    ids[batch["input_ids"] == PAD] = -1
    out["source_ids"] = ids
    return out


def second_batch():
    """Same layout as ``_batch`` with fresh tokens, so per source CE differs between batches."""
    batch = _batch()
    g = torch.Generator().manual_seed(11)
    fresh = torch.randint(1, P["vocab_size"], batch["input_ids"].shape, generator=g)
    fresh[:, 30:] = PAD
    supervised = batch["labels"] != -100
    batch["input_ids"] = fresh
    batch["labels"] = torch.where(supervised, fresh, torch.full_like(fresh, -100))
    batch["evidence_ids"] = torch.randint(1, P["vocab_size"], batch["evidence_ids"].shape, generator=g)
    return batch


def main():
    torch.manual_seed(0)
    model = TinyMoETransformer(**P).to(DEVICE).to(BF16).eval()
    model.set_checkpointing(False, False)

    first, second = _batch(), second_batch()
    # row 0: slot 0 source 0, slot 1 source 1; row 1 the other way round. The second batch swaps both
    batches = [with_sources(first, [(0, 1), (1, 0)]), with_sources(second, [(1, 0), (0, 1)])]
    plain = [dict((k, v) for k, v in b.items() if k != "source_ids") for b in batches]

    stats = evaluate(model, batches, DEVICE, PAD, max_batches=2, selector_by_loop=True)
    assert stats is not None and NEW_KEYS <= set(stats), sorted(stats)
    by_source = stats["per_source_condition_ce"]
    names = {SOURCE_KEYS[0], SOURCE_KEYS[1]}
    assert set(by_source) == names and set(stats["per_source_tokens"]) == names, (
        sorted(by_source), sorted(stats["per_source_tokens"]))
    for name in names:
        assert set(by_source[name]) == {"gold", "mixed", "distractors", "none"}, by_source[name]
    print(f"1. per source CE present for {sorted(names)}, every condition               PASS")

    for cond_idx, cond in enumerate(CONDITIONS):
        if cond not in stats["per_condition_ce"]:
            continue
        num = den = 0.0
        for src_idx, name in enumerate(SOURCE_KEYS[:2]):
            weight = sum(
                int(((b["labels"][:, 1:] != -100) & (b["condition_ids"][:, 1:] == cond_idx)
                     & (b["source_ids"][:, 1:] == src_idx)).sum())
                for b in batches
            )
            assert weight > 0, (cond, name)
            num += by_source[name][cond] * weight
            den += weight
        assert abs(num / den - stats["per_condition_ce"][cond]) < 1e-5, (
            cond, num / den, stats["per_condition_ce"][cond])
    for src_idx, name in enumerate(SOURCE_KEYS[:2]):
        expected = sum(int(((b["labels"][:, 1:] != -100) & (b["source_ids"][:, 1:] == src_idx)).sum())
                       for b in batches)
        assert stats["per_source_tokens"][name] == expected, (name, stats["per_source_tokens"][name], expected)
    print("2. token weighted mean over sources equals the pooled CE (1e-5)           PASS")

    baseline = evaluate(model, plain, DEVICE, PAD, max_batches=2, selector_by_loop=True)
    assert not (NEW_KEYS & set(baseline)), "source keys appeared without source_ids"
    assert set(baseline) == set(stats) - NEW_KEYS, (sorted(baseline), sorted(stats))
    for cond, ce in baseline["per_condition_ce"].items():
        assert abs(ce - stats["per_condition_ce"][cond]) < 1e-5
    print("3. without source_ids the keys are exactly today's and the pooled CE is equal PASS")


if __name__ == "__main__":
    main()
