"""The evidence corpus round trip: what the builder wrote is what the document reads.

Flash pairs the query and evidence sides **by position**, so the only thing standing between a
document and another document's passage is that the two segment lists are the same length in the
same order. Nothing in the forward can check that -- a shifted list produces a perfectly finite loss
that is learning the wrong association -- so it is checked here, on a corpus small enough to state
the expected answer by hand.

Five assertions:

1. **The two segment lists have the same length.** One evidence segment per query segment, including
   the empty ones.
2. **Every document reads its OWN evidence**, matched by token id. This is the assertion that
   catches an off-by-one in the pairing.
3. **A document that retrieved nothing gets a zero length segment in its own position**, rather than
   being skipped -- skipping it would shift every later document by one.
4. **Chunks land on their document too**, on the selector's side, which numbers them independently.
5. **Evidence padding lands on a trailing pad segment**, never on a real conversation, so no
   supervised token can attend to another row's leftovers.

The first four are pure index arithmetic and run anywhere; assertion 5 and the end-to-end isolation
check at the bottom need a GPU.
"""
import os, sys, shutil, tempfile
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from types import SimpleNamespace

import numpy as np
import torch

from modules.data.evidence_dataset import EvidenceDataset, evidence_from_batch, EMBED_DIM
from modules.model.attention import cu_seqlens_from_doc_ids, _segment_ids
from modules.model.evidence import evidence_cu_seqlens
from scripts.prepare_evidence_data import EvidenceWriter

BOS, PAD = 0, 1
# small enough that these five documents pack into TWO rows, which is what makes the batch's evidence
# widths unequal and therefore makes row padding exist at all -- assertion 5 has nothing to check on
# a single-row batch
MAX_LEN = 16

# (prompt tokens, evidence tokens, chunk lengths). Token ranges are disjoint per document so an
# evidence token identifies the document it belongs to on sight -- which is what assertion 2 reads.
DOCS = [
    ([10, 11, 12], [100, 101], [2]),
    ([20, 21], [], []),                      # retrieved nothing: the zero length segment case
    ([30, 31, 32, 33], [300, 301, 302], [2, 1]),
    ([40, 41, 42], [400, 401, 402, 403], [4]),
    ([50, 51], [500, 501, 502], [3]),
]


def build_corpus(data_dir, split="evidence_train"):
    state = {}
    writer = EvidenceWriter(data_dir, split, state)
    for tokens, ev, chunk_lens in DOCS:
        ids = [BOS] + tokens
        mask = [0] + [1] * len(tokens)
        ev_chunk = []
        for chunk_idx, length in enumerate(chunk_lens):
            ev_chunk.extend([chunk_idx] * length)
        assert len(ev_chunk) == len(ev), "the fixture's chunk lengths do not cover its evidence"
        keys = np.zeros((len(chunk_lens), EMBED_DIM), dtype=np.float32)
        # first component identifies the chunk's document, so assertion 4 can read it back
        for chunk_idx in range(len(chunk_lens)):
            keys[chunk_idx, 0] = ev[0] if ev else 0.0
        writer.write(ids, mask, ev, ev_chunk, keys)
    writer.sync()
    writer.close()
    return state


def main():
    tmp = tempfile.mkdtemp(prefix="evidence_corpus_")
    try:
        build_corpus(tmp)
        tokenizer = SimpleNamespace(bos_token_id=BOS, pad_token_id=PAD, eos_token_id=PAD)
        dataset = EvidenceDataset(
            tmp, tokenizer, batch_size=2, max_length=MAX_LEN, split="evidence_train",
            num_mtp_tokens=1, shuffle=False,
        )
        batches = list(iter(dataset))
        assert batches, "the dataset yielded nothing"
        batch = batches[0]
        assert "evidence_ids" in batch, "the batch carries no evidence"

        doc_ids = batch["document_ids"]
        B, S = doc_ids.shape
        cu, _ = cu_seqlens_from_doc_ids(doc_ids)
        num_segments = int(cu.numel() - 1)
        seg = _segment_ids(cu, B, S, doc_ids.device)

        doc_slot = batch["evidence_doc_slot"]
        ev_segments = torch.where(doc_slot >= 0, seg[:, :1] + doc_slot, seg[:, -1:].expand_as(doc_slot))
        cu_k, max_k = evidence_cu_seqlens(ev_segments, num_segments)

        assert int(cu_k.numel()) == num_segments + 1, (
            f"{int(cu_k.numel()) - 1} evidence segments against {num_segments} query segments"
        )
        assert int(cu_k[-1]) == ev_segments.numel(), "the evidence cu_seqlens does not cover every token"
        print(f"1. one evidence segment per query segment ({num_segments})       PASS")

        # walk the flattened evidence axis segment by segment and compare against the fixture
        flat_ev = batch["evidence_ids"].reshape(-1).tolist()
        # which fixture document each conversation slot holds, in packing order
        expected = [ev for _, ev, _ in DOCS]
        conv_segments = []
        for r in range(B):
            n_conv = int((doc_slot[r] >= 0).any()) and int(doc_slot[r].max()) + 1
            base = int(seg[r, 0])
            conv_segments.extend(base + j for j in range(n_conv))

        checked, empties = 0, 0
        for local_idx, global_seg in enumerate(conv_segments):
            lo, hi = int(cu_k[global_seg]), int(cu_k[global_seg + 1])
            got = flat_ev[lo:hi]
            want = expected[local_idx]
            assert got == want, (
                f"document {local_idx} reads {got} but retrieved {want} -- the segment lists are "
                f"misaligned by position"
            )
            checked += 1
            empties += int(not want)
        assert checked == len(expected), f"only checked {checked} of {len(expected)} documents"
        print(f"2. every document reads its own evidence ({checked} docs)        PASS")
        assert empties == 1, f"expected exactly one empty-evidence document, found {empties}"
        print("3. a document that retrieved nothing keeps its position   PASS")

        # the selector's side: chunks are numbered independently of the token axis, so their
        # document map is a second chance to get the same pairing wrong
        chunk_slot = batch["chunk_slot"]
        valid = chunk_slot >= 0
        chunk_segments = (seg[:, :1] + chunk_slot)[valid]
        chunk_keys = batch["chunk_keys"][valid]
        marker = chunk_keys[:, 0].tolist()
        for chunk_seg, mark in zip(chunk_segments.tolist(), marker):
            # the key's first component was stamped with its document's first evidence token
            lo, hi = int(cu_k[chunk_seg]), int(cu_k[chunk_seg + 1])
            assert hi > lo, f"a chunk landed on segment {chunk_seg}, which has no evidence tokens"
            assert flat_ev[lo] == int(mark), (
                f"chunk on segment {chunk_seg} came from document {int(mark)} but that segment's "
                f"evidence starts with {flat_ev[lo]}"
            )
        print(f"4. every chunk lands on its own document ({len(marker)} chunks)         PASS")

        # padding must go to a trailing pad segment. Its query token is a pad, so nothing supervised
        # can see it; landing it on a real conversation would be silent contamination.
        labels = batch["labels"]
        pad_slots = doc_slot < 0
        if bool(pad_slots.any()):
            for r in range(B):
                if not bool(pad_slots[r].any()):
                    continue
                home = int(seg[r, -1])
                positions = (seg[r] == home).nonzero().flatten()
                assert bool((labels[r][positions] == -100).all()), (
                    "evidence padding was assigned to a segment carrying supervised tokens"
                )
            print("5. evidence padding lands on a trailing pad segment       PASS")
        else:
            print("5. evidence padding lands on a trailing pad segment       SKIP (none in batch)")

        if not torch.cuda.is_available():
            print("   (GPU absent -- skipping the end to end isolation check)")
            return
        _end_to_end(batch, cu)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def _end_to_end(batch, cu):
    """Run a real model over the batch and confirm a document only moves for its own evidence."""
    from modules.model.transformer import TinyMoETransformer

    torch.manual_seed(0)
    P = dict(
        vocab_size=1024, max_seq_len=MAX_LEN, hidden_size=256, intermediate_size=512,
        head_dim=32, num_layers=2, num_heads=8, num_mlp_experts=8, num_attn_experts=1,
        top_k=2, n_loops=2, num_ir_experts=1, num_ir_entries=256, ir_dim=EMBED_DIM,
        dropout=0.0, ple_embeddings_size=32, mtp_num_extra_tokens=0, lm_head_factor=4,
        evidence_port=True,
    )
    model = TinyMoETransformer(**P).to("cuda").to(torch.bfloat16).eval()
    model.set_checkpointing(False, False)
    with torch.no_grad():
        torch.nn.init.normal_(model.moe.shared_evidence.attn.o_proj.weight, std=0.02)

    batch = {k: (v.to("cuda") if torch.is_tensor(v) else v) for k, v in batch.items()}
    cu = cu.to("cuda")
    ev = evidence_from_batch(model, batch, cu)
    with torch.inference_mode():
        base = model(batch["input_ids"], cu_seqlens=cu, max_seqlen=MAX_LEN, skip_mtp=True)

    # perturb the evidence of the LAST conversation of row 0 only
    doc_slot = batch["evidence_doc_slot"]
    target = int(doc_slot[0].max())
    changed = batch["evidence_ids"].clone()
    hit = doc_slot[0] == target
    changed[0][hit] = (changed[0][hit] + 7) % P["vocab_size"]
    perturbed = dict(batch, evidence_ids=changed)
    ev2 = evidence_from_batch(model, perturbed, cu)
    with torch.inference_mode():
        after = model(batch["input_ids"], cu_seqlens=cu, max_seqlen=MAX_LEN, skip_mtp=True, evidence=ev2)
    with torch.inference_mode():
        before = model(batch["input_ids"], cu_seqlens=cu, max_seqlen=MAX_LEN, skip_mtp=True, evidence=ev)

    seg = _segment_ids(cu, *batch["input_ids"].shape, "cuda")
    own = seg[0] == (int(seg[0, 0]) + target)
    others = (seg[0] < int(seg[0, 0]) + target) & (batch["labels"][0] != -100)
    moved = (after[0][own] - before[0][own]).abs().max().item()
    leaked = (after[0][others] - before[0][others]).abs().max().item() if bool(others.any()) else 0.0
    assert moved > 0.0, "the document did not move when its own evidence changed"
    assert leaked == 0.0, f"an earlier document moved when another's evidence changed ({leaked})"
    print(f"6. end to end: only the owning document moves (own {moved:.4f}, leak {leaked})  PASS")
    # and the whole thing is actually reading: attaching the corpus has to differ from not attaching
    # it, or assertions 1-5 would be describing a pairing nothing consumes
    assert not torch.equal(base, before), "attaching the corpus changed nothing at all"


if __name__ == "__main__":
    main()
