"""The evidence corpus round trip: what the builder wrote is what the document reads.

Flash pairs the query and evidence sides **by position**, so the only thing standing between a
document and another document's passage is that the two segment lists are the same length in the
same order. Nothing in the forward can check that -- a shifted list produces a perfectly finite loss
that is learning the wrong association -- so it is checked here, on a corpus small enough to state
the expected answer by hand.

Nine assertions:

1. **The two segment lists have the same length.** One evidence segment per query segment, including
   the empty ones.
2. **Every document reads its OWN evidence**, matched by token id. This is the assertion that
   catches an off-by-one in the pairing.
3. **A document that retrieved nothing gets a zero length segment in its own position**, rather than
   being skipped -- skipping it would shift every later document by one.
4. **Chunks land on their document too**, on the selector's side, which numbers them independently.
5. **Gold flags line up with chunk_keys/chunk_slot row by row** -- the same alignment as (4), one
   axis further: a document's chunks can disagree with each other on gold, so each chunk carries its
   own within-document index as a second marker rather than reusing (4)'s per-document one.
6. **Evidence padding lands on a trailing pad segment**, never on a real conversation, so no
   supervised token can attend to another row's leftovers.
7. **condition_ids matches each document's own condition across its whole query-side span**
   (including its BOS and separator pad, which belong to the same segment).
8. **answerable_ids is carried per document and is not a relabelling of the condition** -- the
   fixture's first row is `gold` with a refusal target, which is the SQuAD v2 case that makes it a
   column of its own, and the assertion refuses to pass if that row is ever removed.
9. **A corpus missing ``.evgold``/``.cond``/``.ans`` still loads**, with all three columns simply
   absent from every batch, rather than crashing -- the compatibility path for older corpora.
10. **The loss weight floor caps a short conversation's per-token weight at ``1/floor``** while
   leaving a conversation already at or past the floor at exactly ``1/n_supervised``.

All ten are pure index arithmetic (no model, no CUDA call) and run anywhere; only the end-to-end
isolation check at the bottom needs a GPU, and it is printed as its own final check rather than
numbered above, since it runs only when CUDA is available.
"""
import os, sys, shutil, tempfile
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from types import SimpleNamespace

import numpy as np
import torch

from modules.data.evidence_dataset import EvidenceDataset, evidence_from_batch, EMBED_DIM
from modules.model.attention import cu_seqlens_from_doc_ids, _segment_ids
from modules.model.evidence import evidence_cu_seqlens
from scripts.prepare_evidence_data import CONDITIONS, EvidenceWriter

BOS, PAD = 0, 1
# small enough that these five documents pack into TWO rows, which is what makes the batch's evidence
# widths unequal and therefore makes row padding exist at all -- assertion 6 has nothing to check on
# a single-row batch
MAX_LEN = 16

# (prompt tokens, evidence tokens, chunk lengths, per-chunk gold flags, condition name). Token ranges
# are disjoint per document so an evidence token identifies the document it belongs to on sight --
# which is what assertion 2 reads. Every entry of CONDITIONS appears at least once, and the prompt
# token counts (3, 2, 4, 3, 2) deliberately straddle assertion 9's floor (3): two documents sit
# exactly at it, one above, two below.
# the last column is `answerable`, and the first row is why it cannot be read off the condition:
# a natively unanswerable question is built under `gold` like any other and still abstains
DOCS = [
    ([10, 11, 12], [100, 101], [2], [1], "gold", 0),
    ([20, 21], [], [], [], "none", 1),        # retrieved nothing: the zero length segment case
    ([30, 31, 32, 33], [300, 301, 302], [2, 1], [1, 0], "mixed", 1),
    ([40, 41, 42], [400, 401, 402, 403], [4], [0], "distractors", 0),
    ([50, 51], [500, 501, 502], [3], [1], "many", 1),
]


def build_corpus(data_dir, split="evidence_train"):
    state = {}
    writer = EvidenceWriter(data_dir, split, state)
    for tokens, ev, chunk_lens, chunk_gold, condition, answerable in DOCS:
        ids = [BOS] + tokens
        mask = [0] + [1] * len(tokens)
        ev_chunk = []
        for chunk_idx, length in enumerate(chunk_lens):
            ev_chunk.extend([chunk_idx] * length)
        assert len(ev_chunk) == len(ev), "the fixture's chunk lengths do not cover its evidence"
        assert len(chunk_gold) == len(chunk_lens), "the fixture's gold flags do not cover its chunks"
        keys = np.zeros((len(chunk_lens), EMBED_DIM), dtype=np.float32)
        for chunk_idx in range(len(chunk_lens)):
            # first component identifies the chunk's document, so assertion 4 can read it back;
            # second identifies which of that document's OWN chunks this is, which assertion 5 needs
            # because a document's chunks can disagree with each other on gold (see "mixed" above)
            keys[chunk_idx, 0] = ev[0] if ev else 0.0
            keys[chunk_idx, 1] = chunk_idx
        writer.write(ids, mask, ev, ev_chunk, keys, chunk_gold,
                     CONDITIONS.index(condition), answerable)
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
        assert dataset.has_condition_labels, "the fixture wrote .evgold/.cond, they should be found"
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
        # which fixture document each conversation slot holds, in packing order, alongside which
        # ROW it packed into -- assertion 7 needs the row as well as the segment
        expected = [ev for _, ev, _, _, _, _ in DOCS]
        conv_segments, conv_rows = [], []
        for r in range(B):
            n_conv = int((doc_slot[r] >= 0).any()) and int(doc_slot[r].max()) + 1
            base = int(seg[r, 0])
            conv_segments.extend(base + j for j in range(n_conv))
            conv_rows.extend([r] * n_conv)

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

        # gold flags ride the identical [M] axis as (4)'s marker/chunk_keys, so the same flattening
        # applies -- only the lookup (per document AND per that document's own chunk index) is new
        assert "chunk_gold" in batch, "the batch carries no chunk_gold even though .evgold exists"
        chunk_gold_flat = batch["chunk_gold"][valid].tolist()
        chunk_idx_marker = chunk_keys[:, 1].tolist()
        gold_by_doc = {ev[0]: gold for _, ev, _, gold, _, _ in DOCS if ev}
        assert len(chunk_gold_flat) == len(marker), "chunk_gold and chunk_keys disagree on count"
        for doc_marker, idx_marker, got in zip(marker, chunk_idx_marker, chunk_gold_flat):
            want = gold_by_doc[int(doc_marker)][int(idx_marker)]
            assert got == want, (
                f"chunk {int(idx_marker)} of the document marked {int(doc_marker)} has gold={got}, "
                f"the fixture wrote {want}"
            )
        print(f"5. gold flags line up with chunk_keys/chunk_slot ({len(marker)} chunks)  PASS")

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
            print("6. evidence padding lands on a trailing pad segment       PASS")
        else:
            print("6. evidence padding lands on a trailing pad segment       SKIP (none in batch)")

        # condition_ids is a QUERY side column: it has to hold constant over a document's WHOLE
        # block (its BOS, its supervised tokens and its trailing separator pad all belong to the
        # same query segment), not just the supervised span
        assert "condition_ids" in batch, "the batch carries no condition_ids even though .cond exists"
        condition_ids = batch["condition_ids"]
        checked_cond = 0
        for local_idx, (r, global_seg) in enumerate(zip(conv_rows, conv_segments)):
            cond_name = DOCS[local_idx][4]
            want = CONDITIONS.index(cond_name)
            positions = (seg[r] == global_seg).nonzero().flatten()
            got = condition_ids[r][positions].unique().tolist()
            assert got == [want], (
                f"document {local_idx} (condition {cond_name!r}) reads condition_ids {got} over its "
                f"own segment, want [{want}]"
            )
            checked_cond += 1
        assert checked_cond == len(DOCS), f"only checked {checked_cond} of {len(DOCS)} documents"
        print(f"7. condition_ids matches each document's own condition ({checked_cond} docs)  PASS")

        # answerable_ids rides the same axis, and the fixture's first document is the case that
        # makes it a separate column at all: condition `gold`, target a refusal. A reader that
        # derived answerability from the condition would get that row backwards.
        assert "answerable_ids" in batch, "the batch carries no answerable_ids even though .ans exists"
        answerable_ids = batch["answerable_ids"]
        for local_idx, (r, global_seg) in enumerate(zip(conv_rows, conv_segments)):
            want = DOCS[local_idx][5]
            positions = (seg[r] == global_seg).nonzero().flatten()
            got = answerable_ids[r][positions].unique().tolist()
            assert got == [want], (
                f"document {local_idx} (condition {DOCS[local_idx][4]!r}) reads answerable_ids "
                f"{got} over its own segment, want [{want}]"
            )
        gold_row = DOCS[0]
        assert gold_row[4] == "gold" and gold_row[5] == 0, (
            "the fixture no longer contains an unanswerable row under the gold condition -- "
            "assertion 8 would pass without testing the case it exists for"
        )
        print(f"8. answerable_ids is carried per document and is not the condition ({len(DOCS)} docs)  PASS")

        # a corpus predating .evgold/.cond has to load anyway, just without the two columns --
        # simulated by building the same fixture and then removing the two sidecar files, mirroring
        # the corpus that is already on disk
        no_labels_dir = tempfile.mkdtemp(prefix="evidence_corpus_nolabels_")
        try:
            build_corpus(no_labels_dir)
            os.remove(os.path.join(no_labels_dir, "evidence_train.evgold"))
            os.remove(os.path.join(no_labels_dir, "evidence_train.cond"))
            os.remove(os.path.join(no_labels_dir, "evidence_train.ans"))
            legacy = EvidenceDataset(
                no_labels_dir, tokenizer, batch_size=2, max_length=MAX_LEN, split="evidence_train",
                num_mtp_tokens=1, shuffle=False,
            )
            assert not legacy.has_condition_labels, "should have detected the missing sidecar files"
            legacy_batches = list(iter(legacy))
            assert legacy_batches, "the legacy dataset yielded nothing"
            assert "condition_ids" not in legacy_batches[0], (
                "a corpus with no .cond file must not carry condition_ids"
            )
            assert "chunk_gold" not in legacy_batches[0], (
                "a corpus with no .evgold file must not carry chunk_gold"
            )
            assert "evidence_ids" in legacy_batches[0], (
                "the legacy corpus still has real evidence -- only the two new columns should be gone"
            )
            print("9. a corpus missing .evgold/.cond/.ans loads and omits the labels    PASS")
        finally:
            shutil.rmtree(no_labels_dir, ignore_errors=True)

        # the weight floor: 1/max(n_supervised, floor). floor=3 sits exactly at two documents' own
        # token count (3), above two more (2, 2) and below the last (4) -- the fixture was picked so
        # this single floor value exercises "below", "at" and "above" all at once. Packing decisions
        # never depend on the floor, so this reuses `seg`/`conv_segments` computed from the
        # default-floor dataset above rather than re-deriving them.
        floor = 3
        floored = EvidenceDataset(
            tmp, tokenizer, batch_size=2, max_length=MAX_LEN, split="evidence_train",
            num_mtp_tokens=1, shuffle=False, loss_weight_floor_tokens=floor,
        )
        floored_weights = list(iter(floored))[0]["loss_weights"]
        checked_weight = 0
        for local_idx, (r, global_seg) in enumerate(zip(conv_rows, conv_segments)):
            n_supervised = len(DOCS[local_idx][0])
            want = 1.0 / max(n_supervised, floor)
            positions = (seg[r] == global_seg).nonzero().flatten()
            row_weights = floored_weights[r][positions]
            nonzero = row_weights[row_weights > 0]
            assert nonzero.numel() == n_supervised, (
                f"document {local_idx} has {nonzero.numel()} nonzero weight(s), want {n_supervised}"
            )
            assert torch.allclose(nonzero, torch.full_like(nonzero, want), atol=1e-6), (
                f"document {local_idx} (n_supervised={n_supervised}, floor={floor}) has weight(s) "
                f"{nonzero.tolist()}, want {want}"
            )
            checked_weight += 1
        assert checked_weight == len(DOCS), f"only checked {checked_weight} of {len(DOCS)} documents"
        print(f"10. the loss weight floor caps short conversations at 1/floor ({checked_weight} docs)  PASS")

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
    print(f"end to end: only the owning document moves (own {moved:.4f}, leak {leaked})  PASS")
    # and the whole thing is actually reading: attaching the corpus has to differ from not attaching
    # it, or the assertions above would be describing a pairing nothing consumes
    assert not torch.equal(base, before), "attaching the corpus changed nothing at all"


if __name__ == "__main__":
    main()
