"""Standalone correctness checks for dataset packing, attention masks, and label/MTP shifting.
Runs on CPU (no transformer_engine / flash-attn needed)."""
import os, sys, json, tempfile
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch

from modules.data.dataset import Dataset
from modules.model.attention import cu_seqlens_from_doc_ids


class MockTok:
    """[BOS=1] + one token (=2) per character. pad=0, no eos -> no EOS supervision."""
    pad_token_id = 0
    bos_token_id = None
    eos_token_id = None
    def __call__(self, text, truncation=False, add_special_tokens=True):
        ids = ([1] if add_special_tokens else []) + [2] * len(text)
        return {"input_ids": ids}
    def apply_chat_template(self, record, tokenize=False):
        return str(record)


class MockTokEos(MockTok):
    """same but with a distinct eos id -> EOS-terminated docs with supervision."""
    eos_token_id = 3


def make_dataset(records, max_length, num_mtp, batch_size=1, tok=None):
    d = tempfile.mkdtemp()
    root = os.path.join(d, "root"); os.makedirs(root)
    with open(os.path.join(root, "a.jsonl"), "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps({"text": r}) + "\n")
    cfg = os.path.join(d, "cfg.json")  # outside root: FileIterator rglobs *.json inside roots
    with open(cfg, "w") as f:
        json.dump({"pretrain": [{"root": root, "column": "text"}]}, f)
    return Dataset(tok or MockTok(), batch_size=batch_size, max_length=max_length,
                   mode="pretrain", config_path=cfg, num_mtp_tokens=num_mtp)


def brute_block_mask(doc_ids):
    B, S = doc_ids.shape
    out = torch.zeros(B, S, S, dtype=torch.bool)
    for b in range(B):
        for i in range(S):
            for j in range(S):
                if doc_ids[b, i] == doc_ids[b, j] and j <= i:
                    out[b, i, j] = True
    return out


def mask_from_cu(cu, B, S):
    """rebuild [B,S,S] same-segment causal mask from cu_seqlens (flattened B*S)."""
    seg = torch.zeros(B * S, dtype=torch.long)
    internal = cu[1:-1].long()
    seg[internal] = 1
    seg = torch.cumsum(seg, 0).view(B, S)
    same = seg[:, :, None] == seg[:, None, :]
    causal = torch.tril(torch.ones(S, S, dtype=torch.bool))
    return same & causal


def test_packing_and_labels():
    # A: 4 chars -> 5 tok; B: 5 chars -> 6 tok; C: 4 chars -> 5 tok, gets split across rows
    ds = make_dataset(["aaaa", "bbbbb", "cccc"], max_length=16, num_mtp=2, batch_size=1)
    batches = list(iter(ds))
    b0 = batches[0]
    ii = b0["input_ids"][0].tolist()
    did = b0["document_ids"][0].tolist()
    lab = b0["labels"][0].tolist()

    assert b0["input_ids"].shape == (1, 16), b0["input_ids"].shape
    # segments: docA block 7 (5+2pad), docB block 8 (6+2pad), then docC's first token fills
    # the last slot (split continues in the next row)
    assert did == [0]*7 + [1]*8 + [2], did
    # labels: continuation tokens only (not first tok of each doc/chunk, not pad)
    keep = [i for i, v in enumerate(lab) if v != -100]
    assert keep == [1, 2, 3, 4, 8, 9, 10, 11, 12], keep
    # kept label values equal the input token at that position (next-token target lives in input_ids)
    for i in keep:
        assert lab[i] == ii[i], (i, lab[i], ii[i])
    # separator pads after A and B; last slot holds docC's first (BOS) token, not padding
    assert ii[5] == 0 and ii[6] == 0 and ii[7] == 1 and ii[15] == 1, ii
    print("[ok] packing + document_ids + label masking")

    # the rest of docC continues in row 2: 4 remaining tokens + 2 separator pads
    b1 = batches[1]
    did1 = b1["document_ids"][0].tolist()
    ii1 = b1["input_ids"][0].tolist()
    lab1 = b1["labels"][0].tolist()
    assert ii1[:6] == [2, 2, 2, 2, 0, 0], ii1
    assert did1[:6] == [0]*6, did1
    # continuation chunk: first token masked (no visible context), rest supervised
    keep1 = [i for i, v in enumerate(lab1) if v != -100]
    assert keep1 == [1, 2, 3], keep1
    print("[ok] document splitting: remainder continues in next row as its own segment")


def test_eos_supervision():
    # with a real eos id the first separator after each doc is EOS and is supervised
    ds = make_dataset(["aaaa", "bbbbb"], max_length=16, num_mtp=2, batch_size=1, tok=MockTokEos())
    b = list(iter(ds))[0]
    ii = b["input_ids"][0].tolist()
    lab = b["labels"][0].tolist()
    # docA: tokens 0..4, eos at 5, pad at 6; docB: 7..12, eos at 13, pad at 14
    assert ii[5] == 3 and ii[6] == 0, ii
    assert lab[5] == 3, lab     # EOS supervised -> model learns to terminate documents
    assert lab[6] == -100, lab  # second separator pad unsupervised
    assert ii[13] == 3 and lab[13] == 3 and lab[14] == -100, (ii, lab)
    print("[ok] EOS separator after each document is supervised, trailing pads are not")


def test_no_cross_doc_label():
    # the last real token of doc A must never be trained to predict the first token of doc B.
    ds = make_dataset(["aaaa", "bbbbb"], max_length=16, num_mtp=2, batch_size=1)
    b = list(iter(ds))[0]
    lab = b["labels"][0]
    # main loss target for logits[i] is lab[i+1]; doc B starts at pos 7 -> lab[7] must be ignored
    assert lab[7].item() == -100
    print("[ok] no cross-document next-token supervision")


def test_cu_seqlens_matches_blockmask():
    for did in [
        torch.tensor([[0]*7 + [1]*8 + [2]]),
        torch.tensor([[0]*4 + [1]*4, [0]*5 + [1]*3]),   # two rows, ids collide across rows
        torch.tensor([[0]*16]),                          # single full doc
    ]:
        B, S = did.shape
        cu, maxlen = cu_seqlens_from_doc_ids(did)
        ref = brute_block_mask(did)
        got = mask_from_cu(cu, B, S)
        assert torch.equal(ref, got), (did, cu)
        # max_seqlen must be the true longest segment
        seg_lens = (cu[1:] - cu[:-1])
        assert maxlen == int(seg_lens.max()), (maxlen, seg_lens)
    print("[ok] cu_seqlens_from_doc_ids == brute-force block-causal mask (incl. cross-row seam)")


def test_full_length_doc():
    # a doc exactly max_length long -> single segment, no padding, first tok ignored
    ds = make_dataset(["a" * 15], max_length=16, num_mtp=2, batch_size=1)  # 15 chars -> 16 tok
    b = list(iter(ds))[0]
    did = b["document_ids"][0].tolist()
    assert did == [0]*16, did
    lab = b["labels"][0].tolist()
    assert lab[0] == -100 and all(v != -100 for v in lab[1:]), lab
    print("[ok] exact-max-length document packs as one full causal segment")


def test_long_doc_split_not_truncated():
    # a doc longer than max_length is split across rows instead of losing its tail
    ds = make_dataset(["a" * 35], max_length=16, num_mtp=2, batch_size=1)  # 36 tok over 16-rows
    batches = list(iter(ds))
    assert len(batches) == 3, len(batches)
    total_doc_tokens = sum(
        int((b["input_ids"][0] != 0).sum()) for b in batches
    )
    assert total_doc_tokens == 36, total_doc_tokens  # nothing discarded
    # rows 1 and 2 are full continuation segments, row 3 has the remaining 4 tokens + 2 pads
    assert batches[0]["document_ids"][0].tolist() == [0]*16
    assert batches[1]["document_ids"][0].tolist() == [0]*16
    assert batches[2]["input_ids"][0].tolist()[:6] == [2, 2, 2, 2, 0, 0]
    print("[ok] over-length document split across rows, no tokens discarded")


if __name__ == "__main__":
    test_packing_and_labels()
    test_eos_supervision()
    test_no_cross_doc_label()
    test_cu_seqlens_matches_blockmask()
    test_full_length_doc()
    test_long_doc_split_not_truncated()
    print("\nALL DATASET/MASK CHECKS PASSED")
