"""GPU-free checks for the SFT data path (PLAN.md Step 12).

Covers the three things that are silently wrong-but-plausible if they break:

  1. **Loss masking.** ``modules/data/chat.py`` must supervise assistant content and its
     terminating EOS and nothing else. A mask that is one token off, or that leaks the user turn,
     produces a perfectly normal-looking loss curve for a model being trained to parrot prompts.
  2. **Packing/labels.** ``modules/data/sft_dataset.py`` must turn that mask into ``-100`` labels,
     never split a conversation across rows, and give each conversation its own attention segment.
  3. **Order and resume.** The per-epoch permutation has to be identical in every worker and a
     pure function of ``(seed, epoch)``, and the checkpointed position has to be one that resumes
     without skipping or repeating a conversation -- the same guarantee
     ``tests/test_dataset_resume.py`` checks for the pretraining dataset.

Plus a check that the corpus builder in ``scripts/prepare_sft_data.py`` honours the smoltalk2
holdout, the length filter and the train/val split, driven by synthetic in-memory sources exactly
as ``tests/test_prepare_data.py`` drives ``prepare_data.run_phase`` -- no Hub calls, no torch.nn,
no transformer_engine.

Run: `python tests/test_sft_dataset.py` (works anywhere, no GPU needed).
"""
import os
import sys
import shutil
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from modules.data import abstention
from modules.data.chat import ASSISTANT_TOKEN, SYS_BEGIN_TOKEN, SYS_END_TOKEN, USER_TOKEN, ChatTemplate
from modules.data.sft_dataset import SFTDataset
from scripts.prepare_sft_data import build_corpus, render_gsm8k, render_squad_v2


CONTROL_TOKENS = {
    USER_TOKEN: 900,
    ASSISTANT_TOKEN: 901,
    SYS_BEGIN_TOKEN: 902,
    SYS_END_TOKEN: 903,
}


class MockTokenizer:
    """char -> ord(char) + a handful of control tokens, so a test can read ids back as text.

    Ordinary characters map into 0..255 (well clear of the control ids above), which keeps every
    assertion below expressible as "these ids decode to this substring".
    """
    bos_token_id = 500
    eos_token_id = 501
    pad_token_id = 501  # pad == eos, matching the real tokenizer's quirk (see CLAUDE.md)

    def __call__(self, texts, add_special_tokens=False, truncation=False):
        if isinstance(texts, str):
            return {"input_ids": [ord(c) for c in texts]}
        return {"input_ids": [[ord(c) for c in t] for t in texts]}

    def convert_tokens_to_ids(self, token):
        return CONTROL_TOKENS.get(token, -1)


def decode(ids):
    return "".join(chr(i) for i in ids if i < 500)


def test_chat_template_masking():
    template = ChatTemplate(MockTokenizer())
    ids, mask = template.encode([
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "yo"},
        {"role": "user", "content": "more"},
        {"role": "assistant", "content": "ok"},
    ])

    assert len(ids) == len(mask), "mask must be parallel to ids"
    # exact structure: bos, <sys>sys</sys>, <user>hi, <asst>yo<eos>, <user>more, <asst>ok<eos>
    expected = (
        [500, 902] + [ord(c) for c in "sys"] + [903]
        + [900] + [ord(c) for c in "hi"]
        + [901] + [ord(c) for c in "yo"] + [501]
        + [900] + [ord(c) for c in "more"]
        + [901] + [ord(c) for c in "ok"] + [501]
    )
    assert ids == expected, f"unexpected id sequence: {ids}"

    supervised = [i for i, m in zip(ids, mask) if m]
    assert decode(supervised) == "yook", f"only assistant content should be supervised, got {decode(supervised)!r}"
    # both terminating EOS ids supervised, no control token ever supervised
    assert supervised.count(501) == 2, "each assistant turn's terminating EOS must be supervised"
    for control in (500, 900, 901, 902, 903):
        assert control not in supervised, f"control token {control} must never be supervised"
    print("  chat template masking: OK")


def test_chat_template_rejects():
    template = ChatTemplate(MockTokenizer())
    # tool turns -> dropped whole rather than mangled into a user turn
    assert template.encode([
        {"role": "user", "content": "q"},
        {"role": "tool", "content": "{}"},
        {"role": "assistant", "content": "a"},
    ]) is None
    # no assistant turn -> nothing to supervise
    assert template.encode([{"role": "user", "content": "q"}]) is None
    # empty assistant turn -> nothing to supervise
    assert template.encode([
        {"role": "user", "content": "q"}, {"role": "assistant", "content": "   "},
    ]) is None
    # prompt encoding ends with the assistant marker so sampling starts where training did
    prompt = template.encode_prompt([{"role": "user", "content": "q"}])
    assert prompt[-1] == CONTROL_TOKENS[ASSISTANT_TOKEN]
    print("  chat template rejection rules: OK")


def write_corpus(data_dir, split, docs):
    """docs: list of (ids, mask). Writes the bin/idx/mask triple SFTDataset reads."""
    offsets, flat_ids, flat_mask = [0], [], []
    for ids, mask in docs:
        flat_ids.extend(ids)
        flat_mask.extend(mask)
        offsets.append(len(flat_ids))
    np.asarray(flat_ids, dtype=np.uint16).tofile(os.path.join(data_dir, f"{split}.bin"))
    np.asarray(offsets, dtype=np.uint64).tofile(os.path.join(data_dir, f"{split}.idx"))
    np.asarray(flat_mask, dtype=np.uint8).tofile(os.path.join(data_dir, f"{split}.mask"))


def make_docs(n, length=10):
    """n conversations, each `length` tokens with the last 3 supervised (assistant + EOS)."""
    docs = []
    for i in range(n):
        ids = [500] + [ord("a") + (i % 26)] * (length - 4) + [901, ord("z"), 501]
        mask = [0] * (len(ids) - 2) + [1, 1]
        docs.append((ids, mask))
    return docs


def test_dataset_labels_and_segments():
    tmp = tempfile.mkdtemp()
    try:
        write_corpus(tmp, "sft_train", make_docs(8, length=10))
        dataset = SFTDataset(
            tmp, MockTokenizer(), batch_size=2, max_length=32, split="sft_train",
            num_mtp_tokens=2, shuffle=False,
        )
        batches = list(dataset)
        assert batches, "expected at least one batch"

        for batch in batches:
            input_ids, labels, doc_ids = batch["input_ids"], batch["labels"], batch["document_ids"]
            assert input_ids.shape == labels.shape == doc_ids.shape

            for row in range(input_ids.size(0)):
                ids_row, labels_row = input_ids[row].tolist(), labels[row].tolist()
                for token, label in zip(ids_row, labels_row):
                    # a label is either masked out or exactly the token at that position -- the
                    # dataset must never invent or shift a target
                    assert label == -100 or label == token, "label must equal its own input token"
                supervised = [t for t, l in zip(ids_row, labels_row) if l != -100]
                # each packed conversation contributes exactly its 'z' + EOS
                assert set(supervised) <= {ord("z"), 501}, f"leaked prompt tokens: {set(supervised)}"

                # segment ids: non-decreasing, and every conversation block is contiguous
                segs = doc_ids[row].tolist()
                assert segs == sorted(segs), "document_ids must be non-decreasing across a row"
                assert segs[0] == 0

        # nothing lost: 8 conversations x 2 supervised tokens each
        total_supervised = sum(int((b["labels"] != -100).sum()) for b in batches)
        assert total_supervised == 16, f"expected 16 supervised tokens, got {total_supervised}"
        print("  dataset labels + segments: OK")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_dataset_never_splits_a_conversation():
    tmp = tempfile.mkdtemp()
    try:
        # 12-token blocks (10 + 2 separator) into a 32-token row: two fit, the third must start a
        # new row rather than be cut in half
        write_corpus(tmp, "sft_train", make_docs(6, length=10))
        dataset = SFTDataset(
            tmp, MockTokenizer(), batch_size=8, max_length=32, split="sft_train",
            num_mtp_tokens=2, shuffle=False,
        )
        batch = next(iter(dataset))
        for row in range(batch["input_ids"].size(0)):
            labels_row = batch["labels"][row].tolist()
            ids_row = batch["input_ids"][row].tolist()
            # every supervised 'z' must be immediately followed by a supervised EOS: a conversation
            # cut at a row boundary would lose its terminator, which is what teaches the model to stop
            for i, (token, label) in enumerate(zip(ids_row, labels_row)):
                if label == ord("z"):
                    assert labels_row[i + 1] == 501, "a packed conversation lost its supervised EOS"

        # over-long conversations are dropped, not truncated
        write_corpus(tmp, "sft_long", make_docs(2, length=10) + [([500] + [ord("x")] * 60, [0] * 61)])
        long_dataset = SFTDataset(
            tmp, MockTokenizer(), batch_size=8, max_length=32, split="sft_long",
            num_mtp_tokens=2, shuffle=False,
        )
        kept = sum(int((b["document_ids"].max(dim=1).values + 1).sum()) for b in long_dataset)
        assert kept > 0, "the short conversations should still be emitted"
        flat = torch.cat([b["input_ids"].reshape(-1) for b in long_dataset])
        assert (flat == ord("x")).sum() == 0, "the over-long conversation should have been dropped"
        print("  dataset never splits / drops over-long: OK")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_epoch_permutation_is_deterministic():
    tmp = tempfile.mkdtemp()
    try:
        write_corpus(tmp, "sft_train", make_docs(64))
        a = SFTDataset(tmp, MockTokenizer(), max_length=32, split="sft_train", seed=7, epoch=0)
        b = SFTDataset(tmp, MockTokenizer(), max_length=32, split="sft_train", seed=7, epoch=0)
        c = SFTDataset(tmp, MockTokenizer(), max_length=32, split="sft_train", seed=7, epoch=1)

        order_a, order_b, order_c = (d.document_order(64) for d in (a, b, c))
        # every worker generates this independently, so two instances with the same (seed, epoch)
        # must agree exactly or the shards overlap
        assert np.array_equal(order_a, order_b), "same (seed, epoch) must give the same order"
        assert not np.array_equal(order_a, order_c), "a new epoch must reshuffle"
        assert sorted(order_a.tolist()) == list(range(64)), "the order must be a permutation"

        unshuffled = SFTDataset(tmp, MockTokenizer(), max_length=32, split="sft_train", shuffle=False)
        assert np.array_equal(unshuffled.document_order(64), np.arange(64)), "val split must not shuffle"
        print("  per-epoch permutation determinism: OK")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_resume_position_is_gap_free():
    """The checkpointed position must resume without skipping or repeating a conversation.

    Same guarantee (and same granularity) as tests/test_dataset_resume.py makes for pretraining:
    ``doc_idx`` is reported at yield time, and resuming from ``that + 1`` in a single-worker run
    must reproduce exactly the conversations that had not been handed out yet.
    """
    tmp = tempfile.mkdtemp()
    try:
        docs = make_docs(20)
        write_corpus(tmp, "sft_train", docs)
        kwargs = dict(batch_size=1, max_length=32, split="sft_train", num_mtp_tokens=2, shuffle=False)

        full = SFTDataset(tmp, MockTokenizer(), **kwargs)
        all_batches = list(full)

        # stop after two batches, resume from the reported position + 1 (single worker)
        stop_after = 2
        resume_at = int(all_batches[stop_after - 1]["doc_idx"][0]) + 1
        resumed = SFTDataset(tmp, MockTokenizer(), start_doc_idx=resume_at, **kwargs)
        resumed_batches = list(resumed)

        def supervised_chars(batches):
            out = []
            for batch in batches:
                for row in range(batch["labels"].size(0)):
                    ids = batch["input_ids"][row].tolist()
                    labels = batch["labels"][row].tolist()
                    out.extend(i for i, l in zip(ids, labels) if l != -100)
            return out

        combined = supervised_chars(all_batches[:stop_after]) + supervised_chars(resumed_batches)
        assert combined == supervised_chars(all_batches), (
            "resume dropped or duplicated a conversation: "
            f"{len(combined)} supervised tokens vs {len(supervised_chars(all_batches))}"
        )
        print("  resume position is gap-free: OK")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_renderers():
    import random

    rng = random.Random(0)
    answerable = render_squad_v2(
        {"context": "Paris is the capital of France.", "question": "What is the capital?",
         "answers": {"text": ["Paris"], "answer_start": [0]}}, rng,
    )
    assert answerable[-1]["content"] == "Paris"

    unanswerable = render_squad_v2(
        {"context": "Paris is the capital of France.", "question": "How many people live there?",
         "answers": {"text": [], "answer_start": []}}, rng,
    )
    # an empty reference answer must become a *recognizable* abstention, not an empty target --
    # the acceptance metric (abstention precision/recall) can only count what it can identify
    assert unanswerable[-1]["content"] in abstention.ABSTENTIONS_PASSAGE
    assert abstention.is_abstention(unanswerable[-1]["content"])
    assert not abstention.is_abstention("Paris")

    gsm = render_gsm8k(
        {"question": "q?", "answer": "Step? ** 48/2 = <<48/2=24>>24 clips.\n#### 24"}, rng,
    )
    solution = gsm[-1]["content"]
    assert "<<" not in solution and "####" not in solution, "calculator/answer markup must be stripped"
    assert solution.endswith("The answer is 24."), solution
    print("  squad_v2 / gsm8k renderers: OK")


def fake_source(conversations, holdout_keys=None):
    """A generator_factory over an in-memory conversation list, split across two 'files'."""
    holdout_keys = holdout_keys or {}
    half = max(len(conversations) // 2, 1)
    files = [conversations[:half], conversations[half:]]

    def factory(start_file_idx, start_row_idx):
        for file_idx in range(start_file_idx, len(files)):
            rows = files[file_idx]
            row_start = start_row_idx if file_idx == start_file_idx else 0
            for row_idx in range(row_start, len(rows)):
                conversation = rows[row_idx]
                key = holdout_keys.get(id(conversation))
                yield file_idx, row_idx, conversation, key

    return factory


def test_build_corpus():
    tmp = tempfile.mkdtemp()
    try:
        template = ChatTemplate(MockTokenizer())
        keep = [[{"role": "user", "content": f"q{i}"}, {"role": "assistant", "content": f"a{i}"}]
                for i in range(40)]
        excluded = [{"role": "user", "content": "seen"}, {"role": "assistant", "content": "before"}]
        too_long = [{"role": "user", "content": "x" * 200}, {"role": "assistant", "content": "y"}]
        tool_call = [{"role": "user", "content": "q"}, {"role": "tool", "content": "{}"},
                     {"role": "assistant", "content": "a"}]

        conversations = keep + [excluded, too_long, tool_call]
        entries = [{
            "key": "mock",
            "weight": 1.0,
            "generator_factory": fake_source(conversations, {id(excluded): "HOLDOUT"}),
        }]

        state = build_corpus(
            entries, target_tokens=10_000, template=template, data_dir=tmp,
            state_path=os.path.join(tmp, "_state.json"), holdout_hashes={"HOLDOUT"},
            max_doc_tokens=50, val_fraction=0.25, seed=3, encode_batch_size=8,
        )

        skipped = state["skipped"]["mock"]
        assert skipped["holdout"] == 1, f"the pretrained conversation must be excluded: {skipped}"
        assert skipped["too_long"] == 1, f"the over-long conversation must be dropped: {skipped}"
        assert skipped["unrenderable"] == 1, f"the tool-call conversation must be dropped: {skipped}"

        total_docs = sum(s["doc_count"] for s in state["splits"].values())
        assert total_docs == len(keep), f"expected {len(keep)} kept conversations, got {total_docs}"
        assert state["splits"]["sft_val"]["doc_count"] > 0, "val split should not be empty at 25%"

        for split in ("sft_train", "sft_val"):
            idx = np.fromfile(os.path.join(tmp, f"{split}.idx"), dtype=np.uint64)
            bin_bytes = os.path.getsize(os.path.join(tmp, f"{split}.bin"))
            mask_bytes = os.path.getsize(os.path.join(tmp, f"{split}.mask"))
            assert np.all(np.diff(idx) >= 0), f"{split}.idx not monotonic"
            assert int(idx[-1]) * 2 == bin_bytes, f"{split}.idx/.bin out of sync"
            assert int(idx[-1]) == mask_bytes, f"{split}.mask is not one byte per token"

        # and the written corpus is actually readable by the dataset, end to end
        dataset = SFTDataset(tmp, MockTokenizer(), batch_size=2, max_length=64,
                             split="sft_train", num_mtp_tokens=2, shuffle=False)
        assert sum(int((b["labels"] != -100).sum()) for b in dataset) > 0
        print("  build_corpus holdout/length/split bookkeeping: OK")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    print("=== SFT data path (PLAN.md Step 12) ===")
    test_chat_template_masking()
    test_chat_template_rejects()
    test_dataset_labels_and_segments()
    test_dataset_never_splits_a_conversation()
    test_epoch_permutation_is_deterministic()
    test_resume_position_is_gap_free()
    test_renderers()
    test_build_corpus()
    print("all SFT data tests passed")
