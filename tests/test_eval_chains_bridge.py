"""Row bookkeeping of the written bridge read in ``scripts/eval_chains.py --generate-bridge``.

The read scores each candidate after the generated bridge, so the scored row and its positions
must move with the prefix. This checks, in pure Python:

1. ``chains.scored_row`` with no prefix is the plain answer-after-prompt row, and a prefix shifts
   every scored position by its length with each position predicting the candidate token after it;
2. ``chains.bridge_candidate_ids`` on a fake whitespace tokenizer: an empty body scores the bare
   candidate, a body puts the candidate after it with its period and EOS, and a boundary merge
   falls back to the candidate tokenized alone;
3. with the pruned tokenizer, when the generated text is exactly the trained bridge, prompt plus
   prefix plus the gold candidate is the training row token for token, and the scored targets are
   the supervised tail after the marker; a wrong intermediate entity in the bridge changes the
   prefix ids only, the candidate tail stays the gold bridge's.

No GPU; check 3 needs the tokenizer (``utils.TOKENIZER_DIR``) and is skipped without it.
"""
import os, sys, random
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.data import chains
from utils import TOKENIZER_DIR


class WordTokenizer:
    """Splits on spaces, a leading space kept on each word; ``merge`` glues a pair into one id."""

    def __init__(self, merge=None):
        self.vocab = {}
        self.merge = merge

    def _id(self, piece):
        return self.vocab.setdefault(piece, len(self.vocab) + 10)

    def __call__(self, text, add_special_tokens=False):
        pieces, word = [], ""
        for ch in text:
            if ch == " " and word:
                pieces.append(word)
                word = ""
            word += ch
        if word:
            pieces.append(word)
        if self.merge:
            joined = "".join(pieces)
            if self.merge in joined:
                at = joined.index(self.merge)
                pieces = [joined[:at], self.merge, joined[at + len(self.merge):]]
                pieces = [p for p in pieces if p]
        return {"input_ids": [self._id(p) for p in pieces]}


def main():
    # 1. scored rows
    prompt, cand = [1, 2, 3, 4], [7, 8, 0]
    row, pos, tgt = chains.scored_row(prompt, [], cand)
    assert row == prompt + cand and tgt == cand
    assert pos == [3, 4, 5] and all(row[p + 1] == t for p, t in zip(pos, tgt))
    prefix = [20, 21, 22]
    row, pos, tgt = chains.scored_row(prompt, prefix, cand)
    assert row == prompt + prefix + cand and tgt == cand
    assert pos == [6, 7, 8] and all(row[p + 1] == t for p, t in zip(pos, tgt))
    print("1. scored_row: no prefix is the plain row, a prefix shifts the scored positions  PASS")

    # 2. candidate ids on a fake tokenizer
    tok, eos = WordTokenizer(), 0
    p, c = chains.bridge_candidate_ids(tok, "", "Kiva Dorn", eos)
    assert p == [] and c == tok("Kiva Dorn.")["input_ids"] + [eos]
    body = "Step 1: Port Vala. Answer:"
    p, c = chains.bridge_candidate_ids(tok, body, "Kiva Dorn", eos)
    assert p == tok(body)["input_ids"]
    assert p + c[:-1] == tok(f"{body} Kiva Dorn.")["input_ids"] and c[-1] == eos
    assert c[0] == tok(" Kiva")["input_ids"][0]
    merged = WordTokenizer(merge=": K")
    p, c = chains.bridge_candidate_ids(merged, body, "Kiva Dorn", eos)
    assert p == merged(body)["input_ids"] and c == merged(" Kiva Dorn.")["input_ids"] + [eos]
    print("2. bridge_candidate_ids: empty body, joint tokenization, boundary merge fallback  PASS")

    # 3. the real tokenizer: the trained bridge reproduces the training row
    if not os.path.isdir(TOKENIZER_DIR):
        print(f"SKIP: no tokenizer at {TOKENIZER_DIR}, check 3 not run")
        return
    from transformers import AutoTokenizer
    from modules.data.chat import ChatTemplate
    from scripts import prepare_chain_data as pcd
    from scripts.prepare_evidence_data import evidence_prompt

    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_DIR)
    template = ChatTemplate(tokenizer)
    _, held = chains.split_compositions(42)
    rng = random.Random(3)
    qs = [chains.sample_question(rng, held, hops=h) for h in (1, 2, 3) for _ in range(40)]
    convs = [pcd.conversation(q, bridge=True) for q in qs]
    for q, (ids, mask) in zip(qs, template.encode_batch(convs)):
        prompt = template.encode_prompt([{"role": "user", "content": evidence_prompt(q.question)}])
        text = chains.bridge_answer(q)
        prefix_text, rest = chains.split_bridge(text)
        assert rest == f"{q.answer}."
        for k, cand in enumerate(q.candidates):
            p, c = chains.bridge_candidate_ids(tokenizer, prefix_text[:-1], cand, template.eos_id)
            if k == 0:
                gold_prefix = p
            assert p == gold_prefix, "the prefix ids must not depend on the candidate"
        p, c = chains.bridge_candidate_ids(tokenizer, prefix_text[:-1], q.answer, template.eos_id)
        row, pos, tgt = chains.scored_row(prompt, p, c)
        assert row == ids, f"scored row differs from the training row for {text!r}"
        supervised = [t for t, m in zip(ids, mask) if m]
        assert supervised[-len(tgt):] == tgt
        assert all(row[x + 1] == t for x, t in zip(pos, tgt))
        assert not any(mask[:len(prompt)]) and all(mask[len(prompt):])
        if q.hops >= 2:
            # a wrong intermediate changes the prefix ids only, never the candidate tail
            wrong = prefix_text[:-1].replace(q.entities[1], "Zorvath Quellin")
            assert wrong != prefix_text[:-1]
            wp, wc = chains.bridge_candidate_ids(tokenizer, wrong, q.answer, template.eos_id)
            assert wp != p and wc == c, f"candidate tail moved with the bridge for {text!r}"
            assert chains.bridge_hits(wrong, q.entities)[0] is False
    print(f"3. {len(qs)} trained bridges: prompt + prefix + gold candidate is the training row, "
          f"a wrong intermediate moves only the prefix  PASS")
    print("all eval chain bridge checks passed")


if __name__ == "__main__":
    main()
