"""The post-training mix builder and the chat read, CPU only (no model, no Hub, no tokenizer).

1. shares: defaults sum to 1 after normalisation, an override is applied and renormalised, an unknown
   label or a non positive share is refused, the slice table follows the profile, the merge argument
   list carries every slice with its share;
2. realised shares and the share check from a merge.json;
3. the EOS check: a conversation whose assistant turn ends with a supervised EOS passes; a truncated
   turn, an EOS with mask 0 and a document with a second turn missing its EOS fail; documents with no
   assistant marker (filler text) are not counted;
4. a real merge of two plain synthetic slices through merge_evidence_splits.py (--no-evidence): the
   realised token shares land within 3% of the targets and the EOS check passes on the merged output
   for the chat slice's documents;
5. eval_chat's n-gram and summary helpers and the 40 prompts (two multi-turn, each ending on a user turn).
"""
import os, sys, json, subprocess, tempfile
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from scripts import build_post_sft_mix as m
from scripts import eval_chat

ASSISTANT, EOS, USER, BOS = 5, 1, 7, 0


def write_split(prefix, docs):
    """docs: list of (ids, mask); writes bin idx mask."""
    ids = np.concatenate([np.asarray(i, dtype=np.uint16) for i, _ in docs])
    mask = np.concatenate([np.asarray(k, dtype=np.uint8) for _, k in docs])
    idx = np.concatenate([[0], np.cumsum([len(i) for i, _ in docs])]).astype(np.uint64)
    ids.tofile(prefix + ".bin"); mask.tofile(prefix + ".mask"); idx.tofile(prefix + ".idx")


def chat_doc(n_words, turns=1):
    ids, mask = [BOS], [0]
    for t in range(turns):
        ids += [USER] + [20 + t] * 3 + [ASSISTANT] + [30 + t] * n_words + [EOS]
        mask += [0] * 4 + [0] + [1] * n_words + [1]
    return ids, mask


def text_doc(n):
    return [BOS] + [40] * n + [EOS], [1] * (n + 2)


# 1
sh = m.parse_shares("pilot", [])
assert abs(sum(sh.values()) - 1) < 1e-9 and abs(sh["chat"] - 0.40) < 1e-9 and abs(sh["qa"] - 0.25) < 1e-9
assert abs(sh["bios"] + sh["chains"] - 0.35) < 1e-9
sh2 = m.parse_shares("pilot", ["chat=0.5"])
assert abs(sum(sh2.values()) - 1) < 1e-9 and sh2["chat"] > sh["chat"]
for bad in (["nope=0.1"], ["chat=0"]):
    try:
        m.parse_shares("pilot", bad); raise SystemExit(f"accepted {bad}")
    except ValueError:
        pass
run2 = m.parse_shares("run2", [])
assert set(run2) == {"chat", "qa", "replay"} and abs(run2["replay"] - 0.35) < 1e-9
slices = m.profile_slices("pilot", "OUT", "ROOT", sh)
assert [s.label for s in slices] == ["chat", "qa", "bios", "chains"]
assert slices[0].train == os.path.join("OUT", "postchat_train")
args = m.merge_args("S", 1000, slices, "train")
assert args.count("--slice") == 4 and "chat=OUT/postchat_train:0.400000:1" in args
assert "val" in " ".join(m.merge_args("S", 1000, slices, "val")).replace("inject_val", "val")

# 2
meta = {"slices": [{"label": "chat", "prompt_tokens": 400}, {"label": "qa", "prompt_tokens": 250}, {"label": "bios", "prompt_tokens": 350}]}
real = m.realised_shares(meta)
assert abs(real["chat"] - 0.4) < 1e-9
assert m.share_problems(meta, {"chat": 0.4, "qa": 0.25, "bios": 0.35}) == []
assert len(m.share_problems(meta, {"chat": 0.5, "qa": 0.25, "bios": 0.25})) == 2

# 3
with tempfile.TemporaryDirectory() as d:
    p = os.path.join(d, "good")
    write_split(p, [chat_doc(5), chat_doc(3, turns=2), text_doc(9)])
    r = m.check_eos_supervised(p, ASSISTANT, EOS)
    assert r["documents"] == 3 and r["with_assistant"] == 2 and r["turns"] == 3 and r["bad"] == 0, r
    trunc = chat_doc(5); trunc = (trunc[0][:-1], trunc[1][:-1])
    nomask = chat_doc(5); nomask = (nomask[0], nomask[1][:-1] + [0])
    second = chat_doc(4, turns=2); second = (second[0][:-1], second[1][:-1])   # the second turn loses its EOS
    for name, doc in (("trunc", trunc), ("nomask", nomask), ("second", second)):
        pb = os.path.join(d, name)
        write_split(pb, [chat_doc(2), doc])
        rb = m.check_eos_supervised(pb, ASSISTANT, EOS)
        assert rb["bad"] == 1 and rb["bad_examples"] == [1], (name, rb)

    # a document that ends on a user turn is counted apart, never bad
    user_end = chat_doc(3, turns=1); user_end = (user_end[0] + [USER, 21, 21], user_end[1] + [0, 0, 0])
    pu = os.path.join(d, "userend")
    write_split(pu, [chat_doc(2), user_end])
    ru = m.check_eos_supervised(pu, ASSISTANT, EOS)
    assert ru["bad"] == 0 and ru["ends_non_assistant"] == 1 and ru["with_assistant"] == 2, ru
    assert r["ends_non_assistant"] == 0, r

    # QA source share from a .src sidecar; None without one
    np.array([0, 1, 3, 2, 2, 0], dtype=np.uint8).tofile(os.path.join(d, "qa.src"))
    assert abs(m.qa_source_share(os.path.join(d, "qa")) - 0.5) < 1e-9
    assert m.qa_source_share(os.path.join(d, "none")) is None

    # run2 QA sources: none when the builds are gone, the shuffled copy preferred over the builder's output
    assert m.run2_qa_sources(os.path.join(d, "root")) is None
    P, R = os.path.join(d, "root", "prepared"), os.path.join(d, "root", "prepared_run2")
    os.makedirs(P); os.makedirs(R)
    for base in (os.path.join(P, "evidence_run2_train"),):
        open(base + ".bin", "w").close(); open(base + ".evidx", "w").close()
    assert m.run2_qa_sources(os.path.join(d, "root"))[0].endswith("evidence_run2_train")
    sh_base = os.path.join(R, "run2_qa_shuf_train")
    open(sh_base + ".bin", "w").close(); open(sh_base + ".evidx", "w").close()
    got = m.run2_qa_sources(os.path.join(d, "root"))
    assert got[0].endswith("run2_qa_shuf_train") and got[1].endswith("evidence_run2_val"), got

    # 4
    rng =np.random.RandomState(0)
    chat_docs = [chat_doc(int(rng.randint(5, 40)), turns=int(rng.randint(1, 3))) for _ in range(3000)]
    filler_docs = [text_doc(int(rng.randint(20, 120))) for _ in range(3000)]
    write_split(os.path.join(d, "c_train"), chat_docs)
    write_split(os.path.join(d, "f_train"), filler_docs)
    out = os.path.join(d, "out")
    os.makedirs(out)
    cmd = [sys.executable, os.path.join(ROOT, "scripts", "merge_evidence_splits.py"), "--out-dir", out, "--split", "mix",
           "--target-tokens", "100000", "--seed", "42", "--no-evidence", "--overwrite",
           "--slice", f"chat={os.path.join(d, 'c_train')}:0.4:1", "--slice", f"bios={os.path.join(d, 'f_train')}:0.6:1"]
    res = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    assert res.returncode == 0, res.stdout[-800:] + res.stderr[-800:]
    meta = json.load(open(os.path.join(out, "mix.merge.json")))
    assert m.share_problems(meta, {"chat": 0.4, "bios": 0.6}) == [], m.realised_shares(meta)
    sl = np.fromfile(os.path.join(out, "mix.slice"), dtype=np.uint8)
    rr = m.check_eos_supervised(os.path.join(out, "mix"), ASSISTANT, EOS, doc_ids=np.flatnonzero(sl == 0))
    assert rr["bad"] == 0 and rr["documents"] == int((sl == 0).sum()) and rr["turns"] >= rr["documents"], rr

# 5
assert eval_chat.has_repeated_ngram([1, 2, 3, 4, 9, 1, 2, 3, 4]) is True
assert eval_chat.has_repeated_ngram([1, 2, 3, 4, 5, 6, 7, 8]) is False
assert eval_chat.has_repeated_ngram([1, 2, 3]) is False
s = eval_chat.summarise([{"stopped": True, "new_tokens": 10, "repeat4": False}, {"stopped": False, "new_tokens": 30, "repeat4": True}])
assert s == {"n": 2, "stop_rate": 0.5, "mean_new_tokens": 20.0, "repeat4_rate": 0.5}, s
prompts = eval_chat.load_prompts(eval_chat.DEFAULT_PROMPTS)
assert len(prompts) == 40 and all(p["messages"][-1]["role"] == "user" for p in prompts)
assert sum(len(p["messages"]) > 2 for p in prompts if p["category"] == "multi_turn") == 2
print("test_post_sft_mix: ok")
