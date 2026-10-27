"""The chain generator: what ``modules/data/chains.py`` and ``scripts/prepare_chain_data.py`` write.

A composition reading is only worth anything if the questions cannot be answered without composing,
so this checks the properties the measurement leans on:

1. hops come out at 25 / 50 / 25, distractor counts span 6..14 and every distractor kind occurs;
2. the question names only the start entity, never a bridge or the answer; the answer occurs only in
   the final gold chunk; the decoy chain ends in a different entity; no entity string is a
   substring of another inside one question;
3. candidates contain the answer, hold only answer-type entities and number at least four;
4. the leak detectors (most frequent entity, least frequent entity, the object of the named
   subject, any final relation object, a final relation object whose subject recurs) sit within
   0.05 of chance for two or more hops;
4b. the second-to-last hop picker (``penultimate_fed``, a known bias of the distractor recipe) is
   reported, sits above chance on 2-hop and 3-hop eval questions and equals chance on one hop;
5. held out 2-hop compositions never appear in the train split's 2-hop or 3-hop questions, every
   relation still appears in a seen composition, and 4-hop questions exist only in the hop4 split;
6. ``.evhop`` is nonzero exactly where ``.evgold`` is, with hop h on exactly one chunk per hop, and
   the metadata lines pair with the documents;
7. the assistant answer and the eval prompt agree: ``encode_prompt`` of the user turn is a prefix of
   the training row, so the eval scores the same context training saw;
8. ``EvidenceDataset`` reads the splits back;
8b. the ``--bridge`` build writes the same buffers and sidecars with the written bridge as the
   supervised span and the intermediates in the metadata;
5b. the statistics helpers: a cell at chance passes, a cell above it fails, an unsaturated 1-hop
   curve reads "not readable";
5c. the written bridge form: ``bridge_answer``, ``split_bridge``, ``bridge_hits``, and the plain
   conversation still answers with ``q.answer``.

Needs the pruned tokenizer (``utils.TOKENIZER_DIR``); no GPU.
"""
import os, sys, json, random, shutil, tempfile, hashlib
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from transformers import AutoTokenizer

from modules.data import chains
from modules.data.chat import ChatTemplate
from scripts import prepare_chain_data as pcd
from scripts.prepare_evidence_data import CONDITIONS, EMBED_DIM, evidence_prompt
from utils import TOKENIZER_DIR


class FakeEmbedder:
    """Deterministic unit vectors from the text, so equal texts get equal keys."""

    def encode(self, texts):
        out = np.zeros((len(texts), EMBED_DIM), dtype=np.float32)
        for i, text in enumerate(texts):
            seed = int.from_bytes(hashlib.sha1(text.encode()).digest()[:4], "big")
            v = np.random.RandomState(seed).randn(EMBED_DIM).astype(np.float32)
            out[i] = v / np.linalg.norm(v)
        return out


def read_split(data_dir, split):
    def load(suffix, dtype):
        return np.fromfile(os.path.join(data_dir, f"{split}.{suffix}"), dtype=dtype)
    idx, evkeyidx = load("idx", np.uint64), load("evkeyidx", np.uint64)
    evgold, evhop = load("evgold", np.uint8), load("evhop", np.uint8)
    ids, mask = load("bin", np.uint16), load("mask", np.uint8)
    cond, ans = load("cond", np.uint8), load("ans", np.uint8)
    with open(os.path.join(data_dir, f"{split}.chains.jsonl"), encoding="utf-8") as f:
        meta = [json.loads(line) for line in f]
    docs = []
    for i in range(len(cond)):
        a, b = int(idx[i]), int(idx[i + 1])
        ca, cb = int(evkeyidx[i]), int(evkeyidx[i + 1])
        docs.append({"ids": ids[a:b].tolist(), "mask": mask[a:b].tolist(), "gold": evgold[ca:cb].tolist(),
                     "hop": evhop[ca:cb].tolist(), "cond": CONDITIONS[cond[i]], "ans": int(ans[i]),
                     "meta": meta[i]})
    return docs, len(evgold), len(evhop)


def draw(n, seed, held, **kw):
    rng = random.Random(seed)
    return [chains.sample_question(rng, held, **kw) for _ in range(n)]


def main():
    seen, held = chains.split_compositions(42)

    # relations: typed, at least three templates each, both slots present
    for name, rel in chains.RELATIONS.items():
        assert len(rel.templates) >= 3 and all("{s}" in t and "{o}" in t for t in rel.templates), name
        assert rel.domain in chains.ENTITY_TYPES and rel.range in chains.ENTITY_TYPES
    assert len(chains.compositions(2)) == 14, len(chains.compositions(2))
    assert len(held) >= 2 and held.isdisjoint(seen) and held | seen == set(chains.compositions(2))
    assert {r for t in seen for r in t} == {r for t in chains.compositions(2) for r in t}
    assert chains.split_compositions(42) == (seen, held), "split is not deterministic"
    assert all(not chains.contains_held_out(t, held) for t in chains.seen_compositions(3, held))
    assert len(chains.seen_compositions(3, held)) >= 4
    print(f"0. {len(chains.RELATIONS)} typed relations, {len(held)} of 14 pairs held out, "
          f"{len(chains.seen_compositions(3, held))} seen 3-hop tuples                      PASS")

    # 1. hop mix, distractor counts, kinds
    qs = draw(3000, 1, held)
    hop_share = {h: sum(q.hops == h for q in qs) / len(qs) for h in (1, 2, 3)}
    assert abs(hop_share[1] - 0.25) < 0.04 and abs(hop_share[2] - 0.5) < 0.04 and abs(hop_share[3] - 0.25) < 0.04, hop_share
    counts = {len(q.chunks) - q.hops for q in qs}
    assert counts == set(range(6, 15)), f"distractor counts {sorted(counts)}"
    kinds = {k for q in qs for k in q.chunk_kind}
    assert kinds == set(chains.CHUNK_KINDS), kinds
    print(f"1. hops {hop_share[1]:.2f}/{hop_share[2]:.2f}/{hop_share[3]:.2f}, distractors 6..14, kinds {sorted(kinds)}  PASS")

    # 2. leakage inside one question
    for q in qs:
        low = q.question.lower()
        assert q.entities[0].lower() in low
        assert not any(e.lower() in low for e in q.entities[1:]), q.question
        holders = [c for c in q.chunks if q.answer in c]
        gold_final = [c for c, h in zip(q.chunks, q.chunk_hop) if h == q.hops]
        assert holders == gold_final, (q.question, holders)
        decoy_answers = [o for (s, r, o), kd in zip(q.chunk_facts, q.chunk_kind) if kd == "C"
                         and r == q.composition[-1]]
        assert decoy_answers and all(a != q.answer for a in decoy_answers)
        assert q.chunk_hop.count(0) == len(q.chunks) - q.hops
        assert sorted(h for h in q.chunk_hop if h) == list(range(1, q.hops + 1))
        entities = set()
        for s, r, o in q.chunk_facts:
            entities.update((s, o))
        ents = sorted(e.lower() for e in entities)
        for i, a in enumerate(ents):
            for b in ents[:i] + ents[i + 1:]:
                assert a not in b, (a, b)
        assert len(q.chunks) == len(q.chunk_hop) == len(q.chunk_kind) == len(q.chunk_facts)
    # a decoy start shares a name part with the true start
    shared = sum(
        1 for q in qs
        if any(kd == "C" and s != q.entities[0] and s in [x for x in (f[0] for f in q.chunk_facts)]
               and set(s.lower().split()) & set(q.entities[0].lower().split())
               for (s, _, _), kd in zip(q.chunk_facts, q.chunk_kind))
    )
    assert shared > 0.5 * len(qs), f"decoy starts rarely share a name part ({shared}/{len(qs)})"
    print("2. question names only the start, answer only in the final gold chunk, decoy differs  PASS")

    # 3. candidates
    for q in qs:
        assert q.answer in q.candidates and len(q.candidates) >= chains.MIN_CANDIDATES
        assert len(set(q.candidates)) == len(q.candidates)
        types = chains._types_of(q)
        assert all(types[c] == q.answer_type for c in q.candidates)
        assert set(q.candidates) == {o for s, r, o in q.chunk_facts if r == q.composition[-1]}
        final_kinds = [kd for (s, r, o), kd in zip(q.chunk_facts, q.chunk_kind) if r == q.composition[-1]]
        assert final_kinds.count("gold") >= 1 and final_kinds.count("C") >= 1
    mean_c = {h: np.mean([len(q.candidates) for q in qs if q.hops == h]) for h in (1, 2, 3)}
    print(f"3. candidates are the final relation objects, hold the answer, >= 4 (mean by hops "
          f"{mean_c[1]:.1f}/{mean_c[2]:.1f}/{mean_c[3]:.1f})   PASS")

    # 4. leak detectors
    multi = draw(2000, 2, held, hops=2) + draw(1000, 3, held, hops=3)
    base = chains.shortcut_baselines(multi)
    for name in ("frequent", "infrequent", "named_subject", "final_object", "recurring_subject"):
        assert abs(base[name] - base["random"]) <= 0.05, (name, base)
    one = chains.shortcut_baselines(draw(500, 4, held, hops=1))
    assert one["named_subject"] > 0.9, one
    print("4. shortcut baselines vs chance (hops >= 2): " + ", ".join(
        f"{k} {v:.3f}" for k, v in base.items()) + "   PASS")

    # 4b. the second-to-last hop picker is computed and sits above chance on a small eval set
    eval_qs = pcd.make_split_questions("eval", 0, 42, held, per_hop=300)
    for hops in (2, 3):
        b = chains.shortcut_baselines([q for q in eval_qs if q.hops == hops])
        assert "penultimate_fed" in b and b["penultimate_fed"] > b["random"] + 0.03, (hops, b)
    b1 = chains.shortcut_baselines([q for q in eval_qs if q.hops == 1])
    assert abs(b1["penultimate_fed"] - b1["random"]) < 1e-12, b1
    b2 = chains.shortcut_baselines([q for q in eval_qs if q.hops == 2])
    print(f"4b. penultimate_fed above chance on 2-hop eval: {b2['penultimate_fed']:.3f} vs random "
          f"{b2['random']:.3f}, chance on 1-hop   PASS")

    # 5. held out compositions and the 4-hop split
    train_qs = draw(3000, 5, held)
    assert all(q.composition not in held for q in train_qs if q.hops == 2)
    assert all(not chains.contains_held_out(q.composition, held) for q in train_qs)
    assert {q.hops for q in train_qs} == {1, 2, 3}
    assert {q.composition for q in train_qs if q.hops == 1} == {(n,) for n in chains.RELATIONS}
    ho = [pcd.make_split_questions("heldout_tmpl", 0, 42, held, per_hop=40)]
    assert all(q.hops == 2 and q.composition in held for q in ho[0])
    h4 = pcd.make_split_questions("hop4", 0, 42, held, per_hop=40)
    assert all(q.hops == 4 for q in h4)
    for name in ("train", "val", "eval"):
        got = pcd.make_split_questions(name, 200, 42, held, per_hop=30)
        assert all(q.hops <= 3 for q in got), name
    assert pcd.make_split_questions("train", 20, 42, held)[0].question == \
        pcd.make_split_questions("train", 20, 42, held)[0].question
    print("5. held out pairs absent from train 2/3-hop, 4-hop only in hop4, splits deterministic  PASS")

    # 9. statistics helpers (pure python)
    at_chance = chains.cell_stats([i % 4 == 0 for i in range(400)], [0.25] * 400)
    assert abs(at_chance["acc"] - 0.25) < 1e-9 and abs(at_chance["z_vs_chance"]) < 1e-6
    above = chains.cell_stats([True] * 300 + [False] * 100, [0.25] * 400)
    assert above["z_vs_chance"] > 10
    def cell(hops, kept, stats):
        return dict(stats, split="s", depth=3, hops=hops, kept=kept)
    ok = chains.validity_report([cell(2, 1, at_chance), cell(1, 1, above), cell(1, 0, at_chance)])
    assert ok["valid"] and ok["readable"], ok
    bad = chains.validity_report([cell(2, 1, above), cell(1, 1, above)])
    assert not bad["valid"] and len(bad["failing"]) == 1
    flat = chains.validity_report([cell(2, 1, at_chance), cell(1, 1, at_chance)])
    assert not flat["valid"] and not flat["readable"] and not flat["failing"]
    print("5b. cell statistics and the validity verdict (valid / failing / not readable)  PASS")

    # 5c. the written bridge form
    q1 = draw(1, 6, held, hops=1)[0]
    q3 = draw(1, 7, held, hops=3)[0]
    marker = chains.BRIDGE_ANSWER_MARKER
    assert chains.bridge_answer(q1) == f"{marker} {q1.answer}.", chains.bridge_answer(q1)
    e = q3.entities
    want3 = f"Step 1: {e[1]}. Step 2: {e[2]}. {marker} {e[3]}."
    assert chains.bridge_answer(q3) == want3, chains.bridge_answer(q3)
    for q in (q1, q3):
        text = chains.bridge_answer(q)
        assert text.endswith(f"{marker} {q.answer}.") and text.count(marker) == 1
        at = [text.index(x) for x in q.entities[1:-1]]
        assert at == sorted(at) and all(a < text.index(marker) for a in at)
    prefix, rest = chains.split_bridge(want3)
    assert prefix == f"Step 1: {e[1]}. Step 2: {e[2]}. {marker} " and rest == f"{e[3]}."
    assert chains.split_bridge(f"Step 1: {e[1]}. {marker}") == (f"Step 1: {e[1]}. {marker} ", "")
    assert chains.split_bridge(f"Step 1: {e[1]}.") == (f"Step 1: {e[1]}.", None)
    assert chains.bridge_hits(want3, e) == [True, True]
    assert chains.bridge_hits(f"Step 1: {e[1]}. {marker} {e[3]}.", e) == [True, False]
    assert chains.bridge_hits("anything", q1.entities) == []
    longer = ["Kiva Dorn", "Port Vala", "Lis Amber", "Rho Teln"]
    assert chains.bridge_hits("Step 1: Port Valada. Step 2: Lis Amberly. Answer: Rho Teln.", longer) == [False, False]
    assert chains.bridge_hits("Step 1: Port Vala. Step 2: XLis Amber. Answer: Rho Teln.", longer) == [True, False]
    assert pcd.conversation(q3) == [{"role": "user", "content": evidence_prompt(q3.question)},
                                    {"role": "assistant", "content": q3.answer}]
    assert pcd.conversation(q3, bridge=True)[1]["content"] == want3
    print(f"5c. written bridge: {want3!r}   PASS")

    if not os.path.isdir(TOKENIZER_DIR):
        print(f"SKIP: no tokenizer at {TOKENIZER_DIR}, builder checks 6 to 8 not run")
        return
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_DIR)
    template = ChatTemplate(tokenizer)
    tmp = tempfile.mkdtemp(prefix="chains_")
    try:
        built = {}
        for name, count, per_hop in (("train", 300, 0), ("eval", 0, 20), ("heldout_tmpl", 0, 30), ("hop4", 0, 20)):
            split = f"chains_{name}"
            questions = pcd.make_split_questions(name, count, 42, held, per_hop)
            pcd.build_split(tmp, split, questions, template, FakeEmbedder(), held, batch=64)
            built[split] = (questions, *read_split(tmp, split))

        # 6. sidecar alignment
        for split, (questions, docs, n_gold, n_hop) in built.items():
            assert len(docs) == len(questions) and n_gold == n_hop, split
            for q, d in zip(questions, docs):
                assert [int(g > 0) for g in d["hop"]] == d["gold"], split
                assert d["hop"] == q.chunk_hop and d["cond"] == "mixed" and d["ans"] == 1
                assert d["meta"]["question"] == q.question and d["meta"]["answer"] == q.answer
                assert d["meta"]["chunk_kind"] == q.chunk_kind and d["meta"]["candidates"] == q.candidates
                for h in range(1, q.hops + 1):
                    assert d["hop"].count(h) == 1
        train_docs = built["chains_train"][1]
        assert all(not d["meta"]["held_out"] for d in train_docs)
        assert all(d["meta"]["held_out"] for d in built["chains_heldout_tmpl"][1])
        assert {d["meta"]["hops"] for d in built["chains_hop4"][1]} == {4}
        assert all(d["meta"]["hops"] != 4 for s in ("chains_train", "chains_eval") for d in built[s][1])
        print("6. .evhop is nonzero exactly where .evgold is, one chunk per hop, metadata pairs  PASS")

        # 7. eval prompt is a prefix of the training row, answer + EOS is the supervised span
        for q, d in list(zip(*[built["chains_eval"][0], built["chains_eval"][1]]))[:25]:
            prompt = template.encode_prompt([{"role": "user", "content": evidence_prompt(q.question)}])
            assert d["ids"][:len(prompt)] == prompt, "eval prompt differs from the training prefix"
            assert not any(d["mask"][:len(prompt)])
            supervised = [t for t, m in zip(d["ids"], d["mask"]) if m]
            assert supervised == tokenizer(q.answer, add_special_tokens=False)["input_ids"] + [template.eos_id]
        print("7. eval prompt is the training prefix, supervised span is answer + EOS          PASS")

        # 8. dataset
        from modules.data.evidence_dataset import EvidenceDataset
        for split, (questions, docs, _, _) in built.items():
            ds = EvidenceDataset(tmp, tokenizer, batch_size=2, max_length=1024, split=split,
                                 num_mtp_tokens=1, shuffle=False, max_evidence_tokens=3072)
            batches = list(iter(ds))
            assert batches and "evidence_ids" in batches[0], split
            assert ds.has_condition_labels
        print("8. EvidenceDataset reads every split                                              PASS")

        # 8b. the bridge build: same questions, chunks and sidecars, only the assistant turn and
        # the metadata's bridge field differ
        questions = built["chains_eval"][0]
        pcd.build_split(tmp, "chains_bridge_eval", questions, template, FakeEmbedder(), held, batch=64,
                        bridge=True)
        for suffix in ("ev", "evidx", "evchunk", "evkey", "evkeyidx", "evgold", "evhop", "cond", "ans"):
            with open(os.path.join(tmp, f"chains_eval.{suffix}"), "rb") as a, \
                    open(os.path.join(tmp, f"chains_bridge_eval.{suffix}"), "rb") as b:
                assert a.read() == b.read(), f"bridge build changed .{suffix}"
        docs = read_split(tmp, "chains_bridge_eval")[0]
        for q, d, plain in zip(questions, docs, built["chains_eval"][1]):
            prompt = template.encode_prompt([{"role": "user", "content": evidence_prompt(q.question)}])
            assert d["ids"][:len(prompt)] == prompt and not any(d["mask"][:len(prompt)])
            supervised = [t for t, m in zip(d["ids"], d["mask"]) if m]
            assert supervised == tokenizer(chains.bridge_answer(q), add_special_tokens=False)["input_ids"] + \
                [template.eos_id]
            meta, plain_meta = dict(d["meta"]), dict(plain["meta"])
            assert meta.pop("bridge") == q.entities[1:-1]
            assert meta.pop("qid").split("-")[-1] == plain_meta.pop("qid").split("-")[-1]
            assert meta == plain_meta
            assert "bridge" not in plain["meta"]
        print("8b. bridge build: buffers and sidecars byte identical, assistant turn is the bridge  PASS")
        # 9. streaming: byte identical to drawing every question first, plain and bridge, with the
        # training hop mix widened to 1..4 the way the corpus driver patches it; resumable
        def files_of(directory, split):
            return {os.path.basename(p): open(p, "rb").read() for p in pcd.split_files(directory, split)
                    if os.path.exists(p) and not p.endswith((".build.json", ".build_state.json"))}

        saved_defaults = chains.sample_hops.__defaults__
        try:
            for hop_mix in (saved_defaults, ((0.25, 0.25, 0.25, 0.25),)):
                chains.sample_hops.__defaults__ = hop_mix
                for bridge in (False, True):
                    ref_dir, new_dir = os.path.join(tmp, f"ref_{len(hop_mix[0])}_{bridge}"),                         os.path.join(tmp, f"new_{len(hop_mix[0])}_{bridge}")
                    os.makedirs(ref_dir), os.makedirs(new_dir)
                    qs = pcd.make_split_questions("train", 330, 42, held)
                    ref_report = pcd.build_split(ref_dir, "t", qs, template, FakeEmbedder(), held, batch=64,
                                                 bridge=bridge)
                    new_report = pcd.build_split_streaming(new_dir, "t", "train", 330, 0, 42, held, template,
                                                           FakeEmbedder(), batch=50, bridge=bridge)
                    assert files_of(ref_dir, "t") == files_of(new_dir, "t")
                    for key in ("questions", "prompt_tokens", "ev_tokens", "chunks", "by_hops", "by_composition",
                                "kinds", "distractors"):
                        assert ref_report[key] == new_report[key], key
                    for hops, base in ref_report["shortcuts"].items():
                        assert all(abs(new_report["shortcuts"][hops][k] - v) < 1e-9 for k, v in base.items())
            chains.sample_hops.__defaults__ = ((0.25, 0.25, 0.25, 0.25),)
            for name, count, per_hop in (("val", 80, 0), ("eval", 0, 15), ("hop4", 0, 15)):
                ref_dir, new_dir = os.path.join(tmp, f"ref_{name}"), os.path.join(tmp, f"new_{name}")
                os.makedirs(ref_dir), os.makedirs(new_dir)
                qs = pcd.make_split_questions(name, count, 42, held, per_hop)
                pcd.build_split(ref_dir, "t", qs, template, FakeEmbedder(), held, batch=64,
                                facts=name not in ("train", "val"))
                pcd.build_split_streaming(new_dir, "t", name, count, per_hop, 42, held, template, FakeEmbedder(),
                                          batch=7, facts=name not in ("train", "val"))
                assert files_of(ref_dir, "t") == files_of(new_dir, "t"), name
            print("9a. streamed builds equal the materialized ones byte for byte (plain, bridge, 1..3 and 1..4 hop "
                  "mixes, val, eval, hop4)   PASS")

            # 9b. resume after a stop and after a crash before the first batch, a junk tail, other arguments,
            # the completion record, --overwrite
            ref_dir = os.path.join(tmp, "ref_4_False")
            ref_files = files_of(ref_dir, "t")
            for mode, limit in (("stop", 3), ("crash", 1), ("crash", 4)):
                d = os.path.join(tmp, f"res_{mode}_{limit}")
                os.makedirs(d)
                calls = {"n": 0}

                def stop_check(mode=mode, limit=limit):
                    calls["n"] += 1
                    if calls["n"] == limit:
                        if mode == "crash":
                            raise RuntimeError("killed")
                        return True
                    return False

                try:
                    pcd.build_split_streaming(d, "t", "train", 330, 0, 42, held, template, FakeEmbedder(),
                                              batch=50, checkpoint_seconds=0, stop_check=stop_check)
                    raise AssertionError("the build did not stop")
                except pcd.BuildInterrupted:
                    assert mode == "stop"
                except RuntimeError:
                    assert mode == "crash"
                state = json.load(open(os.path.join(d, "t.build_state.json")))
                assert state["questions"] == (limit - 1) * 50
                for suffix in ("bin", "evhop", "chains.jsonl", "ev"):
                    with open(os.path.join(d, f"t.{suffix}"), "ab") as f:
                        f.write(b"junk")
                try:
                    pcd.build_split_streaming(d, "t", "train", 330, 0, 43, held, template, FakeEmbedder(), batch=50)
                    raise AssertionError("a resume with other arguments was accepted")
                except SystemExit as refused:
                    assert "other arguments" in str(refused)
                try:
                    pcd.build_split_streaming(d, "t", "train", 330, 0, 42, held, template, FakeEmbedder(), batch=50,
                                              bridge=True)
                    raise AssertionError("a resume with the bridge form was accepted")
                except SystemExit:
                    pass
                report = pcd.build_split_streaming(d, "t", "train", 330, 0, 42, held, template, FakeEmbedder(),
                                                   batch=50)
                assert report["questions"] == 330
                assert files_of(d, "t") == ref_files, (mode, limit)
                assert not os.path.exists(os.path.join(d, "t.build_state.json"))
                stamp = {n: os.path.getmtime(os.path.join(d, n)) for n in os.listdir(d)}
                assert pcd.build_split_streaming(d, "t", "train", 330, 0, 42, held, template, FakeEmbedder(),
                                                 batch=50) is None
                assert stamp == {n: os.path.getmtime(os.path.join(d, n)) for n in os.listdir(d)}
                try:
                    pcd.build_split_streaming(d, "t", "train", 330, 0, 43, held, template, FakeEmbedder(), batch=50)
                    raise AssertionError("a finished split was rebuilt with other arguments")
                except FileExistsError:
                    pass
                again = pcd.build_split_streaming(d, "t", "train", 330, 0, 43, held, template, FakeEmbedder(),
                                                  batch=50, overwrite=True)
                assert again["questions"] == 330 and files_of(d, "t") != ref_files
            print("9b. a stopped or crashed streamed build resumes byte identical, other arguments are refused, a "
                  "finished split is left alone, --overwrite starts over   PASS")
        finally:
            chains.sample_hops.__defaults__ = saved_defaults
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("all chain generator checks passed")


if __name__ == "__main__":
    main()
