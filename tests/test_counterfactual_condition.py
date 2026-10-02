"""The counterfactual condition's model-free parts: eligibility, the swapped evidence, the readings.

``scripts/eval_abstention.py`` needs the model stack to import, so what it does with a model
(generation, log-probabilities) is not run here. What decides which items count and what the
numbers mean lives in ``modules/data/entity_swap.py`` and is checked on synthetic records:

1. Eligibility is decided once per record with the four reasons (unanswerable, type not swappable,
   answer not in the passage, no substitute) and the eligible ones carry a substitute that is the
   same type, differs from every reference and does not occur in the passage.
2. The swapped chunks contain the substitute and no original (every occurrence replaced), their
   token ids are the re-tokenized swapped text, unchanged chunks keep their ids, and the evidence
   row carries one re-embedded key per chunk.
3. The readings: per stratum follow rate, ``mr_gen`` (None when nothing followed and nothing stuck),
   mean ``mr_ll``, other rate, and the gate (PASS, FAIL, "n too small"), over the items answered
   correctly with the unswapped chunk and again over every eligible item; ``sigmoid_ratio`` and the
   frequency strata are exact at their boundaries.
4. The per-record JSON entry omits the token arrays and evidence packaging and survives
   ``json.dumps``.
5. ``eval_abstention.py`` (read as source, not imported): ``counterfactual`` is an evidence
   condition, the gold requirement is enforced, and every frozen function is syntactically
   identical to the same function at the commit this work started from.
6. The injected-fact source: re-rendering a card with an attribute swapped holds the substitute and
   not the original (skipped when ``modules/data/biographies.py`` is absent).

Needs the pruned tokenizer (``utils.TOKENIZER_DIR``) for 1 and 2; no GPU, no model.
"""
import os, sys, ast, json, math, random, hashlib, subprocess
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from modules.data.entity_swap import (
    FREQ_STRATA, INELIGIBLE_REASONS, Gazetteer, answer_type, counterfactual_evidence_row, counterfactual_readings,
    frequency_stratum, prepare_counterfactual, record_for_json, sigmoid_ratio, swap_in_text,
)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FROZEN = ("load_model", "generate_batch", "_pack_evidence_batch", "teacher_forced_calibration",
          "normalize_answer", "exact_match", "token_f1", "abstention_scores", "load_squad_split",
          "squad_references")
BASELINE_COMMIT = "d3a0fde"


class FakeEmbedder:
    def encode(self, texts):
        out = np.zeros((len(texts), 384), dtype=np.float32)
        for i, text in enumerate(texts):
            seed = int.from_bytes(hashlib.sha1(text.encode()).digest()[:4], "big")
            v = np.random.RandomState(seed).randn(384).astype(np.float32)
            out[i] = v / np.linalg.norm(v)
        return out


def make_records(tokenizer):
    ids = lambda text: tokenizer(text, add_special_tokens=False)["input_ids"]
    passage = ("Warsaw is the capital of Poland. Warsaw lies on the Vistula. "
               "The river Vistula flows north.")
    other = "Krakow, Gdansk and Lodz are large Polish cities."
    rows = [
        # eligible: a LOCATION answer present twice in the first of two chunks
        dict(question="What is the capital of Poland?", references=["Warsaw"], unanswerable=False,
             chunks=[passage, other]),
        # the reference is not in the passage
        dict(question="Where is the Vistula north of?", references=["Gdynia"], unanswerable=False,
             chunks=[passage, other]),
        dict(question="What is the capital of Hungary?", references=[], unanswerable=True,
             chunks=[passage]),
        dict(question="Why is it large?", references=["its population"], unanswerable=False,
             chunks=["It has a large population, its population grew."]),
        dict(question="What is the capital of France?", references=["Paris"], unanswerable=False,
             chunks=[passage]),
        dict(question="What is the capital of Narnia?", references=["Cair Paravel"], unanswerable=False,
             chunks=["Cair Paravel is the seat of the High King."]),
    ]
    records = []
    for i, row in enumerate(rows):
        records.append({
            "id": str(i), "question": row["question"], "references": row["references"],
            "unanswerable": row["unanswerable"],
            "gold_chunks": [(t, ids(t)) for t in row["chunks"]],
        })
    return records


def main():
    from utils import TOKENIZER_DIR
    if not os.path.isdir(TOKENIZER_DIR):
        print(f"SKIP: no tokenizer at {TOKENIZER_DIR}")
        return
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_DIR)

    records = make_records(tokenizer)
    pairs = [("Where was he born?", c) for c in ("Lyon", "Nice", "Brest", "Metz", "Nancy", "Rennes",
                                                   "Lille", "Dijon", "Tours", "Nantes")]
    pairs += [("What is the capital of Spain?", "Madrid"), ("What is the capital of Poland?", "Warsaw"),
              ("What is the capital of Hungary?", "Budapest")]
    gazetteer = Gazetteer.from_pairs(pairs)
    counts = prepare_counterfactual(records, gazetteer, random.Random(0), tokenizer)

    # 1. eligibility
    reasons = [r.get("cf_reason") for r in records]
    assert reasons == [None, "answer_not_in_passage", "unanswerable", "type_not_swappable",
                       "answer_not_in_passage", None], reasons
    assert counts["eligible"] == 2 and counts["unanswerable"] == 1 and counts["type_not_swappable"] == 1
    assert counts["answer_not_in_passage"] == 2
    assert records[1]["cf"] is None and records[2]["cf"] is None and records[3]["cf"] is None
    for r in (records[0], records[5]):
        cf = r["cf"]
        assert answer_type(r["question"], cf["original"]) == cf["type"] == "LOCATION"
        assert cf["substitute"] not in r["references"] and cf["substitute"] != cf["original"]
        passage = " ".join(t for t, _ in r["gold_chunks"])
        assert swap_in_text(passage, cf["substitute"], "")[1] == 0, "the substitute is already in the passage"
        assert (r["cf_original"], r["cf_substitute"], r["cf_type"]) == (
            cf["original"], cf["substitute"], cf["type"])
    # decided once: a second pass leaves a record that already has an entry alone
    before = records[0]["cf"]
    again = prepare_counterfactual(records, gazetteer, random.Random(99), tokenizer)
    assert records[0]["cf"] is before and again["eligible"] == 2
    # no substitute: a gazetteer holding only the answer itself
    solo = make_records(tokenizer)[:1]
    got = prepare_counterfactual(solo, Gazetteer.from_pairs([("Where?", "Warsaw")]), random.Random(0),
                                 tokenizer)
    assert got["no_substitute"] == 1 and solo[0]["cf_reason"] == "no_substitute"
    # an alias that survives the swap makes the record ineligible
    tricky = make_records(tokenizer)[:1]
    tricky[0].update(question="Where was he born?", references=["Lyon", "Paris France"],
                     gold_chunks=[("He was born in Lyon France, a fine town.",
                                   tokenizer("He was born in Lyon France, a fine town.",
                                             add_special_tokens=False)["input_ids"])])

    class FixedGazetteer:
        """The draw rules keep a substitute out of every alias, so a leftover is forced with a stub."""

        def draw(self, *args, **kwargs):
            return "Paris"

    got = prepare_counterfactual(tricky, FixedGazetteer(), random.Random(0), tokenizer)
    assert got["alias_remains"] == 1 and tricky[0]["cf"] is None, got
    assert tricky[0]["cf_reason"] == "alias_remains" and "alias_remains" in INELIGIBLE_REASONS
    # every alias is swapped when none is left behind
    both = make_records(tokenizer)[:1]
    both[0].update(question="Where was he born?", references=["Lyon", "the City of Lights"],
                   gold_chunks=[("Lyon, the City of Lights, is old. Lyon is big.",
                                 tokenizer("Lyon, the City of Lights, is old. Lyon is big.",
                                           add_special_tokens=False)["input_ids"])])
    prepare_counterfactual(both, gazetteer, random.Random(0), tokenizer)
    cf = both[0]["cf"]
    assert cf is not None and cf["count"] == 3, both[0].get("cf_reason")
    assert "Lyon" not in cf["chunks"][0][0] and "City of Lights" not in cf["chunks"][0][0]
    print("1. eligibility: four reasons counted once, substitute of the same type     PASS")

    # 2. the swapped evidence
    r = records[0]
    cf = r["cf"]
    swapped_texts = [t for t, _ in cf["chunks"]]
    assert cf["count"] == 2 and len(swapped_texts) == 2
    for text, ids in cf["chunks"]:
        assert ids == tokenizer(text, add_special_tokens=False)["input_ids"]
        assert swap_in_text(text, cf["original"], "")[1] == 0, "an original survived the swap"
    assert cf["substitute"] in swapped_texts[0] and swapped_texts[0].count(cf["substitute"]) == 2
    assert swapped_texts[1] == r["gold_chunks"][1][0] and cf["chunks"][1][1] == r["gold_chunks"][1][1], \
        "a chunk without the answer must keep its text and ids"
    assert cf["substitute"] in tokenizer.decode(cf["chunks"][0][1])
    row = counterfactual_evidence_row(cf["chunks"], FakeEmbedder())
    assert row["keys"].shape == (2, 384) and len(row["ids"]) == len(row["chunk_ids"])
    assert sorted(set(row["chunk_ids"])) == [0, 1]
    assert not np.allclose(row["keys"][0], FakeEmbedder().encode([r["gold_chunks"][0][0]])[0]), \
        "the swapped chunk was not re-embedded"
    assert counterfactual_evidence_row([], FakeEmbedder()) is None
    print("2. swapped chunks: substitute in, original out, re-tokenized, re-embedded   PASS")

    # 3. readings
    def rec(stratum, follow, stuck, mr, gold_em=1.0):
        return {"stratum": stratum, "cf_follow": float(follow), "cf_stuck": float(stuck),
                "mr_ll": mr, "gold_em": gold_em}
    rows = ([rec("0", 1, 0, 0.1)] * 40 + [rec("0", 0, 0, 0.5)] * 2
            + [rec("1000+", 1, 0, 0.2)] * 20 + [rec("1000+", 0, 1, 0.9)] * 20
            + [rec("10-99", 0, 0, None)] * 5 + [rec("10-99", 1, 0, 0.3, gold_em=0.0)] * 3)
    readings = counterfactual_readings(rows)
    ok = readings["correct_with_gold"]
    assert list(ok) == ["0", "10-99", "1000+", "all"], list(ok)
    zero = ok["0"]
    assert zero["n"] == 42 and zero["n_follow"] == 40 and abs(zero["follow_rate"] - 40 / 42) < 1e-12
    assert zero["mr_gen"] == 0.0 and abs(zero["other_rate"] - 2 / 42) < 1e-12
    assert abs(zero["mr_ll"] - (40 * 0.1 + 2 * 0.5) / 42) < 1e-12 and zero["gate"] == "PASS"
    head = ok["1000+"]
    assert head["mr_gen"] == 0.5 and head["follow_rate"] == 0.5 and head["gate"] == "FAIL"
    thin = ok["10-99"]
    assert thin["n"] == 5 and thin["mr_gen"] is None and thin["mr_ll"] is None
    assert thin["gate"] == "n too small"
    assert readings["all_eligible"]["10-99"]["n"] == 8 and readings["all_eligible"]["10-99"]["n_follow"] == 3
    assert ok["all"]["n"] == 87
    assert abs(sigmoid_ratio(0.0, 0.0) - 0.5) < 1e-12 and sigmoid_ratio(-1e4, 0.0) == 0.0
    assert abs(sigmoid_ratio(3.0, 0.0) + sigmoid_ratio(0.0, 3.0) - 1.0) < 1e-12
    assert sigmoid_ratio(1e4, 0.0) == 1.0 and not math.isnan(sigmoid_ratio(-800.0, 0.0))
    assert [frequency_stratum(c) for c in (0, 1, 9, 10, 99, 100, 999, 1000, 10 ** 6)] == \
        ["0", "1-9", "1-9", "10-99", "10-99", "100-999", "100-999", "1000+", "1000+"]
    assert FREQ_STRATA == ("0", "1-9", "10-99", "100-999", "1000+")
    print("3. readings: follow, mr_gen (None case), mr_ll, gate, strata, sigmoid        PASS")

    # 4. the per-record entry
    full = {"id": "7", "question": "q", "references": ["a"], "condition_unanswerable": False,
            "completion": "b", "abstained": False, "em": 0.0, "f1": 0.0, "p_max": 0.5,
            "memory_mass": 0.3, "memory_mass_by_loop": {0: 0.1}, "memory_mass_by_expert": {0: 0.3},
            "cf_original": "a", "cf_substitute": "b", "ll_orig": -3.0, "ll_swap": -1.0, "mr_ll": 0.1,
            "stratum": "0", "prompt_ids": [1, 2], "forced_ids": [1], "forced_mask": [1],
            "evidence_row": {"ids": [1]}, "answer_forced": ([1], [1]), "gold_chunks": [("t", [1])],
            "cf": {"chunks": []}}
    entry = record_for_json(full)
    for dropped in ("prompt_ids", "forced_ids", "forced_mask", "evidence_row", "answer_forced",
                    "gold_chunks", "cf"):
        assert dropped not in entry, dropped
    for kept in ("id", "question", "references", "condition_unanswerable", "completion", "abstained",
                 "em", "f1", "p_max", "memory_mass", "memory_mass_by_loop", "memory_mass_by_expert",
                 "cf_original", "cf_substitute", "ll_orig", "ll_swap", "mr_ll", "stratum"):
        assert kept in entry, kept
    assert json.loads(json.dumps(entry))["memory_mass_by_expert"] == {"0": 0.3}
    assert "prompt_ids" in full, "record_for_json must copy, not mutate"
    print("4. per-record JSON entry: token arrays gone, fields kept                     PASS")

    # 5. eval_abstention.py read as source
    path = os.path.join(ROOT, "scripts", "eval_abstention.py")
    source = open(path, encoding="utf-8").read()
    tree = ast.parse(source)
    conditions = None
    for node in tree.body:
        if isinstance(node, ast.Assign) and getattr(node.targets[0], "id", "") == "EVIDENCE_CONDITIONS":
            conditions = ast.literal_eval(node.value)
    assert conditions == ("gold", "none", "distractors", "mixed", "counterfactual"), conditions
    assert "counterfactual needs gold in the same run" in source
    try:
        old = subprocess.run(["git", "show", f"{BASELINE_COMMIT}:scripts/eval_abstention.py"],
                             cwd=ROOT, capture_output=True, timeout=30, check=True).stdout.decode("utf-8")
    except (OSError, subprocess.SubprocessError):
        old = None
        print("5. SKIP the frozen function comparison: git show is not available")
    if old is not None:
        new_funcs = {n.name: ast.dump(n) for n in tree.body if isinstance(n, ast.FunctionDef)}
        old_funcs = {n.name: ast.dump(n) for n in ast.parse(old).body if isinstance(n, ast.FunctionDef)}
        for name in FROZEN:
            assert name in old_funcs and name in new_funcs, name
            assert new_funcs[name] == old_funcs[name], f"frozen function {name} changed"
        print(f"5. eval_abstention: counterfactual registered, gold required, "
              f"{len(FROZEN)} frozen functions unchanged  PASS")

    # 6. the injected-fact source
    try:
        from modules.data import biographies as bio
    except ImportError:
        print("6. SKIP the injected-fact card swap: modules/data/biographies.py is absent")
    else:
        pools = bio.make_pools(0)
        people = bio.make_people(pools, {1: 2, 10: 2}, 0)
        person = people[0]
        for attribute in bio.ENTITY_ATTRIBUTES:
            value = person.attributes[attribute]
            substitute = next(v for v in pools[attribute] if v != value)
            card = bio.render_store_chunk(person, {attribute: substitute})
            assert substitute in card and (value not in card or value in substitute), attribute
            assert bio.question_for(attribute, person.name).endswith("?")
        print("6. injected-fact card re-rendered with the substitute, original gone        PASS")
    print("all counterfactual condition checks passed")


if __name__ == "__main__":
    main()
