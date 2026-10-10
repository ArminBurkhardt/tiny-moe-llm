"""PopQA as a closed-book rank task: candidates, tiers, the prior control and the score direction.

Six checks on a 30-row synthetic frame, no GPU and no download:

1. **Candidates** share the row's relation, start with the gold, are distinct after normalization,
   never contain an accepted alias of the gold, and are the same for a row whatever ``limit`` or
   run it is scored in (seeded by the row id).
2. **Relations with fewer distinct objects than candidates are excluded** and listed with their
   count, and their rows are skipped.
3. **Tiers are thirds of the whole set** by subject popularity; the log10 bins and relation
   classes are what the report says.
4. **The prior control** replaces the subject with ``X``: its context contains no subject, and a
   question that does not name its subject has no prior item.
5. **Direction**: the shared tables print ``1 - norm_rank``, so a better model scores higher.
6. **End to end with a scripted backend** (needs ``scripts/closed_book_rank.py``): a backend that
   knows every gold object ranks it first on the real context and at chance on the prior one.

``scripts/eval_benchmarks.py`` imports the model at module scope (Transformer Engine, a GPU stack),
so this test puts two stand-in modules in ``sys.modules`` first. Nothing here runs the model.
"""
import os, sys, types, json, hashlib
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

import numpy as np

stub_transformer = types.ModuleType("modules.model.transformer")
stub_transformer.TinyMoETransformer = object
stub_attention = types.ModuleType("modules.model.attention")
stub_attention.cu_seqlens_from_doc_ids = None
stub_attention._segment_ids = None
sys.modules.setdefault("modules.model.transformer", stub_transformer)
sys.modules.setdefault("modules.model.attention", stub_attention)

from scripts import eval_benchmarks as eb  # noqa: E402
from scripts.eval_abstention import normalize_answer  # noqa: E402


def frame_rows():
    rows = []
    pob = [f"Town{c}" for c in "ABCDEFGHIJKL"]
    for i in range(12):
        alias = [pob[(i + 1) % 12]] if i % 4 == 0 else []
        rows.append({"id": 100 + i, "subj": f"Subject{i} Name", "prop": "place of birth", "obj": pob[i],
                     "s_pop": 10 ** (1 + i / 6), "o_pop": 5,
                     "possible_answers": json.dumps([pob[i]] + alias),
                     "question": f"In what city was Subject{i} Name born?"})
    jobs = ["actor", "poet", "judge", "chef", "pilot", "sailor"]
    for i in range(12):
        rows.append({"id": 200 + i, "subj": f"Person{i} Other", "prop": "occupation", "obj": jobs[i % 6],
                     "s_pop": 10 ** (3 + i / 6), "o_pop": 9, "possible_answers": str([jobs[i % 6]]),
                     "question": f"What is Person{i} Other's occupation?"})
    for i in range(5):
        rows.append({"id": 300 + i, "subj": f"Mum{i} Child", "prop": "mother", "obj": f"Mother{i % 3}",
                     "s_pop": 10 ** (5 + i / 6), "o_pop": 2, "possible_answers": json.dumps([f"Mother{i % 3}"]),
                     "question": f"Who is the mother of Mum{i} Child?"})
    # a question that never names its subject: no prior item
    rows.append({"id": 400, "subj": "Hidden Subject", "prop": "place of birth", "obj": "TownZ",
                 "s_pop": 10 ** 6.5, "o_pop": 1, "possible_answers": json.dumps(["TownZ"]),
                 "question": "In what city was the writer born?"})
    return rows


def test_candidates_tiers_prior():
    rendered = [eb.render_popqa(r) for r in frame_rows()]
    assert len(rendered) == 30 and all(rendered)
    assert eb._popqa_answers('["a", "b"]') == ["a", "b"] and eb._popqa_answers("['a']") == ["a"]
    assert eb._popqa_answers(np.array(["x", "y"])) == ["x", "y"] and eb._popqa_answers("plain") == ["plain"]
    assert eb._popqa_answers(None) == []

    n = 5
    plan = eb.plan_popqa(rendered, n, seed=0, limit=None)
    assert plan["excluded_relations"] == {"mother": 3}, plan["excluded_relations"]
    assert plan["skipped"]["excluded_relation"] == 5
    by_id = {r["id"]: r for r in rendered}
    pools = {}
    for r in rendered:
        pools.setdefault(r["prop"], set()).add(r["obj"])
    assert len(plan["items"]) == 25 and not any(i["group"] == "mother" for i in plan["items"])

    for item in plan["items"]:
        row = by_id[item["item_id"]]
        assert item["candidates"][0] == item["gold"] == row["obj"] and len(item["candidates"]) == n
        assert set(item["candidates"]) <= pools[row["prop"]], "a candidate from another relation"
        norms = [normalize_answer(c) for c in item["candidates"]]
        assert len(set(norms)) == n
        banned = {normalize_answer(a) for a in row["answers"] if a != row["obj"]}
        assert not banned & set(norms[1:]), "an accepted alias was offered as a distractor"
    row0 = next(i for i in plan["items"] if i["item_id"] == "100")
    assert "TownB" not in row0["candidates"], "row 100 lists TownB as another correct answer"
    print("1. candidates: same relation, distinct, aliases excluded  PASS")

    again = eb.plan_popqa(rendered, n, seed=0, limit=None)
    assert [i["candidates"] for i in again["items"]] == [i["candidates"] for i in plan["items"]]
    limited = eb.plan_popqa(rendered, n, seed=3, limit=9)
    assert 0 < len(limited["items"]) <= 9
    full = {i["item_id"]: i for i in plan["items"]}
    for item in limited["items"]:
        assert item["candidates"] == full[item["item_id"]]["candidates"], "candidates depend on --limit"
        assert item["tier"] == full[item["item_id"]]["tier"], "tiers depend on --limit"
    print("1b. candidates and tiers do not depend on the subsample    PASS")
    print("2. thin relations are excluded and listed                  PASS")

    cutoffs = eb.popqa_tier_cutoffs([r["s_pop"] for r in rendered])
    tiers = [eb.popqa_tier(r["s_pop"], cutoffs) for r in rendered]
    assert [tiers.count(t) for t in eb.POPQA_TIERS] == [10, 10, 10], [tiers.count(t) for t in eb.POPQA_TIERS]
    assert tiers[0] == "tail" and tiers[-1] == "head"
    assert [eb.popqa_pop_bin(v) for v in (5, 99, 100, 999, 1000, 9999, 10000, 1e7)] == [
        "<2", "<2", "2-3", "2-3", "3-4", "3-4", ">=4", ">=4"]
    assert eb.popqa_class("director") == "entity" and eb.popqa_class("genre") == "attribute"
    assert eb.popqa_class("capital of") == "entity" and eb.popqa_class("sport") == "attribute"
    assert eb.popqa_class("unheard of") == "other"
    print("3. tiers are thirds, bins and classes as reported          PASS")

    for item in plan["items"]:
        row = by_id[item["item_id"]]
        if row["subj"] == "Hidden Subject":
            continue
        assert row["subj"].lower() not in item["prior_context"].lower(), item["prior_context"]
        assert "X" in item["prior_context"] and item["context"] == f"Q: {row['question']}\nA:"
    hidden = next(i for i in plan["items"] if i["item_id"] == "400")
    assert hidden["prior_context"] is None and plan["skipped"]["prior_missing"] == 1
    assert eb.popqa_prior_question("What is Ada's occupation?", "ada") == "What is X's occupation?"
    assert eb.popqa_prior_question("Who met Adamson and Adam?", "Adam") == "Who met Adamson and X?"
    assert eb.popqa_prior_question("What did Adamson say?", "Adam") is None
    assert eb.popqa_prior_question("Is Madam here?", "Ada") is None
    print("4. prior control hides the subject                         PASS")

    task = eb.TASKS["popqa"]
    assert task.kind == "rank" and task.metric == "norm_rank" and task.chance == 0.5
    assert "popqa" not in eb.MC_TASKS + eb.GEN_TASKS and eb.RANK_TASKS == ["popqa"]
    assert abs(eb.headline({"norm_rank": 0.3}, task) - 0.7) < 1e-12
    assert eb.headline({"norm_rank": float("nan")}, task) is None
    assert eb.metric_label(task) == "1-norm_rank"
    assert eb.metric_label(eb.TASKS["piqa"]) == eb.TASKS["piqa"].metric
    print("5. shared tables print 1 - norm_rank                       PASS")
    return plan


class ScriptedBackend:
    """Bytes as tokens. Knows the gold object of every real context; scores the rest by a hash."""

    bos_id = 1

    def __init__(self, gold_by_context):
        self.gold_by_context = gold_by_context

    def encode_many(self, texts):
        return [list(t.encode("utf-8")) for t in texts]

    def score(self, batch):
        out = []
        for context_ids, continuation_ids in batch:
            context = bytes(i for i in context_ids if i != self.bos_id).decode("utf-8")
            candidate = bytes(continuation_ids).decode("utf-8").strip()
            if self.gold_by_context.get(context) == candidate:
                logprob = -1.0
            else:
                digest = hashlib.sha1((context + "|" + candidate).encode()).digest()
                logprob = -5.0 - digest[0] / 64.0
            out.append((logprob, False))
        return out


def test_end_to_end(plan):
    try:
        import scripts.closed_book_rank  # noqa: F401
    except ImportError:
        print("SKIP: scripts/closed_book_rank.py is not there yet, the scripted backend check is not run")
        return
    gold = {i["context"]: i["gold"] for i in plan["items"]}
    result = eb.score_rank_task(ScriptedBackend(gold), plan, 5, batch_size=16, max_len=256, progress="test")
    entity = result["entity"]
    assert entity["all"]["norm_rank"] == 0.0 and result["norm_rank"] == 0.0, entity["all"]
    assert result["n"] == entity["all"]["n"] and 0 < result["n"] < 25
    assert entity["prior_all"]["norm_rank"] > 0.15, "the prior reading should not know the gold object"
    assert entity["delta_prior_minus_real"]["all"]["diff_norm_rank"] > 0.15
    assert set(entity["tier"]) <= set(eb.POPQA_TIERS) and entity["tier"]
    assert "attribute" in result and result["items"] and len(result["items_prior"]) < len(result["items"]) + 1
    assert json.dumps(result)
    print("6. scripted backend: real context ranks gold first, prior does not  PASS")


def main():
    plan = test_candidates_tiers_prior()
    test_end_to_end(plan)


if __name__ == "__main__":
    main()
