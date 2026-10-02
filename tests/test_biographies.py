"""The biography generator: pools, people, renders, store cards and probes.

What a closed-book reading leans on, checked here:

1. the pools have the stated sizes, distinct values, no value a whitespace-token prefix of another
   in its pool, and no value of one pool containing or sitting inside another's;
2. ``make_pools`` and ``make_people`` are deterministic per seed and differ across seeds;
3. attribute draws are uniform (a loose chi-square bound); corpus frequency is still exposure
   weighted, which is why chance is read against a fresh-name control (see 7b);
4. every span of ``render_bio`` slices exactly the value or the name it labels, each attribute value
   occurs once, and every template of every attribute is exercised;
5. a render with overrides has the structure of the plain render (same labels, same positions of the
   sentences) and carries the substituted text;
6. the store card holds every value verbatim and no probe context holds its value;
7. facts and pools round trip through their files; 7b. a fresh name is deterministic per key and
   shares no word with any person's name or any pool value;
8. with the tokenizer: every value is 1 to 8 tokens, a card is under 128 tokens, a bio is a
   plausible length.

GPU free. Needs the pruned tokenizer (``utils.TOKENIZER_DIR``) for the last block.
"""
import os, sys, re, random, tempfile, shutil
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.data import biographies as bio
from modules.data.biographies import (
    ATTRIBUTES, DEFAULT_PERSONS_PER_TIER, POOL_SIZES, _TEMPLATES, make_people, make_pools,
    probe_prompt, question_for, render_bio, render_store_chunk,
)
from utils import TOKENIZER_DIR


def whitespace_prefix_free(values):
    return bio._whitespace_prefix_free(values)


def chi_square(counts, n_values):
    total = sum(counts.values())
    expected = total / n_values
    return sum((counts.get(k, 0) - expected) ** 2 / expected for k in counts) + \
        (n_values - len(counts)) * expected


def main():
    pools = make_pools(0)

    # 1. pools
    for attribute, size in POOL_SIZES.items():
        values = pools[attribute]
        assert len(values) == size, (attribute, len(values))
        assert len(set(values)) == size
        assert whitespace_prefix_free(values), attribute
    assert set(pools) == set(ATTRIBUTES)
    non_date = [(a, v) for a in ATTRIBUTES if a != "birth_date" for v in pools[a]]
    for i, (_, a) in enumerate(non_date):
        for _, b in non_date[i + 1:]:
            assert a.lower() not in b.lower() and b.lower() not in a.lower(), (a, b)
    assert all(re.fullmatch(r"[A-Z][a-z]+ \d{1,2}, \d{4}", d) for d in pools["birth_date"])
    assert all(1950 <= int(d[-4:]) <= 2005 for d in pools["birth_date"])
    assert all(v == v.lower() for v in pools["major"])
    print("1. pools: sizes, distinct, prefix free, no value inside another          PASS")

    # 2. determinism
    assert make_pools(0) == pools and make_pools(1) != pools
    tiers = {10: 3, 1: 5}
    a, b = make_people(pools, tiers, 7), make_people(pools, tiers, 7)
    assert a == b and make_people(pools, tiers, 8) != a
    assert [p.person_id for p in a] == list(range(8))
    assert [p.tier for p in a] == [10] * 3 + [1] * 5
    print("2. pools and people are deterministic per seed                          PASS")

    # 3. uniform draws, unique names
    people = make_people(pools, DEFAULT_PERSONS_PER_TIER, 3)
    assert len(people) == 3600 and len({p.name for p in people}) == 3600
    names = sorted(p.name for p in people)
    assert all(not names[i + 1].startswith(names[i]) for i in range(len(names) - 1))
    for attribute in ATTRIBUTES:
        counts = {}
        for p in people:
            counts[p.attributes[attribute]] = counts.get(p.attributes[attribute], 0) + 1
        n = len(pools[attribute])
        chi2 = chi_square(counts, n)
        assert chi2 < (n - 1) + 6 * (2 * (n - 1)) ** 0.5, (attribute, chi2)
    by_tier = {t: sum(1 for p in people if p.tier == t) for t in (1, 10, 100, 1000)}
    assert by_tier == {1: 2000, 10: 1000, 100: 500, 1000: 100}, by_tier
    print("3. 3600 unique names, attribute draws uniform (chi-square)              PASS")

    # 4. spans slice exactly, every template shows up
    seen = {a: set() for a in ATTRIBUTES}
    for a in ATTRIBUTES:
        assert len(_TEMPLATES[a]) >= 5, a
    sample = people[:40]
    rng = random.Random(0)
    for person in sample:
        for _ in range(60):
            text, spans = render_bio(person, rng)
            labels = [label for _, _, label in spans]
            for attribute in ATTRIBUTES:
                assert labels.count(attribute) == 1, (attribute, labels)
            assert 5 <= len(spans) <= 10
            for start, end, label in spans:
                shown = text[start:end]
                assert shown == (person.name if label == "name" else person.attributes[label]), \
                    (label, shown)
            first_sentence_end = text.index(". ") if ". " in text else len(text)
            assert any(label == "name" and start < first_sentence_end for start, _, label in spans), \
                "the first sentence does not name the person"
            for attribute in ATTRIBUTES:
                for k, template in enumerate(_TEMPLATES[attribute]):
                    pattern = re.escape(template).replace(r"\{s\}", r"(?:%s|[Hh]e|[Ss]he)" %
                                                          re.escape(person.name))
                    pattern = pattern.replace(r"\{v\}", re.escape(person.attributes[attribute]))
                    if re.search(pattern, text):
                        seen[attribute].add(k)
    for attribute in ATTRIBUTES:
        assert seen[attribute] == set(range(len(_TEMPLATES[attribute]))), (attribute, seen[attribute])
    pronoun_docs = sum(1 for p in sample if bio._pronoun(p) == "he")
    assert 0 < pronoun_docs < len(sample), "pronoun is not varying across people"
    print("4. spans slice values and names exactly; all templates exercised        PASS")

    # 5. overrides keep the structure
    person = people[5]
    substitute = {"employer": pools["employer"][0] if pools["employer"][0] != person.attributes["employer"]
                  else pools["employer"][1]}
    for seed in range(30):
        text0, spans0 = render_bio(person, random.Random(seed))
        text1, spans1 = render_bio(person, random.Random(seed), name_override="Person KTV",
                                   value_overrides=substitute)
        assert [label for _, _, label in spans0] == [label for _, _, label in spans1]
        for start, end, label in spans1:
            expected = ("Person KTV" if label == "name" else substitute.get(label, person.attributes.get(label)))
            assert text1[start:end] == expected, (label, text1[start:end])
        assert person.name not in text1 and person.attributes["employer"] not in text1
        assert text0.count(". ") == text1.count(". ")
    card = render_store_chunk(person, substitute)
    assert substitute["employer"] in card and person.attributes["employer"] not in card
    print("5. overrides keep structure, swap values, replace the name              PASS")

    # 6. cards and probes
    for p in people:
        for attribute in ATTRIBUTES:
            assert p.attributes[attribute] in p.store_chunk
        assert p.store_chunk == render_store_chunk(p)
        assert p.store_chunk.startswith(p.name + ".")
    for p in people[:200]:
        for attribute in ATTRIBUTES:
            for form, terminator in (("indist", "."), ("heldout", "\n")):
                context, term = probe_prompt(attribute, p.name, form)
                assert term == terminator and p.name in context
                assert p.attributes[attribute] not in context
                assert not context.endswith(" ") and not context.endswith(terminator)
            assert probe_prompt(attribute, p.name, "heldout")[0].startswith("Q: ")
            assert probe_prompt(attribute, p.name, "heldout")[0].endswith("\nA:")
            assert question_for(attribute, p.name) in probe_prompt(attribute, p.name, "heldout")[0]
    # the in-distribution probe is template 0 with the full name
    for attribute in ATTRIBUTES:
        assert probe_prompt(attribute, "N", "indist")[0] + " {v}." == _TEMPLATES[attribute][0].replace("{s}", "N")
    print("6. cards hold every value; probes never contain the value               PASS")

    # 7. files
    tmp = tempfile.mkdtemp(prefix="bios_")
    try:
        bio.save_facts(os.path.join(tmp, "f.jsonl"), people[:50])
        assert bio.load_facts(os.path.join(tmp, "f.jsonl")) == people[:50]
        bio.save_pools(os.path.join(tmp, "p.json"), pools)
        assert bio.load_pools(os.path.join(tmp, "p.json")) == pools
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("7. facts and pools round trip                                           PASS")

    # 7b. fresh names for the prior control
    fresh = bio.FreshNames(people, pools)
    taken = {w.lower() for p in people for w in p.name.split()}
    fresh_seen = set()
    for key in [f"{i}:birth_city:indist" for i in range(500)]:
        name = fresh.name(key)
        assert name == fresh.name(key), "fresh name is not deterministic"
        assert name not in {p.name for p in people}
        words = name.split()
        assert len(words) == 3 and all(w.lower() not in taken for w in words), name
        assert not any(w.lower() in pool_text for w in words for pool_text in
                       ["\n".join(v.lower() for vs in pools.values() for v in vs)]), name
        fresh_seen.add(name)
    assert len(fresh_seen) > 490, "fresh names collide too often"
    assert fresh.name("a") != fresh.name("b")
    print("7b. fresh names: deterministic, absent from every person name and pool      PASS")

    # 8. token counts
    if not os.path.isdir(TOKENIZER_DIR):
        print(f"8. SKIP: no tokenizer at {TOKENIZER_DIR}")
        print("all biography checks passed")
        return
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_DIR)
    for attribute in ATTRIBUTES:
        lengths = [len(x) for x in tokenizer(pools[attribute], add_special_tokens=False)["input_ids"]]
        assert 1 <= min(lengths) and max(lengths) <= 8, (attribute, min(lengths), max(lengths))
    card_lengths = [len(x) for x in tokenizer([p.store_chunk for p in people], add_special_tokens=False)["input_ids"]]
    assert max(card_lengths) < 128, max(card_lengths)
    rng = random.Random(1)
    bio_lengths = [len(x) for x in tokenizer([render_bio(p, rng)[0] for p in people[:400]],
                                             add_special_tokens=False)["input_ids"]]
    mean_len = sum(bio_lengths) / len(bio_lengths)
    assert 50 <= mean_len <= 130 and max(bio_lengths) <= 170, (mean_len, max(bio_lengths))
    print(f"8. tokens: values 1 to 8, cards max {max(card_lengths)}, bios mean {mean_len:.0f} "
          f"max {max(bio_lengths)}                        PASS")
    print("all biography checks passed")


if __name__ == "__main__":
    main()
