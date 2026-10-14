"""Answer typing, same-type substitutes and whole-word swaps (``modules/data/entity_swap.py``).

1. ``answer_type`` agrees with a fixed table of 40 (question, answer) cases.
2. ``Gazetteer.draw`` respects the type, the variant pool, every exclusion rule (equal, substring,
   superstring after normalization) and ``avoid_text``, and is reproducible per generator.
3. ``swap_in_text`` counts every occurrence, ignores case, and keeps whole words whole ("Paris" is
   not inside "Parisian").
4. A seeded sample of 50 swaps is printed for a human audit. It reads the cached SQuAD v2
   validation parquet when one is on disk (``data/benchmarks/squad_v2_validation``), else a small
   built-in table. The audit is read by a person; the asserts above are what the test enforces.

No GPU, no network, no tokenizer.
"""
import os, sys, random, glob
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.data.entity_swap import (
    ENTITY_TYPES, SWAPPABLE, Gazetteer, answer_type, answer_variant, asking_head, normalize,
    alias_remains, swap_all_aliases, swap_in_text,
)

TYPING_TABLE = [
    ("When was the treaty signed?", "June 28, 1919", "DATE"),
    ("What day did it open?", "12 March 1990", "DATE"),
    ("When did the war end?", "September 1945", "DATE"),
    ("When was it founded?", "2004-05-17", "DATE"),
    ("In what year did he die?", "1887", "YEAR"),
    ("What year was the bridge built?", "in 1932", "YEAR"),
    ("When did the band form?", "1990s", "YEAR"),
    ("How many people live there?", "3,500,000", "NUMBER"),
    ("How long is the river?", "250 miles", "NUMBER"),
    ("How many children did she have?", "three", "NUMBER"),
    ("How much did it cost?", "$4.5 million", "NUMBER"),
    ("What percentage voted yes?", "62%", "NUMBER"),
    ("How many seasons?", "twenty-one", "NUMBER"),
    ("Who wrote the novel?", "Charles Dickens", "PERSON"),
    ("Who was the first president?", "George Washington", "PERSON"),
    ("Whose theory explains it?", "Albert Einstein", "PERSON"),
    ("Which physicist proposed the idea?", "Niels Bohr", "PERSON"),
    ("What author wrote it?", "Jane Austen", "PERSON"),
    ("Who owns the airline?", "Delta Air Lines Inc.", "ORGANIZATION"),
    ("Where was he born?", "Warsaw", "LOCATION"),
    ("Where is the headquarters?", "Palo Alto, California", "LOCATION"),
    ("What city hosted the games?", "Montreal", "LOCATION"),
    ("Which country annexed it?", "Germany", "LOCATION"),
    ("What is the capital of France?", "Paris", "LOCATION"),
    ("Which river flows through the city?", "Thames", "LOCATION"),
    ("What state is it in?", "Nebraska", "LOCATION"),
    ("What company built the engine?", "Rolls-Royce", "ORGANIZATION"),
    ("Which university did she attend?", "University of Chicago", "ORGANIZATION"),
    ("What team won the final?", "Denver Broncos", "ORGANIZATION"),
    ("Which band recorded the album?", "The Beatles", "ORGANIZATION"),
    ("What newspaper reported it?", "New York Times", "ORGANIZATION"),
    ("What was the name of the ship?", "Santa Maria", "PROPER"),
    ("What language did they speak?", "Latin", "PROPER"),
    ("Which religion did he convert to?", "Buddhism", "PROPER"),
    ("What is the process called?", "photosynthesis", "COMMON"),
    ("Why did it fail?", "lack of funding", "COMMON"),
    ("How did they travel?", "by train", "COMMON"),
    ("What did the report conclude?", "that the bridge was unsafe", "COMMON"),
    ("What material is it made of?", "steel", "COMMON"),
    ("What is the answer?", "", "COMMON"),
]

BUILTIN_PAIRS = [
    ("Where was {n} born?", c) for n, c in zip(range(30), (
        "Warsaw Berlin Lisbon Madrid Vienna Prague Athens Oslo Dublin Cairo Lima Quito Hanoi Seoul "
        "Tokyo Nairobi Accra Dakar Tunis Rabat Sofia Riga Tallinn Vilnius Minsk Kyiv Tbilisi Baku "
        "Yerevan Tashkent").split())
] + [
    ("Who wrote book {n}?", f"{a} {b}") for n, (a, b) in enumerate(zip(
        "Anna Boris Carla David Elena Felix Greta Hugo Irene Jonas Karen Lukas Marta Nils Olga Pavel "
        "Quinn Rosa Stefan Tanya Ulrich Vera Walter Xenia Yusuf Zoe Adam Bruno Clara Dario".split(),
        "Adler Brandt Conti Dvorak Eriksen Fischer Garcia Hoffmann Ivanov Jensen Kowalski Larsen "
        "Moreau Novak Olsen Petrov Quist Rossi Schmidt Torres Unger Vogel Weber Xu Young Zimmer "
        "Albers Bauer Cruz Dietz".split()))
] + [
    ("In what year did event {n} happen?", str(1800 + 7 * n)) for n in range(25)
]


def check_typing():
    assert ENTITY_TYPES == ("DATE", "YEAR", "NUMBER", "PERSON", "LOCATION", "ORGANIZATION", "PROPER",
                            "COMMON")
    assert "COMMON" not in SWAPPABLE and set(SWAPPABLE) < set(ENTITY_TYPES)
    wrong = []
    for question, answer, expected in TYPING_TABLE:
        got = answer_type(question, answer)
        if got != expected:
            wrong.append((question, answer, expected, got))
    assert not wrong, f"typing disagrees on {len(wrong)} of {len(TYPING_TABLE)}: {wrong}"
    assert asking_head("In what city was it built?") == "city"
    assert asking_head("Why?") == ""
    print(f"1. answer_type matches all {len(TYPING_TABLE)} table cases                      PASS")


def check_draw():
    pairs = BUILTIN_PAIRS + [("What is the capital of {n}?", c) for n, c in enumerate(
        "Quito Lima Bogota Caracas Santiago Montevideo Asuncion Brasilia Georgetown Paramaribo "
        "Havana Kingston Managua Belmopan Tegucigalpa Panama Ottawa Washington Mexico Guatemala "
        "Nassau Bridgetown Castries Roseau".split())]
    gaz = Gazetteer.from_pairs(pairs)
    assert gaz.size("LOCATION") >= 40 and gaz.size("PERSON") == 30 and gaz.size("YEAR") == 25
    assert gaz.size("DATE") == 0 and gaz.size("NUMBER") == 0

    rng = random.Random(0)
    for _ in range(200):
        draw = gaz.draw("LOCATION", ["Warsaw"], rng)
        assert draw is not None and draw != "Warsaw"
    # the variant pool ("capital") has 24 members, so it is used and holds capitals only
    capitals = {c for _, c in pairs[-24:]}
    for _ in range(100):
        assert gaz.draw("LOCATION", ["Quito"], rng, variant="capital") in capitals
    # a variant pool under the floor falls back to the type pool
    assert gaz.draw("LOCATION", [], rng, variant="river") is not None

    # exclusion rules: equal, substring and superstring after normalization, articles ignored
    small = Gazetteer.from_pairs([("Where?", x) for x in ("Paris", "Parisian Quarter", "Lyon")])
    for _ in range(50):
        assert small.draw("LOCATION", ["the Paris"], random.Random(_)) == "Lyon"
        assert small.draw("LOCATION", ["Paris", "Lyon"], random.Random(_)) is None
    assert small.draw("LOCATION", ["Parisian Quarter Hotel"], random.Random(1)) == "Lyon"
    only = Gazetteer.from_pairs([("Where?", "Paris"), ("Where?", "Lyon")])
    assert only.draw("LOCATION", ["Parisians"], random.Random(0)) == "Lyon"      # "paris" in "parisians"
    # avoid_text: a candidate already in the text is skipped, whole words only
    assert only.draw("LOCATION", [], random.Random(0), avoid_text="He moved to Lyon in May.") == "Paris"
    assert only.draw("LOCATION", [], random.Random(3), avoid_text="Lyon and Paris") is None
    assert only.draw("PERSON", [], random.Random(0)) is None
    # reproducible per generator
    a = [gaz.draw("PERSON", [], random.Random(5)) for _ in range(3)]
    b = [gaz.draw("PERSON", [], random.Random(5)) for _ in range(3)]
    assert a == b
    # numbers split by unit
    nums = Gazetteer.from_pairs([("How long?", f"{n} miles") for n in range(5, 40)]
                                + [("How old?", f"{n} years") for n in range(5, 12)])
    assert answer_variant("NUMBER", "How long?", "12 miles") == "miles"
    for _ in range(50):
        assert nums.draw("NUMBER", ["12 miles"], rng, variant="miles").endswith("miles")
    # numbers and other proper names do not fall back to the wide pool
    assert nums.draw("NUMBER", [], rng, variant="years") is None            # 7 members
    assert nums.draw("NUMBER", [], rng) is None
    # stored forms: a bare year, no leading article
    years = Gazetteer.from_pairs([("When?", f"in {1900 + n}") for n in range(30)])
    assert all(years.draw("YEAR", [], rng).isdigit() for _ in range(20))
    named = Gazetteer.from_pairs([("Where?", f"the Place{chr(97 + n)}") for n in range(25)])
    assert all(not named.draw("LOCATION", [], rng).lower().startswith("the ") for _ in range(20))
    print("2. draw: type, variant, exclusions, avoid_text, reproducible                 PASS")


def check_swap():
    text = "Paris is large. The Parisian view of paris, and PARIS's markets, differ from Paris."
    out, n = swap_in_text(text, "Paris", "Lyon")
    assert n == 4, n
    assert out == "Lyon is large. The Parisian view of Lyon, and Lyon's markets, differ from Lyon.", out
    assert swap_in_text("Parisian", "Paris", "Lyon") == ("Parisian", 0)
    assert swap_in_text("x", "", "y") == ("x", 0)
    out, n = swap_in_text("It cost $4.5 million (about 4.5 million).", "4.5 million", "9 million")
    assert n == 2 and "$9 million" in out
    out, n = swap_in_text("a.b and axb", "a.b", "Z")
    assert (out, n) == ("Z and axb", 1)                  # the dot is literal, not a wildcard
    out, n = swap_in_text("1990 and 21990 and 1990s", "1990", "1850")
    assert (out, n) == ("1850 and 21990 and 1990s", 1)
    out, n = swap_in_text("Dr. Smith met Mr. Smith", "Smith", "\\1 Jones")
    assert out == "Dr. \\1 Jones met Mr. \\1 Jones" and n == 2   # substitute is literal
    print("3. swap_in_text: counts, case, word boundaries, literal replacement        PASS")


def check_aliases():
    text = "The Netherlands, or Holland, is small. Holland and the netherlands trade; Nederland too."
    out, n = swap_all_aliases(text, ["Holland", "The Netherlands", "Nederland"], "Chile")
    assert n == 5, (out, n)
    assert not alias_remains(out, ["Holland", "The Netherlands", "Nederland"]), out
    assert out.count("Chile") == 5 and "Holland" not in out and "Nederland" not in out
    # a shorter alias inside a longer one is not cut out of it first
    out, n = swap_all_aliases("New York City is big", ["York", "New York City"], "Rome")
    assert (out, n) == ("Rome is big", 1), (out, n)
    # a swap whose substitute joins the neighbouring words into another alias leaves one behind
    out, n = swap_all_aliases("Lyon France", ["Lyon", "Paris France"], "Paris")
    assert n == 1 and alias_remains(out, ["Lyon", "Paris France"]), out
    assert not alias_remains("Parisian", ["Paris"]) and not alias_remains("x", [""])
    print("3b. swap_all_aliases: every alias longest first, leftovers detected         PASS")


def audit_pairs():
    root = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "data", "benchmarks", "squad_v2_validation")
    files = sorted(glob.glob(os.path.join(root, "**", "*.parquet"), recursive=True))
    if files:
        import pandas as pd
        frame = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
        out = []
        for row in frame.to_dict("records"):
            texts = (row.get("answers") or {}).get("text")
            if texts is not None and len(texts):
                out.append((str(row["question"]), str(texts[0]), str(row["context"])))
        return out, f"SQuAD v2 validation ({len(out):,} answerable rows)"
    return [(q.format(n=i), a, f"{a} is mentioned here.") for i, (q, a) in enumerate(BUILTIN_PAIRS)], \
        "built-in table"


def print_audit():
    rows, label = audit_pairs()
    gaz = Gazetteer.from_pairs([(q, a) for q, a, _ in rows])
    rng = random.Random(2024)
    order = list(range(len(rows)))
    rng.shuffle(order)
    shown = errors = 0
    print(f"4. audit of 50 seeded swaps over {label}")
    for i in order:
        question, answer, context = rows[i]
        entity_type = answer_type(question, answer)
        if entity_type not in SWAPPABLE:
            continue
        substitute = gaz.draw(entity_type, [answer], rng, avoid_text=context,
                              variant=answer_variant(entity_type, question, answer))
        swapped, count = swap_in_text(context, answer, substitute) if substitute else (context, 0)
        if substitute is None or count == 0:
            continue
        at = swapped.find(substitute)
        snippet = swapped[max(0, at - 40):at + len(substitute) + 30].replace("\n", " ")
        print(f"   {shown + 1:2d}. [{entity_type:<12}] {question[:60]!r:<64} {answer!r} -> "
              f"{substitute!r}  | ...{snippet}...")
        if alias_remains(swapped, [answer]):
            errors += 1
            print(f"   ERROR: {answer!r} still occurs after the swap")
        shown += 1
        if shown == 50:
            break
    assert shown == 50, f"only {shown} swappable rows to audit"
    print(f"   audit errors (original left in the swapped passage): {errors} of {shown}")
    assert errors == 0


def main():
    check_typing()
    check_draw()
    check_swap()
    check_aliases()
    print_audit()
    print("all entity swap checks passed")


if __name__ == "__main__":
    main()
