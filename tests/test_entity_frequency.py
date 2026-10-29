"""Exact token sequence counts from the rolling hash (``scripts/entity_frequency.py``).

1. Counts equal a brute force scan on a random corpus, for lengths 1 to 6, including sequences that
   overlap themselves ([5, 5] in [5, 5, 5] is 2) and duplicated and empty queries.
2. Planted sequences straddling every chunk boundary are counted once, at several chunk sizes.
3. A tiny modulus makes nearly every window collide with some target; the exact check still gives
   the brute force counts.
4. ``answer_frequencies`` sums the bare and the leading space tokenizations and its cache is reused.

No GPU, no tokenizer file, no network.
"""
import os, sys, tempfile, shutil, random
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from scripts.entity_frequency import answer_frequencies, answer_variants, count_token_sequences


def brute(tokens, seq):
    n = len(seq)
    if n == 0:
        return 0
    return sum(1 for i in range(len(tokens) - n + 1) if list(tokens[i:i + n]) == list(seq))


class FakeTokenizer:
    """Letters map to ids; a leading space is one extra token, like a space-prefixed word piece."""

    def __call__(self, text, add_special_tokens=False):
        ids = []
        for ch in text:
            ids.append(1 if ch == " " else 10 + ord(ch) % 40)
        return {"input_ids": ids}


def write_bin(path, tokens):
    np.asarray(tokens, dtype=np.uint16).tofile(path)


def main():
    tmp = tempfile.mkdtemp(prefix="entfreq_")
    try:
        rng = random.Random(0)
        tokens = [rng.randrange(6) for _ in range(5000)]
        path = os.path.join(tmp, "corpus.bin")
        write_bin(path, tokens)
        queries = [[0], [1, 2], [3, 3], [4, 4, 4], [0, 1, 2, 3], [5, 0, 5, 0, 5], [1, 2, 3, 4, 5, 0],
                   [2], [1, 2], [], [5, 5, 5, 5, 5, 5, 5]]
        got = count_token_sequences(path, queries, chunk_tokens=1000)
        want = [brute(tokens, q) for q in queries]
        assert got == want, (got, want)
        assert want[3] > 0 and want[2] > 0
        write_bin(os.path.join(tmp, "self.bin"), [5, 5, 5, 5, 9, 5, 5])
        assert count_token_sequences(os.path.join(tmp, "self.bin"), [[5, 5], [5, 5, 5]]) == [4, 2]
        print("1. counts equal brute force incl. self overlap, duplicates, empty query     PASS")

        planted = [7] * 4000
        needle = [11, 12, 13, 14, 15, 16, 17]
        for chunk in (50, 64, 97, 128, 4000):
            data = list(planted)
            spots = [0, 49, 50, 63, 64 - 3, 96, 97 - 4, 127, 1000, 3993]
            placed = []
            for s in spots:
                if all(abs(s - p) >= len(needle) for p in placed) and s + len(needle) <= len(data):
                    data[s:s + len(needle)] = needle
                    placed.append(s)
            write_bin(path, data)
            got = count_token_sequences(path, [needle, needle[:3], needle[4:]], chunk_tokens=chunk)
            want = [brute(data, needle), brute(data, needle[:3]), brute(data, needle[4:])]
            assert got == want and got[0] == len(placed), (chunk, got, want, len(placed))
        print("2. sequences straddling chunk boundaries are counted once                   PASS")

        tokens = [rng.randrange(4) for _ in range(3000)]
        write_bin(path, tokens)
        queries = [[0, 1], [2, 3, 0], [1, 1, 1], [3], [0, 1, 2, 3, 0, 1]]
        for modulus in (7, 13, 101):
            got = count_token_sequences(path, queries, chunk_tokens=500, modulus=modulus)
            assert got == [brute(tokens, q) for q in queries], (modulus, got)
        print("3. forced hash collisions (modulus 7, 13, 101) still give exact counts        PASS")

        very_common = [rng.randrange(3) for _ in range(20000)]
        write_bin(path, very_common)
        queries = [[0], [1], [2], [0, 1], [2, 2], [0, 1, 2]]
        want = [brute(very_common, q) for q in queries]
        for modulus in (None, 5):
            got = count_token_sequences(path, queries, chunk_tokens=3000, modulus=modulus)
            assert got == want and want[0] > 5000, (modulus, got, want)
        print("3b. very common single tokens count exactly, also under collisions           PASS")

        tok = FakeTokenizer()
        text_ids = tok("abc")["input_ids"]
        spaced_ids = tok(" abc")["input_ids"]
        corpus = text_ids + [3] + spaced_ids + text_ids + spaced_ids + [3] + text_ids
        write_bin(path, corpus)
        assert len(answer_variants(tok, "abc")) == 2
        cache_dir = os.path.join(tmp, "cache")
        counts = answer_frequencies(path, ["abc", "zzz", "abc"], tok, cache_dir=cache_dir)
        want = brute(corpus, text_ids) + brute(corpus, spaced_ids)
        assert counts == {"abc": want, "zzz": 0} and want >= 5, counts
        cached = [f for f in os.listdir(cache_dir) if f.endswith(".json")]
        assert len(cached) == 1

        class Exploding(FakeTokenizer):
            def __call__(self, *a, **k):
                raise AssertionError("tokenizer used although everything was cached")

        assert answer_frequencies(path, ["abc", "zzz"], Exploding(), cache_dir=cache_dir)["abc"] == want
        print("4. bare plus leading space counted, cache reused                             PASS")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("all entity frequency checks passed")


if __name__ == "__main__":
    main()
