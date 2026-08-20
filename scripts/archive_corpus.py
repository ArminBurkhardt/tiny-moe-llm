"""Archive a prepared corpus so it never has to be downloaded and tokenized again.

``scripts/prepare_data.py`` and ``scripts/prepare_sft_data.py`` are both *destructive by design*:
each downloads one source shard at a time, appends it to ``{split}.bin``/``.idx``, then deletes the
shard, so peak disk stays bounded by the output rather than the source corpora. The consequence is
that the only surviving copy of a build is the output itself, and the next build of the same split
overwrites it in place. Rebuilding then means re-downloading hundreds of GB and re-tokenizing it.

This script is the missing copy. It packs a split's ``.bin``/``.idx``/``.mask`` into one archive
under ``data/archives/`` with a JSON sidecar recording what was in it, and restores it back into
``data/prepared/``. Nothing else in the repo reads these archives -- restore is an explicit step,
so a stale archive can never silently shadow a fresh build.

The sidecar carries two kinds of provenance and keeps them apart on purpose:

  * **measured** -- byte sizes, sha256, and the token/document counts read straight out of the
    files being packed. Always true of the archive.
  * ``manifest_record`` -- the matching slice of ``manifest.json`` (``data_prep.{split}`` for the
    pretraining phases, ``{profile}_prep`` for an SFT/repair build). This is what the *builder*
    reported, and it is not necessarily about these files: the dev box's ``phase1.bin`` is a small
    local stand-in while the manifest describes the 24.7B-token build from the rented box. ``list``
    prints both counts so a mismatch is visible rather than assumed away.

Usage::

    python scripts/archive_corpus.py pack --all           # every split in data/prepared
    python scripts/archive_corpus.py pack repair_train repair_val
    python scripts/archive_corpus.py list
    python scripts/archive_corpus.py restore repair_train --force

Archives are plain ``.tar.gz``, so ``tar xzf`` works if this script ever doesn't.
"""

import argparse
import hashlib
import json
import os
import sys
import tarfile
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import BASE_DIR, TOKENIZER_DIR, logger

DEFAULT_DATA_DIR = os.path.join(BASE_DIR, "data", "prepared")
DEFAULT_ARCHIVE_DIR = os.path.join(BASE_DIR, "data", "archives")
MANIFEST_PATH = os.path.join(BASE_DIR, "manifest.json")

# a split is these files. .bin and .idx are required and meaningless apart; .mask only exists for
# SFT-style corpora, where the supervised region cannot be derived from position.
REQUIRED_SUFFIXES = (".bin", ".idx")
OPTIONAL_SUFFIXES = (".mask",)

CHUNK = 8 << 20


def sha256_file(path):
    """Stream a file through sha256 without holding it in memory.

    Args:
        path: file to hash.

    Returns:
        Lowercase hex digest.
    """
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            block = f.read(CHUNK)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def split_files(data_dir, split):
    """Resolve the on-disk files belonging to one split.

    Args:
        data_dir: directory holding the prepared corpus.
        split: split stem, e.g. ``phase1`` or ``repair_train``.

    Returns:
        List of absolute paths, required suffixes first.

    Raises:
        FileNotFoundError: if a required suffix is missing.
    """
    paths = []
    for suffix in REQUIRED_SUFFIXES:
        path = os.path.join(data_dir, split + suffix)
        if not os.path.isfile(path):
            raise FileNotFoundError(f"{path} does not exist -- {split} is not a prepared split")
        paths.append(path)
    for suffix in OPTIONAL_SUFFIXES:
        path = os.path.join(data_dir, split + suffix)
        if os.path.isfile(path):
            paths.append(path)
    # the prepare scripts' resume sidecar. worth carrying: it records which source file/row each
    # source stopped at, which is the only record of *where* an interrupted build got to.
    for name in (f"_prepare_state_{split}.json", f"_prepare_state_{split.rsplit('_', 1)[0]}.json"):
        path = os.path.join(data_dir, name)
        if os.path.isfile(path) and path not in paths:
            paths.append(path)
            break
    return paths


def discover_splits(data_dir):
    """Every split stem present in a prepared directory, sorted.

    Args:
        data_dir: directory holding the prepared corpus.

    Returns:
        Sorted list of split stems (those with both a ``.bin`` and an ``.idx``).
    """
    stems = []
    for name in os.listdir(data_dir):
        if not name.endswith(".idx"):
            continue
        stem = name[: -len(".idx")]
        if os.path.isfile(os.path.join(data_dir, stem + ".bin")):
            stems.append(stem)
    return sorted(stems)


def measure(data_dir, split):
    """Read the split's real token and document counts off the files themselves.

    The bin file is a flat uint16 token stream and the idx file is uint64 document-start offsets
    plus one trailing entry, so both counts are exact from the byte sizes alone -- no need to open
    a memmap, which matters when the file is 32GB.

    Args:
        data_dir: directory holding the prepared corpus.
        split: split stem.

    Returns:
        Dict with ``tokens``, ``documents`` and, when a mask is present, ``supervised_tokens``.
    """
    bin_bytes = os.path.getsize(os.path.join(data_dir, split + ".bin"))
    idx_bytes = os.path.getsize(os.path.join(data_dir, split + ".idx"))
    stats = {"tokens": bin_bytes // 2, "documents": max(idx_bytes // 8 - 1, 0)}

    mask_path = os.path.join(data_dir, split + ".mask")
    if os.path.isfile(mask_path):
        # uint8 per token, 1 = supervised. counted by streaming rather than memmapped so this stays
        # flat in memory regardless of corpus size.
        supervised = 0
        with open(mask_path, "rb") as f:
            while True:
                block = f.read(CHUNK)
                if not block:
                    break
                supervised += sum(block)
        stats["supervised_tokens"] = supervised
    return stats


def manifest_record(split):
    """The slice of ``manifest.json`` that describes how this split was built, if any.

    Args:
        split: split stem.

    Returns:
        Dict, or None when the manifest is absent or has no matching key. The pretraining phases
        live under ``data_prep.{phase}``; SFT-style builds under ``{profile}_prep``, keyed by the
        profile rather than the split, so ``repair_train`` and ``repair_val`` share one record.
    """
    if not os.path.isfile(MANIFEST_PATH):
        return None
    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    if split in manifest.get("data_prep", {}):
        record = dict(manifest["data_prep"][split])
        record["_source_key"] = f"data_prep.{split}"
        return record

    profile = split.rsplit("_", 1)[0] if split.rsplit("_", 1)[-1] in ("train", "val") else split
    key = f"{profile}_prep"
    if key in manifest:
        record = dict(manifest[key])
        # the holdout hash list is thousands of entries and already lives in manifest.json; the
        # archive only needs to say that the build honoured it.
        record.pop("smoltalk2_holdout_hashes", None)
        record["_source_key"] = key
        return record
    return None


def pack(split, data_dir, archive_dir, force=False, compress=True):
    """Write one split to ``{archive_dir}/{split}.tar.gz`` plus a JSON sidecar.

    The archive is written to a ``.tmp`` path and renamed only once the sidecar is complete, so an
    interrupted pack can never leave a truncated archive that looks finished.

    Args:
        split: split stem to pack.
        data_dir: directory holding the prepared corpus.
        archive_dir: destination directory.
        force: overwrite an existing archive of the same name.
        compress: gzip the tar. .mask files are near-all-zero and compress enormously; token
            streams give roughly a third back.

    Returns:
        Path to the written archive.
    """
    paths = split_files(data_dir, split)
    suffix = ".tar.gz" if compress else ".tar"
    archive_path = os.path.join(archive_dir, split + suffix)
    sidecar_path = os.path.join(archive_dir, split + ".json")

    if os.path.exists(archive_path) and not force:
        raise FileExistsError(f"{archive_path} already exists -- pass --force to overwrite")

    os.makedirs(archive_dir, exist_ok=True)
    raw_bytes = sum(os.path.getsize(p) for p in paths)
    logger.info(f"packing {split}: {len(paths)} files, {raw_bytes / 1e6:,.1f} MB raw")

    members = []
    for path in paths:
        members.append({
            "name": os.path.basename(path),
            "bytes": os.path.getsize(path),
            "sha256": sha256_file(path),
        })

    tmp_path = archive_path + ".tmp"
    started = time.time()
    with tarfile.open(tmp_path, "w:gz" if compress else "w") as tar:
        for path in paths:
            tar.add(path, arcname=os.path.basename(path))
    os.replace(tmp_path, archive_path)

    sidecar = {
        "split": split,
        "archive": os.path.basename(archive_path),
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "tokenizer_dir": TOKENIZER_DIR,
        "measured": measure(data_dir, split),
        "members": members,
        "raw_bytes": raw_bytes,
        "archive_bytes": os.path.getsize(archive_path),
        "manifest_record": manifest_record(split),
    }
    with open(sidecar_path + ".tmp", "w", encoding="utf-8") as f:
        json.dump(sidecar, f, indent=2)
    os.replace(sidecar_path + ".tmp", sidecar_path)

    logger.info(
        f"  -> {archive_path} ({sidecar['archive_bytes'] / 1e6:,.1f} MB, "
        f"{sidecar['archive_bytes'] / max(raw_bytes, 1):.0%} of raw) in {time.time() - started:.0f}s"
    )
    return archive_path


def load_sidecars(archive_dir):
    """Every sidecar in an archive directory, split -> dict.

    Args:
        archive_dir: directory written by :func:`pack`.

    Returns:
        Dict of split stem to sidecar contents, skipping sidecars whose archive is gone.
    """
    if not os.path.isdir(archive_dir):
        return {}
    found = {}
    for name in sorted(os.listdir(archive_dir)):
        if not name.endswith(".json"):
            continue
        with open(os.path.join(archive_dir, name), "r", encoding="utf-8") as f:
            sidecar = json.load(f)
        if os.path.isfile(os.path.join(archive_dir, sidecar.get("archive", ""))):
            found[sidecar["split"]] = sidecar
    return found


def do_list(archive_dir):
    """Print what is archived, measured counts beside what the builder claimed.

    Args:
        archive_dir: directory written by :func:`pack`.
    """
    sidecars = load_sidecars(archive_dir)
    if not sidecars:
        print(f"no archives in {archive_dir}")
        return

    print(f"{archive_dir}\n")
    header = f"{'split':<16} {'tokens':>15} {'docs':>12} {'size':>10}  created"
    print(header)
    print("-" * len(header))
    for split, sidecar in sidecars.items():
        m = sidecar["measured"]
        print(f"{split:<16} {m['tokens']:>15,} {m['documents']:>12,} "
              f"{sidecar['archive_bytes'] / 1e6:>9,.0f}M  {sidecar['created']}")
        if "supervised_tokens" in m:
            share = m["supervised_tokens"] / max(m["tokens"], 1)
            print(f"{'':<16} {m['supervised_tokens']:>15,} supervised ({share:.1%})")
        record = sidecar.get("manifest_record") or {}
        claimed = record.get("realized_tokens") or record.get("total_tokens")
        if claimed and abs(claimed - m["tokens"]) > max(m["tokens"] * 0.01, 1000):
            # not an error: the dev box's phase files are a local stand-in while the manifest
            # describes the rented box's build. printed so nobody reads one for the other.
            print(f"{'':<16} note: {record['_source_key']} reports {claimed:,} tokens "
                  f"-- this archive is a different build")
    print()


def restore(split, data_dir, archive_dir, force=False):
    """Extract one archived split back into ``data_dir`` and verify every member's sha256.

    Args:
        split: split stem to restore.
        data_dir: destination prepared directory.
        archive_dir: directory written by :func:`pack`.
        force: overwrite files that already exist in ``data_dir``.

    Raises:
        FileNotFoundError: no archive for that split.
        FileExistsError: a target file exists and ``force`` is not set.
        ValueError: an extracted file's sha256 does not match the sidecar.
    """
    sidecars = load_sidecars(archive_dir)
    if split not in sidecars:
        raise FileNotFoundError(
            f"no archive for {split} in {archive_dir} (have: {', '.join(sidecars) or 'nothing'})")
    sidecar = sidecars[split]
    archive_path = os.path.join(archive_dir, sidecar["archive"])

    existing = [m["name"] for m in sidecar["members"]
                if os.path.exists(os.path.join(data_dir, m["name"]))]
    if existing and not force:
        raise FileExistsError(
            f"{', '.join(existing)} already in {data_dir} -- pass --force to overwrite")

    os.makedirs(data_dir, exist_ok=True)
    logger.info(f"restoring {split} from {archive_path}")
    with tarfile.open(archive_path, "r:*") as tar:
        for member in tar.getmembers():
            # flat archives by construction; refuse anything that would escape data_dir anyway.
            if os.path.basename(member.name) != member.name or not member.isfile():
                raise ValueError(f"unexpected member in {archive_path}: {member.name}")
        tar.extractall(data_dir, filter="data")

    for member in sidecar["members"]:
        path = os.path.join(data_dir, member["name"])
        actual = sha256_file(path)
        if actual != member["sha256"]:
            raise ValueError(f"{path} sha256 {actual} != archived {member['sha256']}")
        logger.info(f"  {member['name']}: {member['bytes'] / 1e6:,.1f} MB, sha256 ok")

    m = sidecar["measured"]
    logger.info(f"{split} restored: {m['tokens']:,} tokens, {m['documents']:,} documents")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("command", choices=("pack", "list", "restore"))
    parser.add_argument("splits", nargs="*",
                        help="split stems, e.g. phase1 repair_train. omit with --all")
    parser.add_argument("--all", action="store_true",
                        help="pack: every split found in --data-dir")
    parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    parser.add_argument("--archive-dir", default=DEFAULT_ARCHIVE_DIR)
    parser.add_argument("--force", action="store_true",
                        help="pack: overwrite an existing archive. restore: overwrite prepared files")
    parser.add_argument("--no-compress", action="store_true",
                        help="pack: write a plain tar (faster, ~3x the disk)")
    args = parser.parse_args()

    if args.command == "list":
        do_list(args.archive_dir)
        return

    splits = args.splits
    if args.command == "pack" and args.all:
        splits = discover_splits(args.data_dir)
        if not splits:
            raise SystemExit(f"no prepared splits in {args.data_dir}")
        logger.info(f"found {len(splits)} split(s): {', '.join(splits)}")
    if not splits:
        raise SystemExit(f"{args.command} needs at least one split name (or --all for pack)")

    for split in splits:
        if args.command == "pack":
            pack(split, args.data_dir, args.archive_dir,
                 force=args.force, compress=not args.no_compress)
        else:
            restore(split, args.data_dir, args.archive_dir, force=args.force)

    if args.command == "pack":
        do_list(args.archive_dir)


if __name__ == "__main__":
    main()
