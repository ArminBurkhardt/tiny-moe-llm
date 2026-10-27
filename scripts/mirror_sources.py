"""Feed the filler build on a cluster whose compute nodes have no internet.

Three jobs, one on each side of the gap.

On a machine with internet:

    python scripts/mirror_sources.py list --out listing.json [--fineweb-prefix data/] [--margin 0.10] \\
        [the driver's plan flags: --scale --total --chain-tokens --qa-target --phase2-tokens ...]
    python scripts/mirror_sources.py push --listing listing.json --remote USER@HOST:/path/inbox \\
        --window-gb 120 [--scratch DIR] [--transfer-limit-gb 2000]
    python scripts/mirror_sources.py stage-small --hf-home DIR

``list`` reads every filler source's files at one pinned revision with the filters and the sort
order ``prepare_data.py`` uses, records their sizes, and works out how many files each source needs
for the plan (the driver's token yields per file, plus a margin). ``push`` downloads those files in
consumption order and rsyncs them into the cluster's inbox. ``stage-small`` snapshots the small
repos the other stages read (question sets, the replay chat sets, bge, the tokenizer) into one
directory to copy across once.

On the cluster, ``build_run2_corpus.py wrap-prepare-data --mirror-inbox DIR --mirror-listing
listing.json`` uses ``MirrorListing``, ``FakeHfApi`` and ``InboxFetcher`` below in place of the Hub
(no Hub call: the listing replaces ``list_repo_files``, a wait on the inbox replaces the download);
``wrap-hub --stage-dir`` uses ``install_offline_hub`` for the small repos.

Consumption order. ``prepare_data`` lists each source sorted, a phase takes files from the front
until its token target, and the next phase skips the files the earlier ones used (the wrapper's
``--exclude-files-of``), so per source the sorted list is consumed front to back across phase1,
phase2 and ir. A source's partly used last file counts as used by its phase. The sources run in
lockstep inside a phase (smooth weighted round robin), and one source blocked on a missing file
blocks all of them, so ``push`` is demand driven: a source gets a new file only while fewer than
its cap (``source_cap``: the tail the consumer keeps plus a little lookahead) of its files sit in
the inbox, and among the sources that qualify the one furthest behind its own plan goes first.
``needed`` (the plan's file count plus a margin) orders the transfer and feeds the transfer
estimate; it is not a stop. Once a source's needed files are sent the pusher keeps going down the
listing for as long as the consumer keeps asking (the inbox holds no more than the tail plus the
file being read), so a wrong tokens-per-file estimate costs a longer wait and a few extra files, not
a hang. The pusher ends when every listed file is sent, or when the driver's filler stage drops
``DONE_MARK`` into the inbox.

Deletion on the cluster is stateless. The resume state lags the generator by at most
``checkpoint_docs`` documents plus one tokenize batch (``LAG_DOCS``, 2256 at the defaults), which is
under one file, so a restart needs the file the generator was on or the one before: the tail is 2
files. Whenever the consumer opens file ``i`` of a repo it deletes every inbox file of that repo
whose listing index is below ``i - tail``, whoever left it there (a killed process's files do not
pile up). A repo's tail grows when ``InboxFetcher`` sees a file with fewer rows than ``LAG_DOCS``;
the pusher's cap for the repo is always tail + 1 + ``AHEAD_FILES`` from the same rule
(``source_tail`` estimates the shortest file's rows from the listing's sizes and the cap follows),
and a tail the cap cannot hold stops the build with a message instead of hanging.
"""
import argparse
import json
import math
import os
import shlex
import shutil
import subprocess
import sys
import time
from types import SimpleNamespace
from typing import Callable, Dict, Iterable, List, Optional, Sequence

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
LISTING_VERSION = 1
LAG_DOCS = 2000 + 256
TAIL_FILES = 2
AHEAD_FILES = 2
MAX_PRESENT = TAIL_FILES + 1 + AHEAD_FILES     # the cap of a repo whose files all hold more than LAG_DOCS rows
ROW_TOKENS_ASSUMED = 3000      # tokens per document, high on purpose: it sets the shortest file's estimated rows
DONE_MARK = ".filler_done"     # the driver's filler stage drops it in the inbox when the consumer is finished
TRANSFER_LIMIT_BYTES = 2e12
STAGE_INDEX = "index.json"

# repo, path prefixes (the builders' own), suffixes, substring a file must contain
SMALL_DATASETS = [
    ("rajpurkar/squad_v2", ("squad_v2/train", "squad_v2/validation"), (".parquet",), None),
    ("hotpotqa/hotpot_qa", ("distractor/train", "distractor/validation"), (".parquet",), None),
    ("HuggingFaceTB/smoltalk2", ("SFT/",), (".parquet",), "_no_think"),
    ("HuggingFaceH4/ultrachat_200k", ("data/train_sft",), (".parquet",), None),
    ("HuggingFaceH4/no_robots", ("data/train",), (".parquet",), None),
    ("florin-hf/nq_open_gold", ("test_dataset.json",), (".json",), None),
    ("mandarjoshi/trivia_qa", ("rc.wikipedia/validation",), (".parquet",), None),
]
BGE_REPO = "BAAI/bge-small-en-v1.5"
TOKENIZER_REPO = "ikeafisch4/DeepSeek-V4-Pro-tokenizer-65536"


def repo_dir(repo_id: str) -> str:
    return repo_id.replace("/", "__")


def rel_path(repo_id: str, name: str) -> str:
    return f"{repo_dir(repo_id)}/{name}"


def tail_for_rows(min_rows: Optional[int]) -> int:
    """Files a consumer keeps behind the one it reads, given the fewest rows any file holds."""
    if min_rows is None or min_rows >= LAG_DOCS:
        return TAIL_FILES
    return 1 + int(math.ceil(LAG_DOCS / max(1, min_rows)))


def cap_for_tail(tail: int) -> int:
    """Files of one repo the inbox may hold: the file being read, the tail behind it, lookahead in front."""
    return tail + 1 + AHEAD_FILES


def source_tail(src: dict) -> int:
    """A repo's tail from its listing entry: the shortest file's rows, estimated from sizes.

    Rows of a file are taken as ``per_file_tokens`` scaled by the file's share of the mean size, over
    ``ROW_TOKENS_ASSUMED`` tokens per document.
    """
    sizes = [f["size"] for f in src["files"] if f.get("size")]
    if not sizes:
        return TAIL_FILES
    mean = sum(sizes) / len(sizes)
    rows = src["per_file_tokens"] * min(sizes) / mean / ROW_TOKENS_ASSUMED
    return tail_for_rows(int(rows))


def check_caps(listing: dict) -> None:
    """Raise when a source's recorded cap cannot hold its tail (a hand edited or stale listing)."""
    for key, src in listing["sources"].items():
        tail = src.get("tail", source_tail(src))
        cap = src.get("max_present", cap_for_tail(tail))
        if cap < cap_for_tail(tail) - AHEAD_FILES + 1:
            raise RuntimeError(f"{key}: the cap of {cap} files cannot hold a tail of {tail} (needs more than "
                               f"{tail + 1}); relist")


def filter_files(spec, names: Iterable[str]) -> List[str]:
    """The expression ``prepare_data.main`` applies to a repo's file list (same filters, same sort)."""
    return sorted(
        f for f in names
        if f.startswith(spec.file_prefix) and f.endswith(spec.file_suffix)
        and (spec.file_filter is None or spec.file_filter(f))
    )


# ------------------------------------------------------------------------------------ listing

def build_listing(specs, api, plan: Dict[str, dict], margin: float, fineweb_prefix: Optional[str],
                  created: Optional[str] = None) -> dict:
    """Listing of every filler source at a pinned revision.

    Args:
        specs: ``prepare_data.SOURCES`` (objects with key, repo_id, file_prefix, file_suffix,
            file_filter, gated and the phase weights).
        api: has ``dataset_info(repo, files_metadata=True)`` returning ``sha`` and ``siblings``.
        plan: per source key ``{"per_file": tokens, "need": {phase: tokens}}`` from the driver.
        margin: extra share of files on top of the estimate.
        fineweb_prefix: replaces fineweb's file prefix (the driver's widening).
        created: timestamp text, injected for tests.
    """
    out = {"version": LISTING_VERSION, "created": created or time.strftime("%Y-%m-%dT%H:%M:%S"),
           "margin": margin, "sources": {}}
    for spec in specs:
        if spec.phase1_weight == 0.0 and spec.phase2_weight == 0.0:
            continue
        if spec.key == "fineweb" and fineweb_prefix:
            spec.file_prefix = fineweb_prefix
        info = api.dataset_info(spec.repo_id, files_metadata=True)
        sizes = {s.rfilename: int(getattr(s, "size", 0) or 0) for s in info.siblings}
        names = filter_files(spec, sizes)
        if not names:
            raise RuntimeError(f"no files matched {spec.key} under {spec.file_prefix!r}")
        p = plan[spec.key]
        phase_files, cum = [], 0.0
        for phase in ("phase1", "phase2", "ir"):
            cum += p["need"].get(phase, 0.0)
            phase_files.append(int(math.ceil(cum * (1.0 + margin) / p["per_file"])) if cum else 0)
        short = phase_files[-1] > len(names)
        phase_files = [min(n, len(names)) for n in phase_files]
        needed = min(len(names), phase_files[-1] + 1)
        entry = {
            "repo_id": spec.repo_id, "revision": info.sha, "prefix": spec.file_prefix,
            "files": [{"name": n, "size": sizes[n]} for n in names],
            "per_file_tokens": p["per_file"], "need_tokens": p["need"],
            "phase_files": phase_files, "needed": needed,
            "short": short,
        }
        entry["tail"] = source_tail(entry)
        entry["max_present"] = cap_for_tail(entry["tail"])
        out["sources"][spec.key] = entry
    check_caps(out)
    return out


def needed_files(listing: dict, key: str) -> List[dict]:
    src = listing["sources"][key]
    return src["files"][:src["needed"]]


def total_bytes(listing: dict) -> Dict[str, int]:
    return {k: sum(f["size"] for f in needed_files(listing, k)) for k in listing["sources"]}


def phase_of(src: dict, index: int) -> int:
    for p, cum in enumerate(src["phase_files"]):
        if index < cum:
            return p
    return len(src["phase_files"]) - 1


def consumption_order(listing: dict) -> List[dict]:
    """Every needed file once, in the order a lockstep consumer would reach it.

    The key is (phase, fraction of the phase's files of this source), so the sources progress
    together inside a phase.
    """
    rows = []
    for key, src in listing["sources"].items():
        starts = [0] + src["phase_files"][:-1]
        for i, f in enumerate(needed_files(listing, key)):
            p = phase_of(src, i)
            count = max(1, src["phase_files"][p] - starts[p])
            frac = (i - starts[p] + 1) / count
            rows.append({"source": key, "repo_id": src["repo_id"], "name": f["name"], "size": f["size"],
                         "revision": src["revision"], "index": i, "phase": p,
                         "key": (p, min(frac, 1.0), key)})
    rows.sort(key=lambda r: r["key"])
    return rows


def print_listing_summary(listing: dict, limit: float = TRANSFER_LIMIT_BYTES) -> int:
    sizes = total_bytes(listing)
    print(f"{'source':14s} {'files':>6s} {'needed':>7s} {'p1/p2/ir':>14s} {'GB':>9s}")
    for key, src in listing["sources"].items():
        pf = "/".join(str(n) for n in src["phase_files"])
        flag = "  SHORT: the source has fewer files than the plan needs" if src["short"] else ""
        print(f"{key:14s} {len(src['files']):6d} {src['needed']:7d} {pf:>14s} {sizes[key] / 1e9:9.1f}{flag}")
    total = sum(sizes.values())
    print(f"transfer {total / 1e9:,.1f} GB of {limit / 1e9:,.0f} GB allowed ({total / limit:.0%})"
          + ("  OVER THE LIMIT" if total > limit else ""))
    return int(total)


# ------------------------------------------------------------------------------------ push

class Pusher:
    """Demand driven, resumable mirror of the listing's files into a remote inbox.

    Args:
        listing: the listing dict.
        state_path: local json with the files already pushed and the bytes sent.
        fetch: ``fetch(entry) -> local path`` downloads one file.
        send: ``send(local_path, relpath, size)`` puts it in the inbox and returns when complete.
        remote_ls: ``() -> {relpath: size}`` every file in the inbox, rsync temp files included.
        window_bytes: bound on the inbox's bytes.
        transfer_limit: bound on the total bytes sent over all runs.
        max_present: ``None`` takes each source's cap from the listing (tail + 1 + lookahead); an
            integer replaces it for every source and must exceed every tail + 1.
        remove: deletes a local file.
        log: line printer.
    """

    def __init__(self, listing: dict, state_path: str, fetch: Callable, send: Callable,
                 remote_ls: Callable, window_bytes: float, transfer_limit: float = TRANSFER_LIMIT_BYTES,
                 max_present: Optional[int] = None, remove: Callable = os.remove, log: Callable = print):
        self.listing, self.state_path = listing, state_path
        self.fetch, self.send, self.remote_ls = fetch, send, remote_ls
        self.window, self.limit = window_bytes, transfer_limit
        self.remove, self.log = remove, log
        check_caps(listing)
        self.tails = {k: s.get("tail", source_tail(s)) for k, s in listing["sources"].items()}
        self.caps = {k: (max_present if max_present is not None else s.get("max_present", cap_for_tail(self.tails[k])))
                     for k, s in listing["sources"].items()}
        for k, cap in self.caps.items():
            if cap < self.tails[k] + 2:
                raise ValueError(f"{k}: a cap of {cap} files cannot hold a tail of {self.tails[k]}; the consumer "
                                 f"would wait for a file the pusher may not send (needs {self.tails[k] + 2} or more)")
        self.order = consumption_order(listing)
        self.keys = {(r["source"], r["index"]): r["key"] for r in self.order}
        self.rel_keys = {rel_path(r["repo_id"], r["name"]): r["key"] for r in self.order}
        self.state = {"done": [], "pushed_bytes": 0}
        if os.path.exists(state_path):
            with open(state_path) as f:
                self.state.update(json.load(f))
        self.done = set(self.state["done"])

    def worst_case_window(self) -> float:
        """Bytes the inbox holds when every source sits at its cap with its largest files."""
        return sum(self.caps[k] * max((f["size"] for f in src["files"]), default=0)
                   for k, src in self.listing["sources"].items())

    def _save(self) -> None:
        self.state["done"] = sorted(self.done)
        tmp = self.state_path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(self.state, f)
        os.replace(tmp, self.state_path)

    def _next_unsent(self, src: dict, present: Dict[str, int]):
        """The first listed file of a source not yet sent. A file the inbox already holds at its full size
        (a pusher that died between the send and the save) counts as sent, without a second send."""
        for i, f in enumerate(src["files"]):
            rel = rel_path(src["repo_id"], f["name"])
            if rel in self.done:
                continue
            if f["size"] > 0 and present.get(rel) == f["size"]:
                self.done.add(rel)
                self._save()
                self.log(f"{rel} is in the inbox already, counted as sent")
                continue
            return i, f, rel
        return None

    def step(self) -> str:
        """Push at most one file. Returns "pushed", "wait" or "finished"."""
        present = self.remote_ls()
        if any(os.path.basename(rel) == DONE_MARK for rel in present):
            return "finished"
        used = sum(present.values())
        counts: Dict[str, int] = {}
        for rel in present:
            if os.path.basename(rel).startswith("."):
                continue
            counts[rel.split("/", 1)[0]] = counts.get(rel.split("/", 1)[0], 0) + 1
        eligible = []
        pending = False
        earliest = None
        for key, src in self.listing["sources"].items():
            nxt = self._next_unsent(src, present)
            if nxt is None:
                continue
            pending = True
            i, f, rel = nxt
            k = self.keys.get((key, i), (9, float(i), key))
            if earliest is None or k < earliest:
                earliest = k
            n = counts.get(repo_dir(src["repo_id"]), 0)
            row = {"source": key, "repo_id": src["repo_id"], "name": f["name"], "size": f["size"],
                   "revision": src["revision"], "index": i}
            if i < src["needed"]:
                if n < self.caps[key]:
                    eligible.append((self.keys.get((key, i), (9, 0.0, key)), row, rel))
            elif n < self.tails[key] + 2:
                # past the estimate: only while the consumer sits on the last file sent
                eligible.append(((9, float(i), key), row, rel))
        if not pending:
            return "finished"
        if not eligible:
            return "wait"
        ekey, row, rel = min(eligible, key=lambda x: x[0])
        # the earliest needed file is the one the consumer waits on when every file in the inbox is needed
        # later: other sources' lookahead can fill the window, and a file that waited behind it would never
        # be sent, because nothing there is consumed until it arrives
        present_keys = [self.rel_keys[rel] for rel in present if rel in self.rel_keys]
        starved = ekey == earliest and all(k > earliest for k in present_keys)
        if used + row["size"] > self.window and used > 0 and not starved:
            return "wait"
        if self.state["pushed_bytes"] + row["size"] > self.limit:
            raise RuntimeError(f"transfer limit {self.limit:,.0f} bytes reached after "
                               f"{self.state['pushed_bytes']:,.0f}; {rel} not sent")
        local = self.fetch(row)
        self.send(local, rel, row["size"])
        try:
            self.remove(local)
        except OSError:
            pass
        self.done.add(rel)
        self.state["pushed_bytes"] += row["size"]
        self._save()
        extra = "" if row["index"] < self.listing["sources"][row["source"]]["needed"] else " (past the estimate)"
        self.log(f"pushed {rel} ({row['size'] / 1e9:.2f} GB){extra}, {len(self.done)} files, "
                 f"{self.state['pushed_bytes'] / 1e9:,.1f} GB sent")
        return "pushed"

    def run(self, sleep: Callable = time.sleep, poll_seconds: float = 60.0, max_steps: Optional[int] = None) -> dict:
        w = self.worst_case_window()
        if w > self.window:
            self.log(f"warning: the per source caps can fill {w / 1e9:.0f} GB, "
                     f"more than the window {self.window / 1e9:.0f} GB; the window will throttle first")
        steps = 0
        while True:
            status = self.step()
            if status == "finished":
                return self.state
            steps += 1
            if max_steps is not None and steps >= max_steps:
                return self.state
            if status == "wait":
                sleep(poll_seconds)


def split_remote(remote: str):
    host, sep, path = remote.partition(":")
    if not sep or not path:
        raise SystemExit(f"--remote {remote!r}: expected USER@HOST:/path/inbox")
    return host, path


def ssh_remote_ls(host: str, path: str) -> Callable:
    def ls() -> Dict[str, int]:
        cmd = f"find {shlex.quote(path)} -type f -printf '%P\\t%s\\n' 2>/dev/null; true"
        out = subprocess.run(["ssh", host, cmd], capture_output=True, text=True, check=True).stdout
        res = {}
        for line in out.splitlines():
            name, _, size = line.rpartition("\t")
            if name:
                res[name] = int(size)
        return res
    return ls


def rsync_send(host: str, path: str) -> Callable:
    def send(local: str, rel: str, size: int) -> None:
        dest = f"{path.rstrip('/')}/{rel}"
        subprocess.run(["ssh", host, f"mkdir -p {shlex.quote(os.path.dirname(dest))}"], check=True)
        subprocess.run(["rsync", "-a", "--partial", local, f"{host}:{dest}"], check=True)
    return send


def hub_fetch(scratch: str, token: Optional[str]) -> Callable:
    def fetch(row: dict) -> str:
        from huggingface_hub import hf_hub_download
        return hf_hub_download(repo_id=row["repo_id"], filename=row["name"], repo_type="dataset",
                               local_dir=scratch, token=token, revision=row["revision"])
    return fetch


# ------------------------------------------------------------------------------------ cluster side

class MirrorListing:
    """The listing as the cluster sees it: file names, the pinned revision and the tail and cap per repo."""

    def __init__(self, listing: dict):
        self.by_repo = {s["repo_id"]: s for s in listing["sources"].values()}
        self._index = {r: {f["name"]: i for i, f in enumerate(s["files"])} for r, s in self.by_repo.items()}

    def sha(self, repo_id: str) -> str:
        return self.by_repo[repo_id]["revision"]

    def names(self, repo_id: str) -> List[str]:
        return [f["name"] for f in self.by_repo[repo_id]["files"]]

    def size(self, repo_id: str, name: str) -> int:
        for f in self.by_repo[repo_id]["files"]:
            if f["name"] == name:
                return f["size"]
        raise KeyError(name)

    def index(self, repo_id: str, name: str) -> Optional[int]:
        """Position of a file in the repo's listing, None for a name the listing does not hold."""
        return self._index[repo_id].get(name)

    def tail(self, repo_id: str) -> int:
        src = self.by_repo[repo_id]
        return src.get("tail", source_tail(src))

    def cap(self, repo_id: str) -> int:
        src = self.by_repo[repo_id]
        return src.get("max_present", cap_for_tail(self.tail(repo_id)))


class FakeHfApi:
    """Stands in for ``HfApi`` in ``prepare_data.main``: no Hub call, answers from the listing."""

    def __init__(self, listing: MirrorListing, token=None):
        self.listing = listing

    def dataset_info(self, repo_id, *args, **kwargs):
        return SimpleNamespace(sha=self.listing.sha(repo_id))

    def list_repo_files(self, repo_id, *args, **kwargs):
        return self.listing.names(repo_id)


class InboxFetcher:
    """``hf_hub_download`` replacement that waits for a file in the inbox and deletes old ones.

    Deletion keeps no state: opening file ``i`` of a repo deletes every inbox file of that repo whose
    listing index is below ``i - tail``, so the leftovers of a killed process go with the next open.
    The caller's ``os.remove`` of the returned path must be made a no-op (``NoRemoveOs``).

    Args:
        inbox: the directory ``push`` fills, one subdirectory per repo.
        listing: a ``MirrorListing`` for the expected sizes, indices, tails and caps.
        sleep, clock, log: injected for tests.
        wait_log_seconds: how often a wait is logged.
    """

    def __init__(self, inbox: str, listing: MirrorListing, sleep: Callable = time.sleep,
                 clock: Callable = time.time, log: Callable = print, wait_log_seconds: float = 300.0,
                 poll_seconds: float = 5.0):
        self.inbox, self.listing = inbox, listing
        self.sleep, self.clock, self.log = sleep, clock, log
        self.wait_log, self.poll = wait_log_seconds, poll_seconds
        self.last_index: Dict[str, int] = {}
        self.min_rows: Dict[Optional[str], int] = {}
        self.deleted: List[str] = []

    def path(self, repo_id: str, filename: str) -> str:
        return os.path.join(self.inbox, repo_dir(repo_id), filename)

    def tail_for(self, repo_id: Optional[str] = None) -> int:
        """Files kept behind the one being read: the repo's own fewest rows (and any row count noted
        without a repo) against ``LAG_DOCS``, never below the listing's estimate."""
        rows = [v for k, v in self.min_rows.items() if k is None or k == repo_id]
        tail = tail_for_rows(min(rows) if rows else None)
        if repo_id is not None and repo_id in self.listing.by_repo:
            tail = max(tail, self.listing.tail(repo_id))
        return tail

    @property
    def tail(self) -> int:
        return self.tail_for(None)

    def note_rows(self, rows: int, repo_id: Optional[str] = None) -> None:
        """Record a file's row count: that repo's tail grows when a file is shorter than the state's lag.

        Raises:
            RuntimeError: the tail no longer fits the pusher's cap for the repo, which would leave the
                consumer waiting on a file the pusher may not send.
        """
        old = self.min_rows.get(repo_id)
        self.min_rows[repo_id] = rows if old is None else min(old, rows)
        for repo in ([repo_id] if repo_id is not None else list(self.listing.by_repo)):
            if repo in self.listing.by_repo and self.tail_for(repo) + 2 > self.listing.cap(repo):
                raise RuntimeError(f"{repo}: a file of {rows} rows needs a tail of {self.tail_for(repo)}, the "
                                   f"pusher's cap of {self.listing.cap(repo)} files cannot hold it; relist with a "
                                   f"smaller ROW_TOKENS_ASSUMED so the cap grows")

    def download(self, repo_id: str, filename: str, *args, **kwargs) -> str:
        path = self.path(repo_id, filename)
        want = self.listing.size(repo_id, filename)
        t0 = last = self.clock()
        while not (os.path.exists(path) and (not want or os.path.getsize(path) == want)):
            now = self.clock()
            if now - last >= self.wait_log:
                self.log(f"mirror: waiting {now - t0:.0f}s for {repo_id}/{filename} in {self.inbox}")
                last = now
            self.sleep(self.poll)
        idx = self.listing.index(repo_id, filename)
        if idx is not None:
            self.last_index[repo_id] = max(self.last_index.get(repo_id, -1), idx)
            self.drop_below(repo_id, idx - self.tail_for(repo_id))
        return path

    def inbox_files(self, repo_id: str) -> List[str]:
        """Names (listing form) of the finished files of a repo in the inbox; rsync temp files skipped."""
        root = os.path.join(self.inbox, repo_dir(repo_id))
        out = []
        for dirpath, _, files in os.walk(root):
            for name in files:
                if not name.startswith("."):
                    out.append(os.path.relpath(os.path.join(dirpath, name), root).replace(os.sep, "/"))
        return out

    def drop_below(self, repo_id: str, limit: int) -> None:
        """Delete the repo's inbox files with a listing index under ``limit``, whichever process left them."""
        for name in self.inbox_files(repo_id):
            i = self.listing.index(repo_id, name)
            if i is not None and i < limit:
                p = self.path(repo_id, name)
                try:
                    os.remove(p)
                    self.deleted.append(p)
                except OSError:
                    pass

    def clear(self) -> None:
        """At the end of a phase: delete what is behind the highest file opened, keep that file.

        The highest opened file may have been opened without a document of it committed, and the next
        phase then reads it again (it is not in ``files_used``), so it stays; the next phase's opens
        remove it once it is behind the tail.
        """
        for repo_id, top in self.last_index.items():
            self.drop_below(repo_id, top)


class NoRemoveOs:
    """``os`` with ``remove`` ignoring paths under the inbox: ``prepare_data`` deletes its shard itself."""

    def __init__(self, real, inbox: str):
        self._real, self._inbox = real, os.path.abspath(inbox)

    def __getattr__(self, name):
        return getattr(self._real, name)

    def remove(self, path, *args, **kwargs):
        if os.path.abspath(str(path)).startswith(self._inbox + os.sep):
            return None
        return self._real.remove(path, *args, **kwargs)


def install_offline_hub(stage_dir: str, hub=None) -> dict:
    """Make ``HfApi.dataset_info``, ``HfApi.list_repo_files`` and ``hf_hub_download`` read ``stage_dir``.

    ``stage-small`` writes ``stage_dir/index.json`` and the files; a download copies the staged file
    into ``local_dir`` (the builders delete what they download) and returns that path. Call before
    the target script imports ``huggingface_hub`` names.

    Returns:
        The index's ``datasets`` mapping.
    """
    if hub is None:
        import huggingface_hub as hub
    with open(os.path.join(stage_dir, STAGE_INDEX)) as f:
        index = json.load(f)["datasets"]

    def dataset_info(self, repo_id, *args, **kwargs):
        return SimpleNamespace(sha=index[repo_id]["sha"], siblings=[SimpleNamespace(rfilename=n, size=0) for n in index[repo_id]["files"]])

    def list_repo_files(self, repo_id, *args, **kwargs):
        return list(index[repo_id]["files"])

    def hf_hub_download(repo_id, filename, *args, local_dir=None, **kwargs):
        src = os.path.join(stage_dir, repo_dir(repo_id), filename)
        if not os.path.exists(src):
            raise FileNotFoundError(f"{repo_id}/{filename} is not in the staged directory {stage_dir}")
        if local_dir is None:
            return src
        dst = os.path.join(local_dir, filename)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copyfile(src, dst)
        return dst

    hub.HfApi.dataset_info = dataset_info
    hub.HfApi.list_repo_files = list_repo_files
    hub.hf_hub_download = hf_hub_download
    return index


# ------------------------------------------------------------------------------------ stage-small

def small_selection(names: Sequence[str], prefixes, suffixes, must_contain) -> List[str]:
    return sorted(n for n in names if any(n.startswith(p) for p in prefixes) and n.endswith(tuple(suffixes))
                  and (must_contain is None or must_contain in n))


def stage_small(hf_home: str, api, download: Callable, snapshot: Callable, log: Callable = print) -> dict:
    """Download the small repos into ``hf_home``.

    Layout: ``datasets/index.json`` plus ``datasets/<repo>/<file>``, ``hub/`` (the HF cache holding
    bge) and ``tokenizer/``. Point ``HF_HOME`` at ``hf_home`` with ``HF_HUB_OFFLINE=1`` for the model
    loads and ``wrap-hub --stage-dir hf_home/datasets`` for the dataset downloads.

    Args:
        api: has ``dataset_info(repo).sha`` and ``list_repo_files(repo, repo_type, revision)``.
        download: ``download(repo, name, sha, local_dir) -> path``.
        snapshot: ``snapshot(repo, revision, local_dir_or_None, cache_dir_or_None) -> sha``.
    """
    ds_dir = os.path.join(hf_home, "datasets")
    os.makedirs(ds_dir, exist_ok=True)
    index = {"datasets": {}, "models": {}}
    total = 0
    for repo, prefixes, suffixes, must in SMALL_DATASETS:
        sha = api.dataset_info(repo).sha
        names = small_selection(api.list_repo_files(repo, repo_type="dataset", revision=sha), prefixes, suffixes, must)
        if not names:
            raise RuntimeError(f"{repo}: nothing under {prefixes}")
        size = 0
        for n in names:
            path = download(repo, n, sha, os.path.join(ds_dir, repo_dir(repo)))
            size += os.path.getsize(path)
        index["datasets"][repo] = {"sha": sha, "files": names, "bytes": size}
        total += size
        log(f"{repo}: {len(names)} files, {size / 1e9:.2f} GB (revision {sha[:10]})")
    index["models"][BGE_REPO] = snapshot(BGE_REPO, None, None, os.path.join(hf_home, "hub"))
    index["models"][TOKENIZER_REPO] = snapshot(TOKENIZER_REPO, None, os.path.join(hf_home, "tokenizer"), None)
    with open(os.path.join(ds_dir, STAGE_INDEX), "w") as f:
        json.dump(index, f, indent=1)
    total_all = sum(os.path.getsize(os.path.join(r, n)) for r, _, fs in os.walk(hf_home) for n in fs)
    log(f"staged {total_all / 1e9:.2f} GB in {hf_home}; copy it to the cluster, set HF_HOME to it, copy "
        f"{os.path.join(hf_home, 'tokenizer')} to utils.TOKENIZER_DIR")
    return index


# ------------------------------------------------------------------------------------ offline preflight

DATASET_NEEDS = {
    "squad_train": ("rajpurkar/squad_v2", "squad_v2/train"),
    "squad_val": ("rajpurkar/squad_v2", "squad_v2/validation"),
    "hotpot_train": ("hotpotqa/hotpot_qa", "distractor/train"),
    "hotpot_val": ("hotpotqa/hotpot_qa", "distractor/validation"),
    "smoltalk2": ("HuggingFaceTB/smoltalk2", "SFT/"),
    "ultrachat": ("HuggingFaceH4/ultrachat_200k", "data/train_sft"),
    "no_robots": ("HuggingFaceH4/no_robots", "data/train"),
    "nq": ("florin-hf/nq_open_gold", "test_dataset.json"),
    "trivia": ("mandarjoshi/trivia_qa", "rc.wikipedia/validation"),
}
# what each driver stage reads from the Hub or from a model cache (the embedder is bge, the tokenizer is a local dir)
STAGE_NEEDS = {
    "filler": ["listing", "inbox", "tokenizer"],
    "qa": ["squad_train", "hotpot_train", "smoltalk2", "ultrachat", "no_robots", "bge", "tokenizer"],
    "bios_main": ["bge", "tokenizer"], "bios_warm": ["bge", "tokenizer"], "bios_full": ["tokenizer"],
    "val_clean": ["tokenizer"], "rows": ["tokenizer"],
    "chains_plain": ["bge", "tokenizer"], "chains_bridge": ["bge", "tokenizer"],
    "heldout_qa": ["squad_val", "hotpot_val", "bge", "tokenizer"],
    "heldout_chains": ["bge", "tokenizer"],
    "store_openqa": ["nq", "trivia", "hotpot_val", "bge", "tokenizer"],
    "store_edit": ["bge", "tokenizer"],
}


def check_need(need: str, stage_dir: Optional[str], tokenizer_dir: Optional[str], listing: Optional[str],
               inbox: Optional[str]) -> "tuple":
    """Whether one requirement is present, and a short reason when it is not."""
    import glob
    if need == "listing":
        return (bool(listing) and os.path.isfile(listing), f"mirror listing {listing} missing")
    if need == "inbox":
        return (bool(inbox) and os.path.isdir(inbox), f"mirror inbox {inbox} is not a directory")
    if need == "tokenizer":
        d = tokenizer_dir or (os.path.join(stage_dir, "tokenizer") if stage_dir else None)
        if d and os.path.isdir(d) and (os.path.exists(os.path.join(d, "tokenizer.json"))
                                       or os.path.exists(os.path.join(d, "tokenizer_config.json"))):
            return (True, "")
        return (False, f"no tokenizer in {d} (the builders read exactly that path: copy {stage_dir}/tokenizer there "
                       f"or pass --tokenizer-dir)")
    if not stage_dir:
        return (False, "no --offline-stage given")
    if need == "bge":
        hit = glob.glob(os.path.join(stage_dir, "hub", "models--" + BGE_REPO.replace("/", "--"), "snapshots", "*", "config.json"))
        return (bool(hit), f"{BGE_REPO} is not in {stage_dir}/hub (run stage-small)")
    repo, prefix = DATASET_NEEDS[need]
    index_path = os.path.join(stage_dir, "datasets", STAGE_INDEX)
    if not os.path.isfile(index_path):
        return (False, f"{index_path} missing (run stage-small)")
    with open(index_path) as f:
        entry = json.load(f)["datasets"].get(repo)
    if not entry:
        return (False, f"{repo} is not in the staged index")
    names = [n for n in entry["files"] if n.startswith(prefix)]
    missing = [n for n in names if not os.path.exists(os.path.join(stage_dir, "datasets", repo_dir(repo), n))]
    if not names or missing:
        return (False, f"{repo} {prefix}: " + (f"{len(missing)} staged files missing" if names else "no files staged"))
    return (True, "")


def check_stages(stage_names: Sequence[str], stage_dir: Optional[str], tokenizer_dir: Optional[str],
                 listing: Optional[str], inbox: Optional[str]) -> Dict[str, List[tuple]]:
    """Per stage a list of ``(need, ok, reason)``; stages without Hub reads are absent."""
    out = {}
    for name in stage_names:
        needs = STAGE_NEEDS.get(name)
        if needs is None:
            continue
        rows = []
        for n in needs:
            if n in ("listing", "inbox") and not (listing or inbox):
                continue
            ok, why = check_need(n, stage_dir, tokenizer_dir, listing, inbox)
            rows.append((n, ok, why))
        out[name] = rows
    return out


# ------------------------------------------------------------------------------------ cli

def cmd_list(a) -> int:
    sys.path.insert(0, SCRIPT_DIR)
    sys.path.insert(0, os.path.dirname(SCRIPT_DIR))
    import build_run2_corpus as drv
    import scripts.prepare_data as pd
    from huggingface_hub import HfApi
    plan = drv.plan_from_args(a)
    d = drv.derive(plan)
    file_plan = drv.filler_file_need(plan, d)
    listing = build_listing(pd.SOURCES, HfApi(token=os.environ.get("HF_TOKEN")), file_plan, a.margin, a.fineweb_prefix)
    with open(a.out, "w") as f:
        json.dump(listing, f)
    print_listing_summary(listing)
    print(f"wrote {a.out}")
    return 0


def cmd_push(a) -> int:
    with open(a.listing) as f:
        listing = json.load(f)
    host, path = split_remote(a.remote)
    scratch = a.scratch or os.path.join(os.path.dirname(os.path.abspath(a.listing)), "mirror_scratch")
    os.makedirs(scratch, exist_ok=True)
    state = a.state or a.listing + ".pushed.json"
    p = Pusher(listing, state, hub_fetch(scratch, os.environ.get("HF_TOKEN")), rsync_send(host, path),
               ssh_remote_ls(host, path), a.window_gb * 1e9, a.transfer_limit_gb * 1e9)
    print_listing_summary(listing, a.transfer_limit_gb * 1e9)
    p.run(poll_seconds=a.poll_seconds)
    print("all needed files pushed")
    return 0


def cmd_stage_small(a) -> int:
    from huggingface_hub import HfApi, hf_hub_download, snapshot_download
    token = os.environ.get("HF_TOKEN")

    def download(repo, name, sha, local_dir):
        return hf_hub_download(repo_id=repo, filename=name, repo_type="dataset", local_dir=local_dir,
                               token=token, revision=sha)

    def snapshot(repo, revision, local_dir, cache_dir):
        kw = {"local_dir": local_dir} if local_dir else {"cache_dir": cache_dir}
        return snapshot_download(repo_id=repo, revision=revision, token=token, **kw)

    stage_small(a.hf_home, HfApi(token=token), download, snapshot)
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    l = sub.add_parser("list")
    l.add_argument("--out", required=True)
    l.add_argument("--fineweb-prefix", default="data/")
    l.add_argument("--margin", type=float, default=0.10)
    import build_run2_corpus as drv
    drv.add_plan_args(l)
    s = sub.add_parser("push")
    s.add_argument("--listing", required=True)
    s.add_argument("--remote", required=True)
    s.add_argument("--window-gb", type=float, default=120.0)
    s.add_argument("--transfer-limit-gb", type=float, default=TRANSFER_LIMIT_BYTES / 1e9)
    s.add_argument("--scratch", default=None)
    s.add_argument("--state", default=None)
    s.add_argument("--poll-seconds", type=float, default=60.0)
    t = sub.add_parser("stage-small")
    t.add_argument("--hf-home", required=True)
    a = p.parse_args(argv)
    return {"list": cmd_list, "push": cmd_push, "stage-small": cmd_stage_small}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
