"""HFSync retry/backoff, non-fatal failure, upload marking, drop policy and drain. Uses a stub
api object -- no network, no GPU.
"""
import os, sys, tempfile, threading
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.runtime.hf_sync import HFSync
from modules.runtime.status import eta_seconds, format_duration, write_status


class StubApi:
    """Records calls; fails the first `fail_times` attempts for any path in `flaky`."""

    def __init__(self, flaky=(), fail_times=0, always_fail=()):
        self.calls = []
        self.deleted = []
        self.squash_calls = 0
        self.flaky = set(flaky)
        self.always_fail = set(always_fail)
        self.fail_times = fail_times
        self.attempts = {}
        self.lock = threading.Lock()

    def upload_file(self, path_or_fileobj=None, path_in_repo=None, repo_id=None, token=None, **kw):
        with self.lock:
            self.attempts[path_in_repo] = self.attempts.get(path_in_repo, 0) + 1
            n = self.attempts[path_in_repo]
            self.calls.append(path_in_repo)
        if path_in_repo in self.always_fail:
            raise RuntimeError("permanent failure")
        if path_in_repo in self.flaky and n <= self.fail_times:
            raise RuntimeError("transient failure")
        return "ok"

    def delete_file(self, path_in_repo=None, repo_id=None, token=None, **kw):
        with self.lock:
            self.attempts[path_in_repo] = self.attempts.get(path_in_repo, 0) + 1
            n = self.attempts[path_in_repo]
        if path_in_repo in self.always_fail:
            raise RuntimeError("permanent failure")
        if path_in_repo in self.flaky and n <= self.fail_times:
            raise RuntimeError("transient failure")
        with self.lock:
            self.deleted.append(path_in_repo)
        return "ok"

    def super_squash_history(self, repo_id=None, token=None, **kw):
        with self.lock:
            self.squash_calls += 1
        return "ok"


d = tempfile.mkdtemp()

def mkfile(name, body="x"):
    p = os.path.join(d, name)
    with open(p, "w") as f:
        f.write(body)
    return p

# --- happy path -----------------------------------------------------------
api = StubApi()
sync = HFSync("owner/repo", token="t", api=api)
p = mkfile("a.pt")
sync.upload(p, "checkpoints/a.pt")
assert sync.drain(timeout=10), "drain must complete"
assert api.calls == ["checkpoints/a.pt"], api.calls
assert sync.is_uploaded(p)
sync.close()
print("[ok] a queued file uploads once and is marked uploaded")

# --- retry then succeed ---------------------------------------------------
api = StubApi(flaky={"checkpoints/b.pt"}, fail_times=2)
sync = HFSync("owner/repo", token="t", api=api, backoff=0.01)
p = mkfile("b.pt")
sync.upload(p, "checkpoints/b.pt")
assert sync.drain(timeout=10)
assert api.attempts["checkpoints/b.pt"] == 3, api.attempts
assert sync.is_uploaded(p)
sync.close()
print("[ok] transient failures are retried and eventually succeed")

# --- permanent failure is non fatal and leaves the file unmarked ----------
api = StubApi(always_fail={"checkpoints/c.pt"})
sync = HFSync("owner/repo", token="t", api=api, backoff=0.01)
p = mkfile("c.pt")
sync.upload(p, "checkpoints/c.pt")   # must not raise
assert sync.drain(timeout=10)
assert not sync.is_uploaded(p), "a failed upload must NOT be marked uploaded -- retention reads this"
sync.close()
print("[ok] a permanently failing upload never raises into the caller and stays unmarked")

# --- delete queues a remote removal and triggers a (throttled) history squash ---------------
api = StubApi()
sync = HFSync("owner/repo", token="t", api=api, squash_min_interval=1800.0)
sync.delete("checkpoints/old_a.pt")
assert sync.drain(timeout=10)
assert api.deleted == ["checkpoints/old_a.pt"], api.deleted
assert api.squash_calls == 1, "the first delete after startup must trigger a squash"
sync.close()
print("[ok] delete() removes the remote file and squashes history")

# a burst of deletes inside the throttle window squashes only once
api = StubApi()
sync = HFSync("owner/repo", token="t", api=api, squash_min_interval=1800.0)
sync.delete("checkpoints/old_b.pt")
sync.delete("checkpoints/old_c.pt")
sync.delete("checkpoints/old_d.pt")
assert sync.drain(timeout=10)
assert set(api.deleted) == {"checkpoints/old_b.pt", "checkpoints/old_c.pt", "checkpoints/old_d.pt"}
assert api.squash_calls == 1, f"a burst of deletes must squash once, not {api.squash_calls} times"
sync.close()
print("[ok] a burst of deletes squashes history only once, throttled")

# a squash failure must not raise into the caller or block further deletes
class FailingSquashApi(StubApi):
    def super_squash_history(self, repo_id=None, token=None, **kw):
        raise RuntimeError("squash failed")

api = FailingSquashApi()
sync = HFSync("owner/repo", token="t", api=api)
sync.delete("checkpoints/old_e.pt")   # must not raise
assert sync.drain(timeout=10)
assert api.deleted == ["checkpoints/old_e.pt"], "the delete itself must still succeed"
sync.close()
print("[ok] a failing history squash is swallowed; the delete itself still lands")

# a permanently failing delete is swallowed like a permanently failing upload
api = StubApi(always_fail={"checkpoints/old_f.pt"})
sync = HFSync("owner/repo", token="t", api=api, backoff=0.01)
sync.delete("checkpoints/old_f.pt")   # must not raise
assert sync.drain(timeout=10)
assert api.deleted == [], "a permanently failing delete must not be recorded as deleted"
sync.close()
print("[ok] a permanently failing delete never raises into the caller")

# --- disabled sync is a no-op that reports nothing uploaded ---------------
sync = HFSync("", token=None, api=StubApi())
assert not sync.enabled
p = mkfile("d.pt")
sync.upload(p, "checkpoints/d.pt")
sync.delete("checkpoints/d.pt")
assert not sync.is_uploaded(p)
assert sync.drain(timeout=1)
sync.close()
print("[ok] an empty repo id disables uploads and deletes without breaking callers")

# --- status helpers -------------------------------------------------------
assert eta_seconds(0, 100, 0) is None
assert eta_seconds(50, 100, 10) == 5.0
assert eta_seconds(150, 100, 10) == 0.0
assert format_duration(0) == "0m"
assert format_duration(3661) == "1h 1m"
assert format_duration(90000) == "25h 0m"
sp = os.path.join(d, "status.json")
write_status(sp, phase="phase1", tokens=1234)
import json
with open(sp) as f:
    assert json.load(f)["tokens"] == 1234
assert not os.path.exists(sp + ".tmp")
print("[ok] status helpers compute ETA and write atomically")

print("\nHF SYNC CHECKS PASSED")
