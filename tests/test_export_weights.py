"""scripts/export_weights.py: the export is checked before it takes its final name, and the .sha256 is written last.
CPU only: the strict load check is replaced by a fake."""
import hashlib
import os
import sys
import tempfile

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

import export_weights as ew  # noqa: E402


def payload():
    return {"format": ew.FORMAT, "model_state_dict": {"w": torch.ones(4, dtype=torch.bfloat16)}, "model_params": {},
            "token_count": 123, "phase": "evidence", "source": "x.pt"}


def test_failed_check_leaves_no_final_files():
    with tempfile.TemporaryDirectory() as tmp:
        out = os.path.join(tmp, "run2_1B.pt")
        seen = []

        def bad(path):
            seen.append(path)
            assert path != out and os.path.exists(path), "the check runs on the temp file"
            raise RuntimeError("strict load failed")
        try:
            ew.finalize(payload(), out, verify_fn=bad)
            raise AssertionError("expected the check to fail")
        except RuntimeError:
            pass
        assert seen and os.listdir(tmp) == [], os.listdir(tmp)
        # a stale pair of an earlier export is removed before the new file is written, so a failure never leaves a
        # sidecar that describes other bytes
        with open(out, "wb") as f:
            f.write(b"old")
        with open(out + ".sha256", "w") as f:
            f.write("deadbeef  run2_1B.pt\n")
        try:
            ew.finalize(payload(), out, verify_fn=bad)
        except RuntimeError:
            pass
        assert not os.path.exists(out + ".sha256") and not os.path.exists(out + ".tmp")


def test_success_writes_pt_then_sha256():
    with tempfile.TemporaryDirectory() as tmp:
        out = os.path.join(tmp, "run2_1B.pt")
        digest = ew.finalize(payload(), out, verify_fn=lambda p: None)
        with open(out, "rb") as f:
            assert hashlib.sha256(f.read()).hexdigest() == digest
        assert open(out + ".sha256").read() == "%s  run2_1B.pt\n" % digest
        assert sorted(os.listdir(tmp)) == ["run2_1B.pt", "run2_1B.pt.sha256"]
        loaded = torch.load(out, map_location="cpu", weights_only=False)
        assert loaded["token_count"] == 123


def test_no_cuda_refuses_unless_no_verify():
    real = torch.cuda.is_available
    torch.cuda.is_available = lambda: False
    try:
        with tempfile.TemporaryDirectory() as tmp:
            out = os.path.join(tmp, "run2_1B.pt")
            try:
                ew.finalize(payload(), out, verify_fn=ew.verify)
                raise AssertionError("expected a refusal without CUDA")
            except SystemExit as e:
                assert "CUDA" in str(e)
            assert os.listdir(tmp) == []
            ew.finalize(payload(), out, verify_fn=None)
            assert sorted(os.listdir(tmp)) == ["run2_1B.pt", "run2_1B.pt.sha256"]
    finally:
        torch.cuda.is_available = real


if __name__ == "__main__":
    test_no_cuda_refuses_unless_no_verify()
    test_failed_check_leaves_no_final_files()
    test_success_writes_pt_then_sha256()
    print("test_export_weights: ok")
