"""[F] Cheap checks for the F3 launch plumbing (no model, no network)."""
import subprocess
import sys
from pathlib import Path

from desi_agent.train import sft

ROOT = Path(__file__).resolve().parents[1]


def test_rank_from_env(monkeypatch):
    for k in ("LOCAL_RANK", "WORLD_SIZE", "RANK"):
        monkeypatch.delenv(k, raising=False)
    assert sft._rank() == (0, 1, True)
    monkeypatch.setenv("LOCAL_RANK", "1"); monkeypatch.setenv("WORLD_SIZE", "2"); monkeypatch.setenv("RANK", "1")
    assert sft._rank() == (1, 2, False)


def test_resuming_trainer_passes_checkpoint():
    class T:
        def train(self, resume_from_checkpoint=None):
            return resume_from_checkpoint
    assert sft._ResumingTrainer(T(), "ckpt/x").train() == "ckpt/x"
    assert sft._ResumingTrainer(T(), None).train() is None


def test_never_train_on_test():
    import pytest
    with pytest.raises(ValueError):
        sft._load("test")
    with pytest.raises(ValueError):
        sft._load("hf:final/test.jsonl")


def test_kaggle_kernel_source_compiles():
    out = subprocess.run([sys.executable, str(ROOT / "scripts/train_kaggle.py"), "push", "--run", "t", "--user", "u",
                          "--dry-run", "--", "--epochs", "1"], capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    assert '"id": "u/desi-agent-t"' in out.stdout and "max_hours=11.0" in out.stdout
