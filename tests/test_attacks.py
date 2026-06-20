"""Attack-suite smoke test (synthetic, CPU). Exercises the light attacks."""

import pytest

pytest.importorskip("torch")
pytest.importorskip("pennylane")
pytest.importorskip("kyber_py")
pytest.importorskip("PIL")
pytest.importorskip("qrcode")
pytest.importorskip("sklearn")
pytest.importorskip("matplotlib")

from qsfl.attacks import run_attacks  # noqa: E402
from qsfl.crypto import reset_nonce_guard  # noqa: E402
from qsfl.utils.config import find_repo_root, load_config  # noqa: E402


def test_attack_suite_runs(tmp_path):
    reset_nonce_guard()
    overrides = [
        "data.name=synthetic",
        "data.image_size=16",
        "data.as_rgb=true",
        "federated.num_clients=2",
        "federated.rounds=1",
        "federated.local_epochs=1",
        "federated.batch_size=32",
        "model.backbone=resnet",
        "model.resnet.variant=tiny_cnn",
        "quantum.n_qubits=3",
        "quantum.n_layers=1",
        "watermark.secondary.length=64",
        "device=cpu",
        "num_workers=0",
        "run_name=pytest_attacks",
        "attacks.enabled=[fine_tuning,pruning,watermark_removal,membership_inference,poisoning]",
        "attacks.fine_tuning.epochs=1",
        "attacks.pruning.amounts=[0.2,0.5]",
        f"output_dir={tmp_path.as_posix()}/out",
        f"ledger.path={tmp_path.as_posix()}/chain.jsonl",
    ]
    cfg = load_config(find_repo_root() / "configs" / "default.yaml", overrides)
    res = run_attacks(cfg)
    a = res["attacks"]

    assert "error" not in a["fine_tuning"]
    assert "accuracy_after" in a["fine_tuning"]
    assert "ber_secondary" in a["fine_tuning"]
    assert "by_amount" in a["pruning"] and "0.2" in a["pruning"]["by_amount"]
    assert 0.0 <= a["membership_inference"]["auc"] <= 1.0
    assert "accuracy_after" in a["watermark_removal"]
    assert "accuracy_drop" in a["poisoning"]
