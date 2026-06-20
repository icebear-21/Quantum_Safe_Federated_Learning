"""Hermetic end-to-end pipeline test (synthetic data, no network, CPU)."""

import pytest

pytest.importorskip("torch")
pytest.importorskip("pennylane")
pytest.importorskip("kyber_py")
pytest.importorskip("PIL")
pytest.importorskip("qrcode")

from qsfl.crypto import reset_nonce_guard  # noqa: E402
from qsfl.pipeline import run_pipeline  # noqa: E402
from qsfl.utils.config import find_repo_root, load_config  # noqa: E402


def _cfg(tmp_path, **extra):
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
        "run_name=pytest",
        f"output_dir={tmp_path.as_posix()}/out",
        f"ledger.path={tmp_path.as_posix()}/chain.jsonl",
    ]
    overrides += [f"{k}={v}" for k, v in extra.items()]
    return load_config(find_repo_root() / "configs" / "default.yaml", overrides)


def test_end_to_end_authorized_decrypt(tmp_path):
    reset_nonce_guard()
    res = run_pipeline(_cfg(tmp_path))
    v = res["verification"]
    assert v["accept"] is True
    assert v["ber_secondary"] == 0.0
    assert v["nc_primary"] == pytest.approx(1.0)
    assert v["hash_match"] is True
    # encrypt -> decrypt round-trips the model exactly
    assert res["accuracy_deployed"] == pytest.approx(res["accuracy_watermarked"], abs=1e-6)


def test_end_to_end_masked_mode(tmp_path):
    reset_nonce_guard()
    res = run_pipeline(_cfg(tmp_path, **{"aggregation.mode": "masked"}))
    assert res["verification"]["accept"] is True
