import numpy as np
import pytest

from qsfl.utils.metrics import bit_error_rate, normalized_correlation
from qsfl.watermark.uchida import embed_bits, extract_bits


def test_clean_roundtrip_is_perfect():
    rng = np.random.default_rng(0)
    w = rng.standard_normal(5000).astype(np.float32)
    bits = rng.integers(0, 2, 256)
    w2 = embed_bits(w, bits, key_seed=11, margin=0.05)
    out = extract_bits(w2, 256, key_seed=11)
    assert bit_error_rate(bits, out) == 0.0
    assert normalized_correlation(bits, out, binary=True) == pytest.approx(1.0)


def test_wrong_key_extracts_noise():
    rng = np.random.default_rng(1)
    w = rng.standard_normal(5000).astype(np.float32)
    bits = rng.integers(0, 2, 128)
    w2 = embed_bits(w, bits, 7, 0.05)
    out = extract_bits(w2, 128, key_seed=999)  # attacker without the secret key
    assert bit_error_rate(bits, out) > 0.25


def test_embedding_perturbation_is_small():
    rng = np.random.default_rng(2)
    w = rng.standard_normal(20000).astype(np.float32)
    bits = rng.integers(0, 2, 256)
    w2 = embed_bits(w, bits, 3, margin=0.02)
    rel = np.linalg.norm(w2 - w) / np.linalg.norm(w)
    assert rel < 0.2  # minimal-nudge keeps the change modest


def test_dual_watermarker_on_model():
    pytest.importorskip("torch")
    pytest.importorskip("PIL")
    pytest.importorskip("qrcode")
    from qsfl.models.backbones import TinyCNN
    from qsfl.utils.config import find_repo_root, load_config
    from qsfl.watermark import DualWatermarker

    cfg = load_config(find_repo_root() / "configs" / "default.yaml", ["watermark.secondary.length=64"])
    model = TinyCNN(in_channels=3, num_classes=10)
    wm = DualWatermarker(cfg)
    bundle = wm.embed(model)
    ext = wm.extract(model, bundle)
    assert bit_error_rate(bundle.secondary_bits, ext["secondary_extracted"]) == 0.0
    assert normalized_correlation(bundle.primary_bits, ext["primary_extracted"], binary=True) == pytest.approx(1.0)
