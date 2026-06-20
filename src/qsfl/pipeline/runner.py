"""Run the full quantum-safe FL + protection pipeline.

    K_q = VQC(θ)                      quantum key
    C_i = Enc(ΔW_i ; hybrid key)      per-client encryption (inside the server)
    W_g = SecureAgg(C_1..C_N)         federated aggregation
    W_P = WM_1(W_g) ; W_S = WM_2(W_P) dual watermark
    C   = Enc(W_S) ; H = SHA256(C)    encrypt final model, register on ledger
    deploy: retrieve C -> decrypt -> extract P',S' -> verify NC/BER/Hash -> infer
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from qsfl.crypto.encryptor import HybridEncryptor
from qsfl.data import build_federated_data
from qsfl.federated import FederatedServer, evaluate, numpy_to_state, state_to_numpy
from qsfl.ledger import OwnershipRecord, get_ledger
from qsfl.models import build_model
from qsfl.quantum import generate_quantum_key
from qsfl.utils.config import save_config
from qsfl.utils.logging import get_logger
from qsfl.utils.seeding import set_seed
from qsfl.utils.tensor_io import pack_state, unpack_state
from qsfl.verification import verify_ownership
from qsfl.watermark import DualWatermarker

logger = get_logger("pipeline")


@dataclass
class ProtectedArtifacts:
    """In-memory objects produced by training + protection (for pipeline/attacks)."""

    model: Any
    bundle: Any
    data: Any
    encryptor: HybridEncryptor
    quantum_key: bytes
    history: list
    accuracy_clean: float
    accuracy_watermarked: float


def _watermark_commitment(bundle) -> str:
    return hashlib.sha256(
        np.asarray(bundle.primary_bits).tobytes()
        + np.asarray(bundle.secondary_bits).tobytes()
        + str(bundle.primary_key_seed).encode()
        + str(bundle.secondary_key_seed).encode()
    ).hexdigest()


def train_and_protect(cfg) -> ProtectedArtifacts:
    """Layers 1-4: quantum key -> federated secure-agg -> dual watermark.

    Returns the live objects (model, watermark bundle, encryptor, data) so both
    the full pipeline and the attack-evaluation harness can build on them.
    """
    set_seed(int(cfg.seed), deterministic=bool(cfg.deterministic))

    if bool(cfg.quantum.enabled):
        kq = generate_quantum_key(cfg.quantum)
        logger.info("VQC key K_q generated (%d bits, backend=%s)", cfg.quantum.key_bits, cfg.quantum.backend)
    else:
        import os

        kq = os.urandom(int(cfg.quantum.key_bits) // 8)

    data = build_federated_data(cfg)
    encryptor = HybridEncryptor(
        quantum_key=kq,
        backend=str(cfg.aggregation.kem.backend),
        algorithm=str(cfg.aggregation.kem.algorithm),
    )
    server = FederatedServer(cfg, data, encryptor)
    fed = server.run()
    model = fed["model"]
    acc_clean = evaluate(model, data.test_set, cfg.device)
    logger.info("global model accuracy (pre-watermark) = %.4f", acc_clean)

    bundle = None
    acc_wm = acc_clean
    if bool(cfg.watermark.enabled):
        bundle = DualWatermarker(cfg).embed(model)
        acc_wm = evaluate(model, data.test_set, cfg.device)
        logger.info("accuracy after watermark = %.4f (drop %.4f)", acc_wm, acc_clean - acc_wm)

    return ProtectedArtifacts(
        model=model,
        bundle=bundle,
        data=data,
        encryptor=encryptor,
        quantum_key=kq,
        history=fed["history"],
        accuracy_clean=acc_clean,
        accuracy_watermarked=acc_wm,
    )


def run_pipeline(cfg) -> dict:
    """Execute the full pipeline (incl. encrypt/register/verify/deploy) + save results."""
    out_dir = Path(cfg.output_dir) / str(cfg.run_name)
    out_dir.mkdir(parents=True, exist_ok=True)
    save_config(cfg, out_dir / "config.yaml")
    t0 = time.time()

    art = train_and_protect(cfg)
    results: dict = {
        "run_name": str(cfg.run_name),
        "mode": str(cfg.mode),
        "device": str(cfg.device),
        "quantum_key_bits": int(cfg.quantum.key_bits),
        "num_clients": art.data.num_clients,
        "num_classes": art.data.num_classes,
        "client_names": art.data.client_names,
        "training_history": art.history,
        "accuracy_global": art.accuracy_clean,
    }
    if art.bundle is not None:
        results["accuracy_watermarked"] = art.accuracy_watermarked
        results["accuracy_drop_watermark"] = art.accuracy_clean - art.accuracy_watermarked

    # --- encrypt final protected model: C, H = SHA256(C) ---
    w_s = state_to_numpy(art.model)
    enc_model = art.encryptor.encrypt(pack_state(w_s), aad=b"final-model", label="W_S")
    model_hash = enc_model.ciphertext_hash()
    results["model_hash_H"] = model_hash
    (out_dir / "encrypted_model.json").write_text(json.dumps(enc_model.to_dict()))

    # --- Layer 6: ownership registration ---
    ledger = get_ledger(cfg)
    commitment = _watermark_commitment(art.bundle) if art.bundle else ""
    proof = ledger.register(
        OwnershipRecord(model_hash=model_hash, owner=str(cfg.run_name), watermark_commitment=commitment)
    )
    results["ownership_proof"] = proof
    logger.info("registered H on ledger (%s): %s", cfg.ledger.backend, proof)

    # --- Deployment: authorized user retrieves C, decrypts, verifies, deploys ---
    recovered = unpack_state(art.encryptor.decrypt(enc_model))
    deployed = build_model(cfg, art.data.num_classes, art.data.in_channels, art.data.image_size)
    numpy_to_state(deployed, recovered)
    results["accuracy_deployed"] = evaluate(deployed, art.data.test_set, cfg.device)

    if art.bundle is not None:
        extracted = DualWatermarker(cfg).extract(deployed, art.bundle)
        verdict = verify_ownership(art.bundle, extracted, model_hash, ledger, cfg)
        results["verification"] = verdict.to_dict()
        logger.info(
            "verification | NC(P)=%.3f NC(S)=%.3f BER(S)=%.3f hash=%s -> accept=%s",
            verdict.nc_primary, verdict.nc_secondary, verdict.ber_secondary,
            verdict.hash_match, verdict.accept,
        )
        results["watermark"] = {
            "primary_layer": art.bundle.primary_layer,
            "secondary_layer": art.bundle.secondary_layer,
            "primary_bits": int(np.asarray(art.bundle.primary_bits).size),
            "secondary_bits": int(np.asarray(art.bundle.secondary_bits).size),
            "commitment": commitment,
        }

    results["elapsed_sec"] = round(time.time() - t0, 2)
    (out_dir / "results.json").write_text(json.dumps(results, indent=2, default=str))
    logger.info("pipeline complete in %.1fs -> %s", results["elapsed_sec"], out_dir / "results.json")
    return results
