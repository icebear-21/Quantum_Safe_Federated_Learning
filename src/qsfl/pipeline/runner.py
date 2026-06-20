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
from pathlib import Path

import numpy as np

from qsfl.crypto.encryptor import HybridEncryptor
from qsfl.data import build_federated_data
from qsfl.federated import FederatedServer, evaluate, numpy_to_state, state_to_numpy
from qsfl.ledger import OwnershipRecord, get_ledger
from qsfl.models import build_model
from qsfl.quantum import generate_quantum_key
from qsfl.utils.config import save_config
from qsfl.utils.logging import get_logger
from qsfl.utils.tensor_io import pack_state, unpack_state
from qsfl.verification import verify_ownership
from qsfl.watermark import DualWatermarker

logger = get_logger("pipeline")


def _watermark_commitment(bundle) -> str:
    return hashlib.sha256(
        np.asarray(bundle.primary_bits).tobytes()
        + np.asarray(bundle.secondary_bits).tobytes()
        + str(bundle.primary_key_seed).encode()
        + str(bundle.secondary_key_seed).encode()
    ).hexdigest()


def run_pipeline(cfg) -> dict:
    """Execute the pipeline and return (also save) a results dict."""
    from qsfl.utils.seeding import set_seed

    set_seed(int(cfg.seed), deterministic=bool(cfg.deterministic))
    out_dir = Path(cfg.output_dir) / str(cfg.run_name)
    out_dir.mkdir(parents=True, exist_ok=True)
    save_config(cfg, out_dir / "config.yaml")
    t0 = time.time()
    results: dict = {"run_name": str(cfg.run_name), "mode": str(cfg.mode), "device": str(cfg.device)}

    # --- Layer 2: quantum key K_q ---
    if bool(cfg.quantum.enabled):
        kq = generate_quantum_key(cfg.quantum)
        logger.info("VQC key K_q generated (%d bits, backend=%s)", cfg.quantum.key_bits, cfg.quantum.backend)
    else:
        import os

        kq = os.urandom(cfg.quantum.key_bits // 8)
    results["quantum_key_bits"] = int(cfg.quantum.key_bits)

    # --- Layer 1: data + clients ---
    data = build_federated_data(cfg)
    results["num_clients"] = data.num_clients
    results["num_classes"] = data.num_classes
    results["client_names"] = data.client_names

    # --- Layer 2/3: hybrid encryptor + federated secure aggregation ---
    encryptor = HybridEncryptor(
        quantum_key=kq,
        backend=str(cfg.aggregation.kem.backend),
        algorithm=str(cfg.aggregation.kem.algorithm),
    )
    server = FederatedServer(cfg, data, encryptor)
    fed = server.run()
    model = fed["model"]
    results["training_history"] = fed["history"]
    acc_clean = evaluate(model, data.test_set, cfg.device)
    results["accuracy_global"] = acc_clean
    logger.info("global model accuracy (pre-watermark) = %.4f", acc_clean)

    # --- Layer 4: dual watermarking ---
    bundle = None
    if bool(cfg.watermark.enabled):
        bundle = DualWatermarker(cfg).embed(model)
        acc_wm = evaluate(model, data.test_set, cfg.device)
        results["accuracy_watermarked"] = acc_wm
        results["accuracy_drop_watermark"] = acc_clean - acc_wm
        logger.info("accuracy after watermark = %.4f (drop %.4f)", acc_wm, acc_clean - acc_wm)

    # --- encrypt final protected model: C, H = SHA256(C) ---
    w_s = state_to_numpy(model)
    payload = pack_state(w_s)
    enc_model = encryptor.encrypt(payload, aad=b"final-model", label="W_S")
    model_hash = enc_model.ciphertext_hash()
    results["model_hash_H"] = model_hash
    (out_dir / "encrypted_model.json").write_text(json.dumps(enc_model.to_dict()))

    # --- Layer 6: ownership registration ---
    ledger = get_ledger(cfg)
    commitment = _watermark_commitment(bundle) if bundle else ""
    proof = ledger.register(
        OwnershipRecord(model_hash=model_hash, owner=str(cfg.run_name), watermark_commitment=commitment)
    )
    results["ownership_proof"] = proof
    logger.info("registered H on ledger (%s): %s", cfg.ledger.backend, proof)

    # --- Deployment: authorized user retrieves C, decrypts, verifies, deploys ---
    recovered = unpack_state(encryptor.decrypt(enc_model))
    deployed = build_model(cfg, data.num_classes, data.in_channels, data.image_size)
    numpy_to_state(deployed, recovered)
    acc_deployed = evaluate(deployed, data.test_set, cfg.device)
    results["accuracy_deployed"] = acc_deployed

    if bundle is not None:
        extracted = DualWatermarker(cfg).extract(deployed, bundle)
        verdict = verify_ownership(bundle, extracted, model_hash, ledger, cfg)
        results["verification"] = verdict.to_dict()
        logger.info(
            "verification | NC(P)=%.3f NC(S)=%.3f BER(S)=%.3f hash=%s -> accept=%s",
            verdict.nc_primary, verdict.nc_secondary, verdict.ber_secondary,
            verdict.hash_match, verdict.accept,
        )
        results["watermark"] = {
            "primary_layer": bundle.primary_layer,
            "secondary_layer": bundle.secondary_layer,
            "primary_bits": int(np.asarray(bundle.primary_bits).size),
            "secondary_bits": int(np.asarray(bundle.secondary_bits).size),
            "commitment": commitment,
        }

    results["elapsed_sec"] = round(time.time() - t0, 2)
    (out_dir / "results.json").write_text(json.dumps(results, indent=2, default=str))
    logger.info("pipeline complete in %.1fs -> %s", results["elapsed_sec"], out_dir / "results.json")
    return results
