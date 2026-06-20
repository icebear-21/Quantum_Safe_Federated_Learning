"""Federated server: runs rounds, encrypts client updates, securely aggregates.

Per round, for each participating client:
    1. load the current global weights, train locally -> ΔW_i
    2. (masked mode) form  w_i·ΔW_i + mask_i ; (authorized mode) keep ΔW_i
    3. serialize -> encrypt with the hybrid (Kyber ‖ K_q) key  -> C_i
The aggregator then decrypts the C_i and combines them (weighted FedAvg). Both
modes yield the same global update; only the server's visibility differs.
"""

from __future__ import annotations

import numpy as np

from qsfl.crypto.encryptor import HybridEncryptor
from qsfl.federated.secure_agg import (
    add_states,
    generate_pairwise_mask,
    scale_state,
    sum_states,
    weighted_average,
)
from qsfl.federated.trainer import LocalTrainer, evaluate, numpy_to_state, state_to_numpy
from qsfl.models import build_model
from qsfl.utils.logging import get_logger
from qsfl.utils.tensor_io import pack_state, unpack_state

logger = get_logger("federated")


class FederatedServer:
    def __init__(self, cfg, fed_data, encryptor: HybridEncryptor) -> None:
        self.cfg = cfg
        self.data = fed_data
        self.encryptor = encryptor
        self.device = cfg.device
        self.trainer = LocalTrainer(cfg, self.device)
        self.model = build_model(
            cfg, fed_data.num_classes, fed_data.in_channels, fed_data.image_size
        )
        self.global_state = state_to_numpy(self.model)
        self.weights = [float(len(ds)) for ds in fed_data.client_train]

    def _select(self, rnd: int) -> list[int]:
        frac = float(self.cfg.federated.client_fraction)
        n = self.data.num_clients
        k = max(1, int(round(frac * n)))
        if k >= n:
            return list(range(n))
        rng = np.random.default_rng(int(self.cfg.seed) + rnd)
        return sorted(rng.choice(n, size=k, replace=False).tolist())

    def _client_payload(self, client_id: int, participants: list[int], rnd: int):
        numpy_to_state(self.model, self.global_state)
        local_state = self.trainer.train(self.model, self.data.client_train[client_id])
        delta = {k: local_state[k] - self.global_state[k] for k in local_state}
        w = self.weights[client_id]

        if str(self.cfg.aggregation.mode) == "masked":
            seed = int(self.cfg.aggregation.masked.mask_seed) + rnd
            mask = generate_pairwise_mask(client_id, participants, delta, seed)
            payload_state = add_states(scale_state(delta, w), mask)
        else:
            payload_state = delta

        blob = pack_state(payload_state)
        ct = self.encryptor.encrypt(blob, aad=f"client-{client_id}".encode(), label=f"dW_{client_id}")
        return ct, w

    def _aggregate(self, payloads, weights) -> dict[str, np.ndarray]:
        decrypted = [unpack_state(self.encryptor.decrypt(c)) for c in payloads]
        if str(self.cfg.aggregation.mode) == "masked":
            summed = sum_states(decrypted)  # w_i·ΔW_i summed; masks cancel
            return scale_state(summed, 1.0 / float(sum(weights)))
        return weighted_average(decrypted, weights)

    def run(self) -> dict:
        """Train for ``federated.rounds`` rounds; return final state + metrics."""
        rounds = int(self.cfg.federated.rounds)
        history = []
        last_payloads = []
        for rnd in range(rounds):
            participants = self._select(rnd)
            payloads, weights = [], []
            for cid in participants:
                ct, w = self._client_payload(cid, participants, rnd)
                payloads.append(ct)
                weights.append(w)
            agg_update = self._aggregate(payloads, weights)
            self.global_state = add_states(self.global_state, agg_update)
            numpy_to_state(self.model, self.global_state)
            acc = evaluate(self.model, self.data.test_set, self.device)
            history.append({"round": rnd, "participants": participants, "test_acc": acc})
            logger.info(
                "round %d/%d | clients=%d | mode=%s | test_acc=%.4f",
                rnd + 1,
                rounds,
                len(participants),
                self.cfg.aggregation.mode,
                acc,
            )
            last_payloads = payloads
        return {
            "global_state": self.global_state,
            "model": self.model,
            "history": history,
            "client_ciphertexts": last_payloads,
        }
