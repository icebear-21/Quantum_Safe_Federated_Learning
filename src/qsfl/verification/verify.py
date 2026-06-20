"""Ownership acceptance:

    accept  iff  NC(P,P') > τ1  AND  NC(S,S') > τ2  AND  BER(S,S') < τ3  AND  HashMatch
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from qsfl.utils.metrics import bit_error_rate, normalized_correlation


@dataclass
class VerificationResult:
    nc_primary: float
    nc_secondary: float
    ber_secondary: float
    hash_match: bool
    tau1: float
    tau2: float
    tau3: float
    accept: bool
    ledger: dict

    def to_dict(self) -> dict:
        return asdict(self)


def verify_ownership(
    bundle,
    extracted: dict,
    model_hash: str | None,
    ledger,
    cfg,
) -> VerificationResult:
    """Compute the four checks and the AND decision.

    Args:
        bundle: the secret :class:`WatermarkBundle` (originals P, S + seeds).
        extracted: ``{"primary_extracted", "secondary_extracted"}`` from the model.
        model_hash: ``H = SHA256(C)`` to look up on the ledger (None -> skip).
        ledger: a :class:`qsfl.ledger.Ledger`.
        cfg: full config (reads ``verification.tau1/tau2/tau3``).
    """
    v = cfg.verification
    nc_p = normalized_correlation(
        bundle.primary_bits, np.asarray(extracted["primary_extracted"]), binary=True
    )
    nc_s = normalized_correlation(
        bundle.secondary_bits, np.asarray(extracted["secondary_extracted"]), binary=True
    )
    ber_s = bit_error_rate(bundle.secondary_bits, np.asarray(extracted["secondary_extracted"]))

    if model_hash is not None:
        ledger_res = ledger.verify(model_hash)
        hash_match = bool(ledger_res.get("found")) and bool(ledger_res.get("integrity_ok"))
    else:
        ledger_res = {"found": None, "integrity_ok": None}
        hash_match = False

    accept = (
        nc_p > float(v.tau1)
        and nc_s > float(v.tau2)
        and ber_s < float(v.tau3)
        and hash_match
    )
    return VerificationResult(
        nc_primary=nc_p,
        nc_secondary=nc_s,
        ber_secondary=ber_s,
        hash_match=hash_match,
        tau1=float(v.tau1),
        tau2=float(v.tau2),
        tau3=float(v.tau3),
        accept=bool(accept),
        ledger=ledger_res,
    )
