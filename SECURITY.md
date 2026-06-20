# Security Model — Real vs. Simulated

This is a **research prototype**. Honesty about which guarantees are
cryptographically real and which are research simulations is a first-class
requirement. This document maps each security objective to the component that
provides it, and flags its status.

Legend: **🟢 Real** (uses vetted crypto / sound construction) ·
**🟡 Partial** (real mechanism, prototype-grade assumptions) ·
**🔵 Simulated** (stands in for a real-world system; clearly labeled).

---

## Objective → component map

| Objective | Component | Status | Notes |
|---|---|---|---|
| **Privacy preservation** (no raw data sharing) | `federated/` — only `ΔW_i` leave a client | 🟢 / 🟡 | No raw images are shared. Update-level leakage is itself attacked in Layer 5 (gradient leakage, membership inference). `aggregation.mode=masked` strengthens this; see below. |
| **Confidentiality** (encryption) | `crypto/` — Kyber KEM + HKDF + AES-256-GCM | 🟢 | Real AEAD over real PQC KEM. With `kem.backend=liboqs` the KEM is the vetted Open Quantum Safe implementation; `kyber-py` is a pure-Python ML-KEM (not constant-time — fine for a prototype, not for production). |
| **Integrity / tamper detection** | AES-256-GCM auth tag; ledger hash chain | 🟢 | GCM detects any ciphertext/AAD tampering on decrypt. Ledger detects tampering of registered records. |
| **Authenticity** (ownership verification) | `watermark/` + `verification/` | 🟡 | White-box (Uchida-style) dual watermark; ownership accepted iff `NC(P,P')>τ1 ∧ NC(S,S')>τ2 ∧ BER(S,S')<τ3 ∧ HashMatch`. Robustness is empirically evaluated, not proven. |
| **Non-repudiation** | `ledger/` — hash-chained ledger (default) or web3 (optional) | 🔵 / 🟡 | See "Ledger" below. The default is tamper-**evident**, not a decentralized trust anchor. |
| **Quantum safety** | `quantum/` VQC + `crypto/` PQC | 🟢 / 🔵 | PQC (Kyber/ML-KEM) is real and quantum-resistant. The VQC "quantum key" is **simulated** (statevector), not QPU/QKD entropy. |

---

## Component disclosures

### Hybrid key derivation — 🟢 Real

Per update we compute:

```
(kyber_ct, ss) = KEM.encapsulate(pk)                 # post-quantum shared secret
data_key       = HKDF-SHA256(IKM = ss ‖ K_q,         # both sources feed the KDF
                             salt = random_per_update,
                             info = "qsfl/data-key/v1",
                             length = 32)
C_i            = AES-256-GCM(data_key, nonce, ΔW_i, aad)
```

This is a standard **hybrid PQC combiner**: the data key is secure if *either*
the Kyber secret **or** the VQC key `K_q` remains secret. We persist
`{kyber_ct, gcm_nonce, salt, aad, ciphertext}` per update.

- **AES-GCM nonce safety:** a fresh per-update random `salt` means a fresh
  `data_key` per update, so `(data_key, nonce)` reuse — the classic GCM
  catastrophe — does not occur. This is enforced by an explicit invariant and a
  unit test (`tests/test_crypto.py`), not left to chance.

### VQC quantum key (`K_q`) — 🔵 Simulated entropy

`K_q` is derived from measuring a simulated variational quantum circuit
(PennyLane statevector). It is **deterministic given `(theta_seed, seed)`** so
experiments reproduce. This **emulates** QKD-style key material; we do **not**
claim true quantum randomness from a classical simulator. In a real deployment
`K_q` would come from a QKD link or QRNG.

### Post-quantum KEM backend — 🟢 Real (🟡 for the fallback)

- `liboqs` (Open Quantum Safe) — vetted C implementation; used for reported
  results. **This is the backend behind any published numbers.**
- `kyber-py` — pure-Python ML-KEM; no native build, used for development and the
  smoke test. Not constant-time; do not use for production secrets.

A gated round-trip test (`tests/test_kem_liboqs.py`, marker `liboqs`) keeps the
primary backend from silently rotting while everyone develops on the fallback.

### Secure aggregation — 🔵 default simulation / 🟡 optional real masking

- `aggregation.mode=authorized_decrypt` (**default**): an authorized aggregator
  decrypts each `C_i` and runs weighted FedAvg. This is a faithful
  **simulation of encrypted aggregation** — it demonstrates the data flow but the
  aggregator sees individual updates. Labeled as such everywhere.
- `aggregation.mode=masked`: Bonawitz-style pairwise additive masking, where
  masks cancel in the sum so the server never sees an individual `ΔW_i`. A unit
  test asserts the masks truly cancel. Dropout handling is documented in
  `federated/secure_agg.py`; the prototype recovers the sum when the set of
  surviving clients is known.

### Ledger / non-repudiation — 🔵 default / 🟡 optional

- `ledger.backend=hashchain` (**default**): an append-only, SHA256-linked ledger
  in pure Python. It is **tamper-evident** (any edit breaks the chain, verified by
  a test) and produces the reported ownership-proof results. It is **not** a
  decentralized trust anchor: a single party controls the file.
- `ledger.backend=web3` (optional): a minimal Solidity contract on a local dev
  chain (Anvil/Ganache) stores `H = SHA256(C)` and emits an event. A **single-node
  local chain is still not decentralized** — true distributed immutability /
  non-repudiation requires a public chain with independent validators.

---

## What an attacker is assumed to / not to control

- Assumed protected: the embedding keys (`watermark.*.key_seed`), the KEM secret
  key, and `K_q`. These live outside the model artifact and must be kept secret.
- Evaluated (Layer 5): an adversary who can fine-tune, prune, overwrite
  watermarks, query/extract the model, run membership-inference and
  gradient-leakage, or poison training data.

## Reporting

This is academic/prototype code. Do not deploy it to protect real patient data
without a security review and replacing every 🔵/🟡 component with a
production-grade equivalent.
