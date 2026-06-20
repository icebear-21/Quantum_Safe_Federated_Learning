# Quantum-Safe Federated Learning Framework

**Secure Medical Image Sharing & Deep Learning Model Protection**

A research prototype that lets multiple hospitals train deep-learning models on
private medical images, then **encrypts model updates with post-quantum crypto
keyed by a quantum-circuit-generated secret**, **securely aggregates** them into a
global model, **dual-watermarks** the model for ownership, **registers** it on an
immutable ledger, **stress-tests** it against a suite of ML attacks, and finally
**verifies ownership and deploys** it.

The federated path (many hospitals + aggregation) is the superset; the
single-model path is the same protection pipeline with `N = 1` and no
aggregation.

> ⚠️ **Research prototype.** Some guarantees are cryptographically real and some
> are clearly-labeled simulations. Read [SECURITY.md](SECURITY.md) before drawing
> conclusions — it maps every security objective to the component that provides
> it and states exactly what is real vs. simulated.

---

## Pipeline

```
K_q  = VQC(θ)                       # quantum-generated 256-bit key (simulated circuit)
C_i  = Enc(ΔW_i ; hybrid key)       # per-client encryption, key = HKDF(kyber_ss ‖ K_q)
W_g  = SecureAgg(C_1, …, C_N)       # weighted FedAvg (authorized-decrypt or masked)
W_P  = WM_1(W_g)                    # primary  (visible)   watermark embed
W_S  = WM_2(W_P)                    # secondary (invisible) watermark embed
H    = SHA256(C)                    # ownership hash, recorded on the ledger
```

Ownership is accepted at deploy time iff:

```
NC(P, P') > τ1   AND   NC(S, S') > τ2   AND   BER(S, S') < τ3   AND   HashMatch
```

See the [architecture diagram description](#architecture) and
[notation](#notation) below.

---

## Target hardware

Developed to run reproducibly from a clean clone on:

| Resource | Target |
|----------|--------|
| CPU      | 6 cores (dataloader workers / client parallelism capped at 4 by default) |
| RAM      | 64 GB |
| GPU      | 1× NVIDIA H100, 80 GB (CUDA); a **CPU-only fallback** path exists for smoke tests |
| Disk     | 100 GB (MedMNIST is small; old checkpoints are pruned) |
| OS       | Linux x86_64, recent CUDA toolkit, Python 3.10–3.12 |

Quantum circuits are **simulated** (no QPU): a CPU statevector simulator by
default, with a `lightning.gpu` flag for GPU simulation.

---

## Install

There are two supported environments. **pip** is the default/development +
smoke-test path (pure-Python KEM, no native build). **conda** is the recommended
path for the **liboqs-backed "reported results"** environment.

### Reproducible installs

`requirements.in` lists the direct dependencies; `requirements.txt` is the
**compiled, pinned lockfile** you install from. Regenerate the lock for your
platform with:

```bash
make lock     # uv pip compile requirements.in -o requirements.txt  (or pip-tools)
```

### 1. pip / CPU (default, smoke-test ready)

```bash
make setup        # installs from the pinned requirements.txt + the package
make test         # fast CPU-only end-to-end smoke test (minutes), no GPU/native build
```

### 2. GPU / H100

The single most likely thing to break on a fresh install is the **torch CUDA
wheel selection** — no dependency manager fixes that for you, so it is explicit:

```bash
make setup                                   # base deps (installs a CPU torch first)
make setup-gpu                               # swaps in the cu124 H100 wheels:
#   pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
```

Adjust the CUDA tag (`cu121`/`cu124`/`cu126`) to your toolkit via
`make setup-gpu TORCH_CUDA_INDEX=https://download.pytorch.org/whl/cu126`.

### 3. conda (recommended for the liboqs / reported-results path)

conda-forge supplies `liboqs`, `cmake` and a compiler toolchain cleanly, which
sidesteps the painful part of building the native PQC backend:

```bash
conda env create -f environment.yml
conda activate qsfl
make setup-gpu                  # CUDA torch wheels
pip install liboqs-python==0.10.0   # builds against the conda-provided liboqs
```

Then select the native backend with `aggregation.kem.backend=liboqs`.

---

## Run

Everything is config-driven (`configs/*.yaml`), with CLI dotlist overrides.

```bash
# Headline single-model demo (Diagram A, N=1)
make demo
#   = python scripts/run_pipeline.py --config configs/single_model.yaml

# Full federated run (Diagram B)
make train
#   = python scripts/run_pipeline.py --config configs/federated.yaml

# Attack-evaluation suite (Layer 5)
make eval

# Override anything on the CLI:
python scripts/run_pipeline.py --config configs/federated.yaml \
    federated.rounds=30 aggregation.mode=masked model.backbone=vit \
    quantum.backend=lightning.gpu aggregation.kem.backend=liboqs
```

Results (metrics JSON + plots) are written under `results/` (git-ignored).

---

## Reproducibility

- **Global seeding** (Python/NumPy/torch) via `qsfl.utils.seeding.set_seed`.
- **Determinism for reported numbers:** torch deterministic algorithms +
  `CUBLAS_WORKSPACE_CONFIG` are enabled when `deterministic: true`. CUDA ops are
  nondeterministic by default; this makes accuracy/NC/BER reproducible at a small
  performance cost (documented, not hidden).
- **Deterministic quantum key:** `K_q` is reproducible given `(quantum.theta_seed,
  seed)` — this emulates QKD-style entropy; we do **not** claim true quantum
  randomness from a simulator.
- **Pinned deps + saved config:** the fully-resolved config is written next to
  every run's results.

---

## Testing

```bash
make test        # unit tests (CPU, no native build) + a tiny end-to-end run
```

The pytest suite includes a **hermetic end-to-end test** on synthetic data
(`data.name=synthetic`, no network/GPU) plus crypto round-trip, watermark
embed→extract, NC/BER, secure-aggregation mask cancellation, ledger immutability,
and an attack-suite smoke test. The gated liboqs test runs only when the native
backend is installed (`pytest -m liboqs`).

## Outputs

Each run writes to `results/<run_name>/`:

- `config.yaml` — the fully-resolved config (reproducibility).
- `results.json` — accuracies, `H = SHA256(C)`, ownership proof, verification verdict.
- `encrypted_model.json` — the encrypted protected model `C`.
- `attacks.json` + `pruning_robustness.png`, `attack_summary.png` (from `make eval`).

## Architecture

Six layers (see `src/qsfl/`):

1. **Medical data & clients** — MedMNIST v2 subsets emulate heterogeneous
   hospitals (MRI/CT/X-ray/pathology-like); IID and Dirichlet non-IID partitions.
2. **Quantum-safe security** — a Variational Quantum Circuit (R_Y rotations +
   CNOT entanglers) produces `K_q ∈ {0,1}^256`; hybrid HKDF + Kyber KEM +
   AES-256-GCM encrypt each update.
3. **Secure aggregation** — weighted FedAvg, either authorized-decrypt
   (default, simulation) or Bonawitz-style pairwise masking.
4. **Model protection** — Uchida-style white-box dual watermarking (visible image
   `P`, invisible bit-sequence `S`).
5. **Attack evaluation** — model extraction, membership inference, gradient
   leakage, fine-tuning, pruning, watermark removal, poisoning.
6. **Verification & deployment** — NC/BER/hash ownership check, ledger
   registration, authorized decrypt → extract → verify → deploy.

## Notation

| Symbol | Meaning | Symbol | Meaning |
|---|---|---|---|
| `W_i` | local model of client *i* | `W_P` | primary (visibly) watermarked weights |
| `ΔW_i` | local model update | `W_S` | dual watermarked weights |
| `K_q` | quantum-generated 256-bit key | `P, P'` | primary watermark (embedded / extracted) |
| `C_i` | encrypted update of client *i* | `S, S'` | secondary watermark (embedded / extracted) |
| `C` | encrypted model (ciphertext) | `NC` | normalized correlation |
| `W_g` | global federated model | `BER` | bit error rate |
| `H` | `SHA256(C)` ownership hash | `τ1,τ2,τ3` | verification thresholds |

## Repository layout

```
configs/        default + smoke/single_model/federated configs
src/qsfl/       data, quantum, crypto, federated, watermark, attacks,
                ledger, verification, pipeline, utils
scripts/        run_pipeline.py, run_eval.py, demo entrypoints
tests/          unit tests + fast CPU end-to-end smoke test
```

## License

MIT. Datasets retain their own licenses (MedMNIST v2 is CC BY 4.0).
