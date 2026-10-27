"""
Simulation for "Semantic Communication via Analog RF Computing" (ICC 2027),
MNIST-scale version.

Run this in Google Colab (ideally with a GPU runtime: Runtime -> Change
runtime type -> GPU) for the real dataset. It downloads MNIST automatically
via torchvision. If no internet is available (e.g. a sandboxed environment),
it transparently falls back to sklearn's bundled 8x8 digits dataset, with a
smaller encoder, so the exact same code and logic can be sanity-checked
offline before being run on the real thing.

Implements the same pipeline as the digits-scale prototype:
  - Shared encoder E (L=3 linear layers, tanh activations, linear final
    layer -> latent), executed with analog computing noise injected at every
    layer's PRE-activation (eq. 2-4), plus uplink channel noise on the
    latent (eq. 5-6)
  - Two task groups sharing the encoder: fine (10-class digit classification),
    coarse (binary: digit >= 5 vs < 5)
  - K clients split across the two groups, each with its own semantic noise
    profile eps_k = [eps_k^(1),...,eps_k^(L), eps_k^ch]
  - Semantic distortion Delta_k(eps_k) = E_x[KL(p_k(u|x) || phat_k(u|x;eps_k))]  (eq. 8)
  - Offline sensitivity estimation phi_k^(s) via probe-noise fitting (Sec. III-B)
  - Closed-form unified water-filling allocation (Lemma 1 / eq. 12)
  - Baselines: uniform, client-blind (stage-only) water-filling
  - Distortion-energy tradeoff sweep (Fig. 3) + sensitivity profile (Fig. 2)
  - Noise-aware co-design loop: alternate retraining and re-allocation (Fig. 5)
  - Validation of the second-order expansion (eq. 9)

Differences from the digits-scale prototype:
  - Real MNIST (28x28=784-dim input, 60k/10k train/test) instead of 8x8 digits
  - Larger encoder (784 -> 256 -> 64 -> 16) to match the input dimensionality
  - Minibatch training (not full-batch) since the dataset is much larger
  - GPU support (auto-detected)
  - The Gamma sweep range is auto-calibrated from the data/architecture
    instead of hand-tuned, so it doesn't need re-scanning per run
"""

import json
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0)
np.random.seed(0)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# --------------------------------------------------------------------------
# 0. Data: MNIST via torchvision, with a sklearn-digits fallback for offline
#    sanity checks (same code path, smaller scale).
# --------------------------------------------------------------------------
def load_dataset():
    try:
        from torchvision import datasets, transforms
        tfm = transforms.ToTensor()
        train_ds = datasets.MNIST(root="./data", train=True, download=True, transform=tfm)
        test_ds = datasets.MNIST(root="./data", train=False, download=True, transform=tfm)

        Xtr_full = train_ds.data.float().view(-1, 28 * 28) / 255.0
        ytr_full = train_ds.targets.long()
        Xte_full = test_ds.data.float().view(-1, 28 * 28) / 255.0
        yte_full = test_ds.targets.long()

        # calibration split carved out of the training set
        rng = np.random.RandomState(0)
        perm = rng.permutation(len(Xtr_full))
        n_calib = 5000
        idx_calib = perm[:n_calib]
        idx_train = perm[n_calib:]

        Xtr = Xtr_full[idx_train]
        ytr = ytr_full[idx_train]
        Xca = Xtr_full[idx_calib]
        yca = ytr_full[idx_calib]

        # subsample test set for faster evaluation sweeps; still real MNIST
        n_test_eval = 5000
        Xte = Xte_full[:n_test_eval]
        yte = yte_full[:n_test_eval]

        print(f"Loaded real MNIST: train={len(Xtr)} calib={len(Xca)} test={len(Xte)}")
        return Xtr, ytr, Xca, yca, Xte, yte, 28 * 28, "MNIST"

    except Exception as e:
        print(f"[warning] MNIST unavailable ({type(e).__name__}: {e}); "
              f"falling back to sklearn digits (8x8) for an offline sanity run. "
              f"In Colab with internet access this branch will not trigger.")
        from sklearn.datasets import load_digits
        digits = load_digits()
        X = digits.data.astype(np.float32) / 16.0
        y = digits.target.astype(np.int64)
        rng = np.random.RandomState(0)
        perm = rng.permutation(len(X))
        n_train = int(0.6 * len(X))
        n_calib = int(0.2 * len(X))
        idx_train = perm[:n_train]
        idx_calib = perm[n_train:n_train + n_calib]
        idx_test = perm[n_train + n_calib:]
        Xtr = torch.tensor(X[idx_train])
        ytr = torch.tensor(y[idx_train])
        Xca = torch.tensor(X[idx_calib])
        yca = torch.tensor(y[idx_calib])
        Xte = torch.tensor(X[idx_test])
        yte = torch.tensor(y[idx_test])
        print(f"Loaded digits fallback: train={len(Xtr)} calib={len(Xca)} test={len(Xte)}")
        return Xtr, ytr, Xca, yca, Xte, yte, 64, "digits"


Xtr, ytr, Xca, yca, Xte, yte, IN_DIM, DATASET_NAME = load_dataset()
ytr_c = (ytr >= 5).long()
yca_c = (yca >= 5).long()
yte_c = (yte >= 5).long()

Xtr, ytr, ytr_c = Xtr.to(DEVICE), ytr.to(DEVICE), ytr_c.to(DEVICE)
Xca, yca, yca_c = Xca.to(DEVICE), yca.to(DEVICE), yca_c.to(DEVICE)
Xte, yte, yte_c = Xte.to(DEVICE), yte.to(DEVICE), yte_c.to(DEVICE)

N_TRAIN = len(Xtr)

# --------------------------------------------------------------------------
# 1. Shared encoder + per-group decoders (scaled to IN_DIM)
# --------------------------------------------------------------------------
if IN_DIM >= 784:
    LAYER_DIMS = [256, 64, 16]
else:
    LAYER_DIMS = [32, 16, 8]
L = len(LAYER_DIMS)
NZ = LAYER_DIMS[-1]
STAGES = [f"L{l + 1}" for l in range(L)] + ["ch"]
S = len(STAGES)
print(f"Encoder: {IN_DIM} -> " + " -> ".join(str(d) for d in LAYER_DIMS) + f"  (L={L}, Nz={NZ})")


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        dims = [IN_DIM] + LAYER_DIMS
        self.layers = nn.ModuleList([nn.Linear(dims[i], dims[i + 1]) for i in range(L)])

    def forward_clean(self, x):
        h = x
        preacts = []
        for l, layer in enumerate(self.layers):
            y_raw = layer(h)
            preacts.append(y_raw)
            h = torch.tanh(y_raw) if l < L - 1 else y_raw
        return h, preacts

    def forward_noisy(self, x, eps_layers, S_bar_layers):
        h = x
        for l, layer in enumerate(self.layers):
            y_raw = layer(h)
            eps = eps_layers[l]
            if eps > 0:
                std = eps * (S_bar_layers[l] ** 0.5)
                y_raw = y_raw + std * torch.randn_like(y_raw)
            h = torch.tanh(y_raw) if l < L - 1 else y_raw
        return h


class FineDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(NZ, 32), nn.ReLU(), nn.Linear(32, 10))

    def forward(self, z):
        return self.net(z)


class CoarseDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Linear(NZ, 2)

    def forward(self, z):
        return self.net(z)


encoder = Encoder().to(DEVICE)
dec_fine = FineDecoder().to(DEVICE)
dec_coarse = CoarseDecoder().to(DEVICE)

# --------------------------------------------------------------------------
# 2. Clean (eps=0) joint multi-task training, minibatched
# --------------------------------------------------------------------------
BATCH_SIZE = 256


def iterate_minibatches(n, batch_size, shuffle=True):
    idx = np.arange(n)
    if shuffle:
        np.random.shuffle(idx)
    for i in range(0, n, batch_size):
        yield idx[i:i + batch_size]


BATCHES_PER_EPOCH = max(1, (N_TRAIN + BATCH_SIZE - 1) // BATCH_SIZE)
# target a fixed number of gradient steps regardless of dataset size, so the
# same script trains sensibly whether it falls back to the small digits set
# or runs on full MNIST
CLEAN_TARGET_STEPS = 1500
CLEAN_EPOCHS = max(5, -(-CLEAN_TARGET_STEPS // BATCHES_PER_EPOCH))
CODESIGN_TARGET_STEPS = 300
CODESIGN_EPOCHS = max(2, -(-CODESIGN_TARGET_STEPS // BATCHES_PER_EPOCH))
print(f"Batches/epoch={BATCHES_PER_EPOCH}  clean_epochs={CLEAN_EPOCHS}  codesign_epochs/round={CODESIGN_EPOCHS}")


def train_clean(epochs=CLEAN_EPOCHS, lr=1e-3, verbose=True):
    params = list(encoder.parameters()) + list(dec_fine.parameters()) + list(dec_coarse.parameters())
    opt = torch.optim.Adam(params, lr=lr)
    for ep in range(epochs):
        total_loss, nb = 0.0, 0
        for batch_idx in iterate_minibatches(N_TRAIN, BATCH_SIZE):
            bi = torch.as_tensor(batch_idx, device=DEVICE)
            xb, yb, ybc = Xtr[bi], ytr[bi], ytr_c[bi]
            opt.zero_grad()
            z, _ = encoder.forward_clean(xb)
            loss = F.cross_entropy(dec_fine(z), yb) + F.cross_entropy(dec_coarse(z), ybc)
            loss.backward()
            opt.step()
            total_loss += loss.item()
            nb += 1
        if verbose and (ep % max(1, epochs // 10) == 0 or ep == epochs - 1):
            with torch.no_grad():
                z, _ = encoder.forward_clean(Xte)
                acc_f = (dec_fine(z).argmax(1) == yte).float().mean().item()
                acc_c = (dec_coarse(z).argmax(1) == yte_c).float().mean().item()
            print(f"  epoch {ep+1}/{epochs}  loss={total_loss/nb:.4f}  "
                  f"test_acc_fine={acc_f:.3f}  test_acc_coarse={acc_c:.3f}")


print(f"\n[1] Training clean multi-task model on {DATASET_NAME}...")
t0 = time.time()
train_clean(epochs=CLEAN_EPOCHS, lr=1e-3)
print(f"  done in {time.time()-t0:.1f}s")

with torch.no_grad():
    z_test, _ = encoder.forward_clean(Xte)
    acc_f_test = (dec_fine(z_test).argmax(1) == yte).float().mean().item()
    acc_c_test = (dec_coarse(z_test).argmax(1) == yte_c).float().mean().item()
print(f"  final test acc: fine={acc_f_test:.3f}  coarse={acc_c_test:.3f}")

# --------------------------------------------------------------------------
# 3. Calibration: estimate S_bar^(ell) and Sz_bar from clean forward pass
# --------------------------------------------------------------------------
def calibrate():
    with torch.no_grad():
        z, preacts = encoder.forward_clean(Xca)
        S_bar_layers = [float((p ** 2).mean().item()) for p in preacts]
        Sz_bar = float((z ** 2).mean().item())
    return S_bar_layers, Sz_bar


S_bar_layers, Sz_bar = calibrate()
print(f"\n[2] Calibration: S_bar per layer = {[f'{s:.4f}' for s in S_bar_layers]}  Sz_bar = {Sz_bar:.4f}")

# --------------------------------------------------------------------------
# 4. Clients and task assignment
# --------------------------------------------------------------------------
K = 10
K_FINE = 5
K_COARSE = 5
client_group = ["fine"] * K_FINE + ["coarse"] * K_COARSE

# --------------------------------------------------------------------------
# 4b. Non-iid per-client calibration/eval subsets (Dirichlet class skew).
#     Without this, every client in a group sees the SAME calibration/eval
#     data and therefore gets an IDENTICAL sensitivity profile phi_k^(s),
#     so the per-client water-filling has nothing client-specific to
#     exploit beyond the (mild) per-client energy coefficients omega_{k,s}.
#     Giving each client its own locally-skewed slice of the data (as in
#     non-iid federated-learning benchmarks) makes phi genuinely differ
#     client-to-client, which is what the "unified" water-filling is
#     supposed to exploit relative to the client-blind baseline.
# --------------------------------------------------------------------------
N_CLASSES = 10
NONIID_ALPHA = 0.12          # smaller => more skewed per-client class mixtures
SAMPLES_PER_CLIENT_CALIB = 400
SAMPLES_PER_CLIENT_EVAL = 400


def build_noniid_client_indices(y, n_classes, n_clients, samples_per_client, alpha, seed):
    rng = np.random.RandomState(seed)
    y_np = y.detach().cpu().numpy()
    class_pools = {c: np.where(y_np == c)[0] for c in range(n_classes)}
    client_idx = []
    for k in range(n_clients):
        weights = rng.dirichlet(alpha * np.ones(n_classes))
        counts = rng.multinomial(samples_per_client, weights)
        idx_parts = []
        for c in range(n_classes):
            pool = class_pools[c]
            if len(pool) == 0 or counts[c] == 0:
                continue
            idx_parts.append(rng.choice(pool, size=counts[c], replace=True))
        idx_k = np.concatenate(idx_parts) if idx_parts else rng.choice(len(y_np), size=samples_per_client)
        client_idx.append(idx_k)
    return client_idx


_calib_idx = build_noniid_client_indices(yca, N_CLASSES, K, SAMPLES_PER_CLIENT_CALIB, NONIID_ALPHA, seed=100)
_eval_idx = build_noniid_client_indices(yte, N_CLASSES, K, SAMPLES_PER_CLIENT_EVAL, NONIID_ALPHA, seed=200)

Xca_k = {k: Xca[_calib_idx[k]] for k in range(K)}
Xte_k = {k: Xte[_eval_idx[k]] for k in range(K)}
yte_k = {k: yte[_eval_idx[k]] for k in range(K)}
ytec_k = {k: yte_c[_eval_idx[k]] for k in range(K)}

print("\n  Per-client local class mixture (Dirichlet alpha="
      f"{NONIID_ALPHA}, showing most-common digit):")
for k in range(K):
    y_local = yca_k_preview = yca[_calib_idx[k]].detach().cpu().numpy()
    vals, cnts = np.unique(y_local, return_counts=True)
    top = vals[np.argmax(cnts)]
    print(f"  client {k} ({client_group[k]:6s}): dominant digit={top}, "
          f"n={len(y_local)}, unique digits seen={len(vals)}")

# --------------------------------------------------------------------------
# 5. Noisy forward pass / posteriors / semantic distortion for client k
# --------------------------------------------------------------------------
def client_forward(X, eps_vec, S_bar_layers, Sz_bar):
    z = encoder.forward_noisy(X, eps_vec[:L], S_bar_layers)
    eps_ch = eps_vec[L]
    if eps_ch > 0:
        std = eps_ch * (Sz_bar ** 0.5)
        z = z + std * torch.randn_like(z)
    return z


def client_posteriors(X, eps_vec, group, n_mc=8):
    with torch.no_grad():
        z_clean, _ = encoder.forward_clean(X)
        dec = dec_fine if group == "fine" else dec_coarse
        p = F.softmax(dec(z_clean), dim=1)
        probs_accum = torch.zeros_like(p)
        for _ in range(n_mc):
            z_noisy = client_forward(X, eps_vec, S_bar_layers, Sz_bar)
            phat = F.softmax(dec(z_noisy), dim=1)
            probs_accum += phat
        phat_avg = probs_accum / n_mc
    return p, phat_avg


def semantic_distortion(X, eps_vec, group, n_mc=8, eps_floor=1e-8):
    with torch.no_grad():
        z_clean, _ = encoder.forward_clean(X)
        dec = dec_fine if group == "fine" else dec_coarse
        p = F.softmax(dec(z_clean), dim=1).clamp_min(eps_floor)
        kl_accum = 0.0
        for _ in range(n_mc):
            z_noisy = client_forward(X, eps_vec, S_bar_layers, Sz_bar)
            phat = F.softmax(dec(z_noisy), dim=1).clamp_min(eps_floor)
            kl = (p * (p.log() - phat.log())).sum(dim=1)
            kl_accum += kl.mean().item()
        return kl_accum / n_mc


print("\n[3] Sanity check: distortion at eps=0 should be ~0")
for grp in ["fine", "coarse"]:
    d0 = semantic_distortion(Xca, np.zeros(S), grp, n_mc=4)
    print(f"  group={grp}: Delta={d0:.6f}")

# --------------------------------------------------------------------------
# 6. Offline sensitivity estimation: probe single-stage noise, fit phi_k^(s)
# --------------------------------------------------------------------------
PROBE_EPS = np.array([0.03, 0.06, 0.10, 0.15])


def estimate_sensitivity(k, stage_idx, n_mc=12):
    group = client_group[k]
    X_local = Xca_k[k]
    distortions = []
    for eps_probe in PROBE_EPS:
        eps_vec = np.zeros(S)
        eps_vec[stage_idx] = eps_probe
        d = semantic_distortion(X_local, eps_vec, group, n_mc=n_mc)
        distortions.append(d)
    distortions = np.array(distortions)
    x = PROBE_EPS ** 2
    phi = float((x @ distortions) / (x @ x))
    return max(phi, 1e-12), distortions


print("\n[4] Estimating semantic sensitivities phi_k^(s) per client/stage...")
t0 = time.time()
phi = np.zeros((K, S))
for k in range(K):
    for s in range(S):
        phi[k, s], _ = estimate_sensitivity(k, s, n_mc=12)
print(f"  done in {time.time()-t0:.1f}s")

print("\n  Sensitivity table (rows=clients, cols=stages):")
for k in range(K):
    print(f"  client {k} ({client_group[k]:6s}): " + "  ".join(f"{phi[k,s]:.4e}" for s in range(S)))

# --------------------------------------------------------------------------
# 7. Energy coefficients omega_{k,s} (synthetic but structured; see
#    Sec. II-D and the note in the accompanying writeup about why these are
#    not yet a real beamformed-channel realization)
# --------------------------------------------------------------------------
rng2 = np.random.RandomState(42)
layer_base = np.array([1.0 + 0.3 * l for l in range(L)])
# each client also gets a single "hardware/channel quality" scalar, modeling
# the fact that real edge devices and links differ by much more than mild
# per-layer jitter around a common baseline (some clients simply have much
# noisier analog front-ends or much worse uplink channels than others)
client_quality = rng2.lognormal(mean=0.0, sigma=0.6, size=K)
omega = np.zeros((K, S))
for k in range(K):
    hw_variability = rng2.lognormal(mean=0.0, sigma=0.5, size=L)
    omega[k, :L] = layer_base * hw_variability * client_quality[k]
    omega[k, L] = rng2.lognormal(mean=0.3, sigma=0.9) * client_quality[k]

print("\n[5] Energy coefficients omega_{k,s}:")
for k in range(K):
    print(f"  client {k} ({client_group[k]:6s}): " + "  ".join(f"{omega[k,s]:.3f}" for s in range(S)))

# --------------------------------------------------------------------------
# 8. Allocation schemes
# --------------------------------------------------------------------------
def water_filling_unified(phi, omega, Gamma):
    sqrt_term = np.sqrt(phi / omega)
    denom = np.sum(np.sqrt(phi * omega))
    gamma = Gamma * sqrt_term / denom
    eps = gamma ** (-0.5)
    return eps, gamma


def water_filling_client_blind(phi, omega, Gamma):
    phi_agg = phi.sum(axis=0)
    omega_agg = omega.sum(axis=0)
    sqrt_term = np.sqrt(phi_agg / omega_agg)
    denom = np.sum(np.sqrt(phi_agg * omega_agg))
    gamma_s = Gamma * sqrt_term / denom
    eps_s = gamma_s ** (-0.5)
    return np.tile(eps_s, (K, 1)), np.tile(gamma_s, (K, 1))


def uniform_allocation(omega, Gamma):
    eps_common = np.sqrt(omega.sum() / Gamma)
    eps_full = np.full((K, S), eps_common)
    return eps_full, eps_full ** (-2.0)


def total_energy(omega, gamma):
    return float(np.sum(omega * gamma))


# auto-calibrate the Gamma sweep range from the achieved uniform-baseline
# eps (want eps to span roughly [1.0, 0.05], the small-noise-ish regime)
total_omega = omega.sum()
GAMMA_LOW = total_omega / (1.0 ** 2)
GAMMA_HIGH = total_omega / (0.05 ** 2)
GAMMA_SWEEP = np.geomspace(GAMMA_LOW, GAMMA_HIGH, 9)
print(f"\n[6] Auto-calibrated Gamma sweep: {GAMMA_SWEEP[0]:.1f} .. {GAMMA_SWEEP[-1]:.1f}")

eps_wf, gamma_wf = water_filling_unified(phi, omega, GAMMA_SWEEP[len(GAMMA_SWEEP) // 2])
print(f"  mid-sweep eps range (proposed): [{eps_wf.min():.3f}, {eps_wf.max():.3f}]")

# --------------------------------------------------------------------------
# 9. Evaluation
# --------------------------------------------------------------------------
def evaluate_allocation(eps_full, n_mc=15):
    per_client_distortion = np.zeros(K)
    correct = {"fine": 0, "coarse": 0}
    count = {"fine": 0, "coarse": 0}
    for k in range(K):
        grp = client_group[k]
        eps_vec = eps_full[k]
        X_eval, y_eval, yc_eval = Xte_k[k], yte_k[k], ytec_k[k]
        per_client_distortion[k] = semantic_distortion(X_eval, eps_vec, grp, n_mc=n_mc)
        p_clean, phat_avg = client_posteriors(X_eval, eps_vec, grp, n_mc=n_mc)
        pred = phat_avg.argmax(dim=1)
        true = y_eval if grp == "fine" else yc_eval
        correct[grp] += (pred == true).sum().item()
        count[grp] += len(true)
    return {
        "per_client_distortion": per_client_distortion,
        "network_distortion": per_client_distortion.sum(),
        "acc_fine": correct["fine"] / count["fine"] if count["fine"] else float("nan"),
        "acc_coarse": correct["coarse"] / count["coarse"] if count["coarse"] else float("nan"),
    }


# --------------------------------------------------------------------------
# 10. Fig. 3 — Distortion-energy tradeoff sweep
# --------------------------------------------------------------------------
print("\n[7] Running distortion-energy tradeoff sweep (Fig. 3)...")
t0 = time.time()
results_uniform, results_client_blind, results_proposed = [], [], []

for Gamma in GAMMA_SWEEP:
    eps_un, _ = uniform_allocation(omega, Gamma)
    eps_cb, _ = water_filling_client_blind(phi, omega, Gamma)
    eps_wf2, _ = water_filling_unified(phi, omega, Gamma)

    res_un = evaluate_allocation(eps_un, n_mc=30)
    res_cb = evaluate_allocation(eps_cb, n_mc=30)
    res_wf = evaluate_allocation(eps_wf2, n_mc=30)

    results_uniform.append(res_un)
    results_client_blind.append(res_cb)
    results_proposed.append(res_wf)

    print(f"  Gamma={Gamma:10.1f}  "
          f"D_uniform={res_un['network_distortion']:.4f} (fine={res_un['acc_fine']:.3f},coarse={res_un['acc_coarse']:.3f})  "
          f"D_clientblind={res_cb['network_distortion']:.4f} (fine={res_cb['acc_fine']:.3f},coarse={res_cb['acc_coarse']:.3f})  "
          f"D_proposed={res_wf['network_distortion']:.4f} (fine={res_wf['acc_fine']:.3f},coarse={res_wf['acc_coarse']:.3f})")

print(f"  done in {time.time()-t0:.1f}s")

_D_uniform_tmp = [r["network_distortion"] for r in results_uniform]
_D_cb_tmp = [r["network_distortion"] for r in results_client_blind]
_D_wf_tmp = [r["network_distortion"] for r in results_proposed]
_pct_vs_uniform_tmp = [100.0 * (du - dp) / du for du, dp in zip(_D_uniform_tmp, _D_wf_tmp)]
_pct_vs_cb_tmp = [100.0 * (dc - dp) / dc for dc, dp in zip(_D_cb_tmp, _D_wf_tmp)]

with open("sweep_results_mnist.json", "w") as f:
    json.dump({
        "dataset": DATASET_NAME,
        "gamma": GAMMA_SWEEP.tolist(),
        "uniform": [{"D": r["network_distortion"], "acc_fine": r["acc_fine"], "acc_coarse": r["acc_coarse"]} for r in results_uniform],
        "client_blind": [{"D": r["network_distortion"], "acc_fine": r["acc_fine"], "acc_coarse": r["acc_coarse"]} for r in results_client_blind],
        "proposed": [{"D": r["network_distortion"], "acc_fine": r["acc_fine"], "acc_coarse": r["acc_coarse"]} for r in results_proposed],
        "pct_distortion_reduction_vs_uniform": _pct_vs_uniform_tmp,
        "pct_distortion_reduction_vs_client_blind": _pct_vs_cb_tmp,
    }, f, indent=2)

# --------------------------------------------------------------------------
# 11. Plots: Fig. 2, Fig. 3
# --------------------------------------------------------------------------
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, NullFormatter

# Fonts are set for the FINAL printed size: figures below are drawn at the
# width they occupy in the paper (3.5 in = one IEEE column, 7.16 in = two
# columns), so the font sizes here are the printed sizes in points.
plt.rcParams.update({
    "font.size": 9, "axes.labelsize": 9, "axes.titlesize": 9,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 7.5,
    "lines.markersize": 4, "lines.linewidth": 1.2,
    "figure.dpi": 150, "savefig.dpi": 300,
})

fig, ax = plt.subplots(figsize=(3.5, 2.7))
x = np.arange(S)
fine_idx = [k for k in range(K) if client_group[k] == "fine"]
coarse_idx = [k for k in range(K) if client_group[k] == "coarse"]
phi_fine_mean, phi_fine_std = phi[fine_idx].mean(axis=0), phi[fine_idx].std(axis=0)
phi_coarse_mean, phi_coarse_std = phi[coarse_idx].mean(axis=0), phi[coarse_idx].std(axis=0)
bar_w = 0.32
ax.bar(x - bar_w / 2, phi_fine_mean, bar_w, yerr=phi_fine_std, capsize=2,
       label="fine-grained task group", color="#2b6cb0")
ax.bar(x + bar_w / 2, phi_coarse_mean, bar_w, yerr=phi_coarse_std, capsize=2,
       label="coarse task group", color="#dd6b20")
ax.set_xticks(x)
ax.set_xticklabels(STAGES)
ax.set_xlabel("Stage (encoder layers, uplink ch)")
ax.set_ylabel(r"Semantic sensitivity $\phi_k^{(s)}$")
ax.legend()
fig.tight_layout()
fig.savefig("fig2_sensitivity_profile_mnist.png")
print("\n[8] Saved fig2_sensitivity_profile_mnist.png")

D_uniform = [r["network_distortion"] for r in results_uniform]
D_cb = [r["network_distortion"] for r in results_client_blind]
D_wf = [r["network_distortion"] for r in results_proposed]
acc_fine_uniform = [r["acc_fine"] for r in results_uniform]
acc_fine_cb = [r["acc_fine"] for r in results_client_blind]
acc_fine_wf = [r["acc_fine"] for r in results_proposed]

# percentage distortion reduction of the proposed scheme vs. each baseline
pct_reduction_vs_uniform = [100.0 * (du - dp) / du for du, dp in zip(D_uniform, D_wf)]
pct_reduction_vs_cb = [100.0 * (dc - dp) / dc for dc, dp in zip(D_cb, D_wf)]

print("\n  Distortion reduction of proposed scheme at each budget:")
for g, pu, pc in zip(GAMMA_SWEEP, pct_reduction_vs_uniform, pct_reduction_vs_cb):
    print(f"    Gamma={g:10.1f}  vs uniform: {pu:5.1f}%   vs client-blind: {pc:5.1f}%")

fig, axes = plt.subplots(1, 3, figsize=(7.16, 2.5))
ax = axes[0]
ax.loglog(GAMMA_SWEEP, D_uniform, "o-", label="Uniform", color="#718096")
ax.loglog(GAMMA_SWEEP, D_cb, "s-", label="Client-blind", color="#dd6b20")
ax.loglog(GAMMA_SWEEP, D_wf, "^-", label="Proposed", color="#2b6cb0")
ax.set_xlabel(r"Energy budget $\Gamma$")
ax.set_ylabel(r"Network distortion $\sum_k \Delta_k$")
ax.set_title("Distortion")
ax.legend()
ax.grid(True, which="major", alpha=0.3)
ax.xaxis.set_minor_locator(LogLocator(base=10, subs="auto", numticks=12))
ax.xaxis.set_minor_formatter(NullFormatter())

ax = axes[1]
ax.semilogx(GAMMA_SWEEP, pct_reduction_vs_uniform, "o-", label="vs. uniform", color="#718096")
ax.semilogx(GAMMA_SWEEP, pct_reduction_vs_cb, "s-", label="vs. client-blind", color="#dd6b20")
ax.axhline(0, color="black", linewidth=0.8, alpha=0.5)
ax.set_xlabel(r"Energy budget $\Gamma$")
ax.set_ylabel("Distortion reduction (%)")
ax.set_title("Relative gain")
ax.legend()
ax.grid(True, which="major", alpha=0.3)
ax.xaxis.set_minor_locator(LogLocator(base=10, subs="auto", numticks=12))
ax.xaxis.set_minor_formatter(NullFormatter())

ax = axes[2]
ax.semilogx(GAMMA_SWEEP, acc_fine_uniform, "o-", label="Uniform", color="#718096")
ax.semilogx(GAMMA_SWEEP, acc_fine_cb, "s-", label="Client-blind", color="#dd6b20")
ax.semilogx(GAMMA_SWEEP, acc_fine_wf, "^-", label="Proposed", color="#2b6cb0")
ax.set_xlabel(r"Energy budget $\Gamma$")
ax.set_ylabel("Fine-task accuracy")
ax.set_title("Accuracy")
ax.legend()
ax.grid(True, which="major", alpha=0.3)
ax.xaxis.set_minor_locator(LogLocator(base=10, subs="auto", numticks=12))
ax.xaxis.set_minor_formatter(NullFormatter())

fig.tight_layout()
fig.savefig("fig3_distortion_energy_tradeoff_mnist.png")
print("[9] Saved fig3_distortion_energy_tradeoff_mnist.png")

# --------------------------------------------------------------------------
# 12. Noise-aware co-design loop (Fig. 5), minibatched
# --------------------------------------------------------------------------
print("\n[10] Noise-aware co-design loop (Fig. 5)...")
GAMMA_FIXED = GAMMA_SWEEP[len(GAMMA_SWEEP) // 2]
N_ROUNDS = 5
TRAIN_EPOCHS_PER_ROUND = CODESIGN_EPOCHS


def train_noise_aware(eps_full, S_bar_layers, Sz_bar, epochs=2, lr=5e-4):
    params = list(encoder.parameters()) + list(dec_fine.parameters()) + list(dec_coarse.parameters())
    opt = torch.optim.Adam(params, lr=lr)
    last_loss = None
    for ep in range(epochs):
        for batch_idx in iterate_minibatches(N_TRAIN, BATCH_SIZE):
            bi = torch.as_tensor(batch_idx, device=DEVICE)
            opt.zero_grad()
            total_loss = 0.0
            for k in range(K):
                grp = client_group[k]
                eps_vec = eps_full[k]
                xb = Xtr[bi]
                z_noisy = client_forward(xb, eps_vec, S_bar_layers, Sz_bar)
                dec = dec_fine if grp == "fine" else dec_coarse
                y_true = (ytr if grp == "fine" else ytr_c)[bi]
                total_loss = total_loss + F.cross_entropy(dec(z_noisy), y_true)
            total_loss = total_loss / K
            total_loss.backward()
            opt.step()
            last_loss = total_loss.item()
    return last_loss


eps_current, _ = water_filling_unified(phi, omega, GAMMA_FIXED)
res0 = evaluate_allocation(eps_current, n_mc=20)
codesign_history = [{"round": 0, "network_distortion": res0["network_distortion"],
                      "acc_fine": res0["acc_fine"], "acc_coarse": res0["acc_coarse"]}]
print(f"  round 0 (clean-trained model): D={res0['network_distortion']:.4f}  "
      f"acc_fine={res0['acc_fine']:.3f}  acc_coarse={res0['acc_coarse']:.3f}")

for rnd in range(1, N_ROUNDS + 1):
    t0 = time.time()
    final_loss = train_noise_aware(eps_current, S_bar_layers, Sz_bar, epochs=TRAIN_EPOCHS_PER_ROUND)
    S_bar_layers, Sz_bar = calibrate()
    for k in range(K):
        for s in range(S):
            phi[k, s], _ = estimate_sensitivity(k, s, n_mc=12)
    eps_current, _ = water_filling_unified(phi, omega, GAMMA_FIXED)
    res = evaluate_allocation(eps_current, n_mc=20)
    codesign_history.append({"round": rnd, "network_distortion": res["network_distortion"],
                              "acc_fine": res["acc_fine"], "acc_coarse": res["acc_coarse"]})
    print(f"  round {rnd} ({time.time()-t0:.1f}s, train_loss={final_loss:.4f}): "
          f"D={res['network_distortion']:.4f}  acc_fine={res['acc_fine']:.3f}  acc_coarse={res['acc_coarse']:.3f}")

with open("codesign_history_mnist.json", "w") as f:
    json.dump(codesign_history, f, indent=2)

rounds = [h["round"] for h in codesign_history]
D_hist = [h["network_distortion"] for h in codesign_history]
accf_hist = [h["acc_fine"] for h in codesign_history]
accc_hist = [h["acc_coarse"] for h in codesign_history]

fig, axes = plt.subplots(2, 1, figsize=(3.5, 3.9), sharex=True)
ax = axes[0]
ax.plot(rounds, D_hist, "o-", color="#2b6cb0")
ax.set_ylabel(r"Network distortion $\sum_k \Delta_k$")
ax.grid(True, alpha=0.3)
ax.set_xticks(rounds)

ax = axes[1]
ax.plot(rounds, accf_hist, "o-", label="fine task", color="#2b6cb0")
ax.plot(rounds, accc_hist, "s-", label="coarse task", color="#dd6b20")
ax.set_xlabel("Alternation round")
ax.set_ylabel("Task accuracy")
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_xticks(rounds)

fig.tight_layout()
fig.savefig("fig5_codesign_loop_mnist.png")
print("\n[11] Saved fig5_codesign_loop_mnist.png")

# --------------------------------------------------------------------------
# 13. Validate the second-order expansion (eq. 9)
# --------------------------------------------------------------------------
print("\n[12] Validating second-order expansion D_k(eps) ~= phi_k^(s) * eps^2 ...")
eps_check = 0.08
rel_errors = []
for k in range(K):
    grp = client_group[k]
    for s in range(S):
        eps_vec = np.zeros(S)
        eps_vec[s] = eps_check
        D_actual = semantic_distortion(Xca_k[k], eps_vec, grp, n_mc=20)
        D_pred = phi[k, s] * eps_check ** 2
        if D_actual > 1e-9:
            rel_errors.append(abs(D_actual - D_pred) / D_actual)
rel_errors = np.array(rel_errors)
print(f"  median relative error = {100*np.median(rel_errors):.1f}%  "
      f"(mean = {100*np.mean(rel_errors):.1f}%, n={len(rel_errors)} client-stage pairs)")

print("\n=== SUMMARY ===")
print(f"Dataset: {DATASET_NAME}")
print(f"Clean test accuracy: fine={acc_f_test:.3f}, coarse={acc_c_test:.3f}")
print(f"Distortion reduction, proposed vs uniform: "
      f"{100*(D_uniform[0]-D_wf[0])/D_uniform[0]:.1f}% (low budget) to "
      f"{100*(D_uniform[-1]-D_wf[-1])/D_uniform[-1]:.1f}% (high budget)")
print(f"Co-design: D went from {D_hist[0]:.4f} (round 0) to {D_hist[-1]:.4f} (round {N_ROUNDS}), "
      f"a {100*(D_hist[0]-D_hist[-1])/D_hist[0]:.1f}% reduction at fixed budget Gamma={GAMMA_FIXED:.0f}")
