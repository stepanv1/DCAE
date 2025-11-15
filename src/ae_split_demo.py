#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AE splitting demo (CPU-only, multi-core). Shows MSE-only vs. MSE+DCAE.

Run (CLI):
  python ae_split_demo.py --m 15 --k 2 --N 12000 --epochs-mse 20 --epochs-dcae 20 \
    --lambda-dcae 3e-4 --batch-size 4096 --threads 56 --interop-threads 8 \
    --num-workers 8 --prefetch-factor 4 --compile

Run (PyCharm console):
  import importlib, ae_split_demo
  importlib.reload(ae_split_demo)
  ae_split_demo.main([
      '--m','15','--k','2','--N','12000',
      '--epochs-mse','20','--epochs-dcae','20','--lambda-dcae','3e-4',
      '--batch-size','4096','--threads','56','--interop-threads','8',
      '--num-workers','8','--prefetch-factor','4','--compile'
  ])
"""

import os, sys, time, argparse
from pathlib import Path
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from copy import deepcopy
import matplotlib
matplotlib.use("Agg")
import torch
import matplotlib.pyplot as plt
from typing import Optional, Sequence

def sample_l_ball_with_noise(
        N: int,
        m: int,
        ell: int,
        *,
        radius: float = 1.0,
        noise_sigma: float = 0.1,
        rotate: bool = False,
        signal_indices: Optional[Sequence[int]] = None,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    Sample X ∈ R^{N×m}:
      - first, draw points uniformly from an ℓ-ball (radius `radius`) in R^ℓ;
      - then fill the remaining m-ℓ coordinates with i.i.d. N(0, noise_sigma^2);
      - optionally apply a random orthogonal rotation in R^m.

    Args
    ----
    N : number of samples
    m : ambient dimension
    ell : intrinsic dimension of the signal-ball (1 ≤ ell < m)
    radius : radius of the ℓ-ball for the signal part
    noise_sigma : std dev of the Gaussian noise in the (m-ell) nuisance coords
    rotate : if True, apply a random orthogonal transform to mix subspaces
    signal_indices : optional indices (length ell) where to place the signal;
                     if None, uses the first `ell` coordinates
    device, dtype : tensor placement and dtype

    Returns
    -------
    X : (N, m) tensor
    """
    assert 1 <= ell < m, "ell must satisfy 1 ≤ ell < m"

    device = device or torch.device("cpu")

    # --- Uniform on ℓ-ball via direction * radius ---
    # Direction ~ Normal(0, I_ℓ) normalized
    dir_ell = torch.randn(N, ell, device=device, dtype=dtype)
    dir_ell = dir_ell / dir_ell.norm(dim=1, keepdim=True).clamp_min(1e-12)

    # Radius ~ U(0,1)^(1/ℓ) scaled by `radius`
    r = torch.rand(N, device=device, dtype=dtype).pow(1.0 / ell) * radius
    signal = dir_ell * r.unsqueeze(1)  # (N, ell)

    # --- Noise in the remaining m-ell coordinates ---
    noise = torch.randn(N, m - ell, device=device, dtype=dtype) * noise_sigma

    # --- Assemble into R^m (axis-aligned by default) ---
    X = torch.empty(N, m, device=device, dtype=dtype)
    if signal_indices is None:
        # put signal in the first `ell` dims
        X[:, :ell] = signal
        X[:, ell:] = noise
    else:
        assert len(signal_indices) == ell, "`signal_indices` length must be ell"
        mask = torch.ones(m, dtype=torch.bool, device=device)
        mask[torch.as_tensor(signal_indices, device=device)] = False
        X[:, mask] = noise
        X[:, signal_indices] = signal

    # --- Optional random rotation to avoid axis alignment ---
    if rotate:
        # Haar-ish random orthogonal via QR
        Q, _ = torch.linalg.qr(torch.randn(m, m, device=device, dtype=dtype))
        # enforce det(Q) > 0 (optional)
        if torch.det(Q) < 0:
            Q[:, 0] = -Q[:, 0]
        X = X @ Q.T

    return X


# --------------------------- CPU configuration --------------------------------
def configure_cpu(args):
    os.environ["OMP_NUM_THREADS"] = str(args.threads)
    os.environ["MKL_NUM_THREADS"] = str(args.threads)
    os.environ["OPENBLAS_NUM_THREADS"] = str(args.threads)
    os.environ["NUMEXPR_NUM_THREADS"] = str(args.threads)
    os.environ["VECLIB_MAXIMUM_THREADS"] = str(args.threads)
    try: torch.set_num_threads(args.threads)
    except Exception: pass
    try: torch.set_num_interop_threads(args.interop_threads)
    except Exception: pass

def maybe_compile_for_training(enc_eager, dec_eager, enable_compile: bool):
    if not enable_compile or not hasattr(torch, "compile"):
        return enc_eager, dec_eager
    try:
        enc_train = torch.compile(enc_eager, dynamic=True)
        dec_train = torch.compile(dec_eager, dynamic=True)
        print("[compile] torch.compile enabled (CPU)")
        return enc_train, dec_train
    except Exception as e:
        print(f"[compile] skipped: {e}")
        return enc_eager, dec_eager


# ----------------------------- Data generation --------------------------------
def sample_unit_ball(N: int, m: int) -> torch.Tensor:
    """Uniform in unit m-ball via Gaussian direction + U^(1/m) radius."""
    X = torch.randn(N, m)
    X = X / X.norm(dim=1, keepdim=True).clamp_min(1e-12)
    r = torch.rand(N).pow(1.0 / m)
    return X * r.unsqueeze(1)

def generate_dataset(N: int, m: int, noise_sigma: float = 0.0,
                     ell: Optional[int] = None, rotate: bool = False) -> torch.Tensor:
    if ell is None or ell >= m:
        # original full m-ball
        X = torch.randn(N, m)
        X = X / X.norm(dim=1, keepdim=True).clamp_min(1e-12)
        r = torch.rand(N).pow(1.0 / m)
        return (X * r.unsqueeze(1)) + (noise_sigma * torch.randn(N, m))  # if you still want isotropic noise
    else:
        return sample_l_ball_with_noise(N, m, ell, noise_sigma=noise_sigma, rotate=rotate)

def train_val_split(X: torch.Tensor, val_frac: float = 0.2, seed: int = 42):
    """Random train/val split (CPU tensor in, two CPU tensors out)."""
    assert 0.0 < val_frac < 1.0, "--val-frac must be in (0,1)"
    N = X.shape[0]
    n_val = int(round(N * val_frac))
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(N, generator=g)
    idx_val = perm[:n_val]
    idx_tr  = perm[n_val:]
    return X[idx_tr].contiguous(), X[idx_val].contiguous()

def make_loader(X_cpu: torch.Tensor, args, *, shuffle: bool = True) -> DataLoader:
    ds = TensorDataset(X_cpu)  # yields tuples (X,)
    return DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=shuffle,
        drop_last=False,
        num_workers=args.num_workers,
        pin_memory=False,  # CPU-only
        persistent_workers=(args.num_workers > 0),
        prefetch_factor=(args.prefetch_factor if args.num_workers > 0 else None),
    )

# --------------------------------- Models -------------------------------------
class MLP(nn.Module):
    def __init__(self, dims, act="elu", last_linear=False):
        super().__init__()
        layers = []
        Act = nn.ELU if act == "elu" else nn.ReLU
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            if i < len(dims) - 2 or not last_linear:
                layers.append(Act())
        self.net = nn.Sequential(*layers)
    def forward(self, x): return self.net(x)

class Encoder(nn.Module):
    def __init__(self, m, k):
        super().__init__()
        self.core = MLP([m, 512, 256, 128, 64, k], act="elu", last_linear=True)
    def forward(self, x): return self.core(x)

class Decoder(nn.Module):
    def __init__(self, k, m):
        super().__init__()
        self.core = MLP([k, 64, 128, 256, 512, m], act="elu", last_linear=True)
    def forward(self, z): return self.core(z)

# ------------------------------ DCAE penalty ----------------------------------
def dcae_contractive_penalty_hutchinson_from_z(
    z: torch.Tensor,
    xb: torch.Tensor,
    probes: int = 1,
    *,
    create_graph: bool = True,
    retain_graph: bool = True,
) -> torch.Tensor:
    """
    Contractive penalty using an existing forward z = encoder(xb).
    If create_graph=False/retain_graph=False, this is safe for validation.
    """
    if not xb.requires_grad:
        xb.requires_grad_(True)
    B, _ = z.shape
    pen = xb.new_tensor(0.0)
    for _ in range(probes):
        v = torch.empty_like(z).bernoulli_(0.5).mul_(2.0).sub_(1.0)  # ±1
        vz = (z * v).sum()
        gx = torch.autograd.grad(
            vz, xb,
            create_graph=create_graph,
            retain_graph=retain_graph,
            only_inputs=True
        )[0]
        pen = pen + (gx.pow(2).sum() / B)
    return pen / probes


# ------------------------------ Training epoch --------------------------------
def train_epoch(enc, dec, optE, optD, loader, device,
                lam_dcae: float = 0.0, use_dcae: bool = False, dcae_probes: int = 1):
    enc.train(); dec.train()
    mse_sum = dcae_sum = 0.0
    n = 0

    for (xb,) in loader:
        xb = xb.to(device, non_blocking=False)

        # Only ask for input grads when we actually use the DCAE penalty
        if use_dcae and lam_dcae > 0.0:
            xb = xb.requires_grad_(True)

        optE.zero_grad(set_to_none=True)
        optD.zero_grad(set_to_none=True)

        z = enc(xb)                 # ONE forward used for both terms
        xhat = dec(z)
        mse = F.mse_loss(xhat, xb, reduction="mean")

        if use_dcae and lam_dcae > 0.0:
            pen = dcae_contractive_penalty_hutchinson_from_z(z, xb, probes=dcae_probes)
            loss = mse + lam_dcae * pen
        else:
            pen = xb.new_tensor(0.0)
            loss = mse

        loss.backward()
        optE.step(); optD.step()

        bsz = xb.shape[0]
        mse_sum += mse.detach().item() * bsz
        dcae_sum += pen.detach().item() * bsz
        n += bsz

    mse_mean = mse_sum / max(1, n)
    dcae_mean = dcae_sum / max(1, n)
    return {"loss": (mse_mean + lam_dcae * dcae_mean), "mse": mse_mean, "dcae": dcae_mean}


# ------------------------------ Plot helpers ----------------------------------
def embed_all(encoder_eager: torch.nn.Module, X_cpu: torch.Tensor, device, batch: int = 8192):
    encoder_eager.eval()
    zs = []
    with torch.no_grad():
        for i in range(0, X_cpu.shape[0], batch):
            xb = X_cpu[i:i+batch].to(device, non_blocking=False)
            zb = encoder_eager(xb).cpu()
            zs.append(zb)
    return torch.cat(zs, dim=0)

def _as_np(x):
    if isinstance(x, np.ndarray):
        return x
    try:
        import torch
        if torch.is_tensor(x):
            return x.detach().cpu().numpy()
    except Exception:
        pass
    return np.asarray(x)


def plot_latent(Z, color, title: str, out_path: str, s=1):
    Z = _as_np(Z)
    color = _as_np(color)
    Path(os.path.dirname(out_path)).mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(5.2, 4.3))
    if Z.shape[1] >= 2:
        plt.scatter(Z[:, 0], Z[:, 1], c=color, s=s, alpha=0.65)
        plt.xlabel("z1"); plt.ylabel("z2")
    else:
        plt.scatter(np.arange(Z.shape[0]), Z[:, 0], c=color, s=s, alpha=0.65)
        plt.xlabel("index"); plt.ylabel("z1")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def hist_latent_component(Z, idx: int, title: str, out_path: str, bins: int = 80):
    Z = _as_np(Z)
    Path(os.path.dirname(out_path)).mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(5.0, 3.8))
    plt.hist(Z[:, idx], bins=bins, density=True)
    plt.xlabel(f"z{idx+1}")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def plot_train_val_curves(train_vals, val_vals, ylabel: str, title: str, out_path: str):
    Path(os.path.dirname(out_path)).mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(6.4, 4.4))
    plt.plot(train_vals, label=f"train {ylabel}")
    plt.plot(val_vals, label=f"val {ylabel}")
    plt.xlabel("epoch")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()

def plot_all_losses(train_mse, val_mse, train_dcae, val_dcae, train_total, val_total, out_path: str):
    Path(os.path.dirname(out_path)).mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(7.2, 4.8))
    plt.plot(train_total, label="train total")
    plt.plot(val_total,   label="val total")
    plt.plot(train_mse,   label="train mse")
    plt.plot(val_mse,     label="val mse")
    plt.plot(train_dcae,  label="train dcae")
    plt.plot(val_dcae,    label="val dcae")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.title("Train vs Val: total / mse / dcae")
    plt.legend(ncol=2)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()

# ------------------ NumPy-callable snapshot wrappers ------------------
from copy import deepcopy  # (if not already imported at top)

def _to_float32_np(a):
    x = np.asarray(a)
    if x.dtype != np.float32:
        x = x.astype(np.float32, copy=False)
    return x

def make_np_encoder_fn(enc_snapshot: nn.Module, batch: int = 8192):
    enc_cpu = deepcopy(enc_snapshot).to('cpu').eval()
    def f(X_np: np.ndarray):
        X = torch.from_numpy(_to_float32_np(X_np))
        outs = []
        with torch.no_grad():
            for i in range(0, X.shape[0], batch):
                outs.append(enc_cpu(X[i:i+batch]).cpu().numpy())
        return np.concatenate(outs, axis=0)
    return f

def make_np_decoder_fn(dec_snapshot: nn.Module, batch: int = 8192):
    dec_cpu = deepcopy(dec_snapshot).to('cpu').eval()
    def f(Z_np: np.ndarray):
        Z = torch.from_numpy(_to_float32_np(Z_np))
        outs = []
        with torch.no_grad():
            for i in range(0, Z.shape[0], batch):
                outs.append(dec_cpu(Z[i:i+batch]).cpu().numpy())
        return np.concatenate(outs, axis=0)
    return f

def make_np_autoencoder_fn(enc_snapshot: nn.Module, dec_snapshot: nn.Module, batch: int = 8192):
    enc_cpu = deepcopy(enc_snapshot).to('cpu').eval()
    dec_cpu = deepcopy(dec_snapshot).to('cpu').eval()
    def f(X_np: np.ndarray):
        X = torch.from_numpy(_to_float32_np(X_np))
        outs = []
        with torch.no_grad():
            for i in range(0, X.shape[0], batch):
                Z = enc_cpu(X[i:i+batch])
                outs.append(dec_cpu(Z).cpu().numpy())
        return np.concatenate(outs, axis=0)
    return f

def parse_args(arg_list=None):
    import os, sys, argparse

    p = argparse.ArgumentParser(description="AE splitting demo (CPU)")

    # core problem sizes / data
    p.add_argument("--m", type=int, default=15, help="ambient dim (R^m)")
    p.add_argument("--ell", type=int, default=10, help="intrinsic ball dim ℓ (< m) for signal subspace")
    p.add_argument("--k", type=int, default=2, help="latent dim")
    p.add_argument("--N", type=int, default=12000, help="dataset size")
    p.add_argument("--noise-sigma", type=float, default=0.0, help="Gaussian noise on inputs")

    # training schedule
    p.add_argument("--batch-size", type=int, default=4096, help="minibatch size")
    p.add_argument("--epochs-mse", type=int, default=25, help="epochs for Phase-1 (MSE only)")
    p.add_argument("--epochs-dcae", type=int, default=25, help="epochs for Phase-2 (MSE + DCAE)")
    p.add_argument("--lambda-dcae", type=float, default=3e-4, help="DCAE penalty weight (0 disables Phase-2)")
    p.add_argument("--dcae-probes", type=int, default=1, help="Hutchinson probes per batch (>=1)")

    # splits / early stopping
    p.add_argument("--val-frac", type=float, default=0.20, help="validation fraction in (0,1)")
    p.add_argument("--monitor", type=str, default="auto",
                   choices=["auto", "mse", "total"],
                   help="Phase-2 early-stop metric: 'total' (default via 'auto') or 'mse'")
    # global early-stop master switch (default ON), with opt-out flag
    p.add_argument("--early-stop", dest="early_stop", action="store_true",
                   help="enable early stopping (default: on)")
    p.add_argument("--no-early-stop", dest="early_stop", action="store_false",
                   help="disable all early stopping")
    p.set_defaults(early_stop=True)

    # per-phase toggles (default ON), with opt-out flags
    p.add_argument("--early-stop-p1", dest="early_stop_p1", action="store_true",
                   help="enable early stopping in Phase-1 (val MSE)")
    p.add_argument("--no-early-stop-p1", dest="early_stop_p1", action="store_false",
                   help="disable early stopping in Phase-1")
    p.set_defaults(early_stop_p1=True)

    p.add_argument("--early-stop-p2", dest="early_stop_p2", action="store_true",
                   help="enable early stopping in Phase-2 (monitor per --monitor)")
    p.add_argument("--no-early-stop-p2", dest="early_stop_p2", action="store_false",
                   help="disable early stopping in Phase-2")
    p.set_defaults(early_stop_p2=True)

    p.add_argument("--patience", type=int, default=10,
                   help="epochs without improvement before stopping")
    p.add_argument("--min-delta", type=float, default=1e-4,
                   help="minimum improvement to reset patience")

    # system / reproducibility
    p.add_argument("--seed", type=int, default=42, help="PRNG seed")
    p.add_argument("--outdir", type=str, default="demo_plots", help="output directory")

    # CPU threading and dataloader
    p.add_argument("--threads", type=int, default=min(56, os.cpu_count() or 8), help="intra-op threads")
    p.add_argument("--interop-threads", type=int, default=8, help="interop threads")
    p.add_argument("--num-workers", type=int, default=8, help="DataLoader workers (CPU)")
    p.add_argument("--prefetch-factor", type=int, default=4, help="DataLoader prefetch factor")

    # optional torch.compile for Phase-1
    p.add_argument("--compile", action="store_true", help="try torch.compile (CPU) for Phase-1")

    # parse (support programmatic call via arg_list)
    args = p.parse_args(arg_list if arg_list is not None else sys.argv[1:])

    # ----------- safety guards / normalization -----------
    args.dcae_probes    = max(1, int(args.dcae_probes))
    args.batch_size     = max(8, int(args.batch_size))
    args.num_workers    = max(0, int(args.num_workers))
    args.prefetch_factor = max(2, int(args.prefetch_factor))

    # val_frac must be in (0,1)
    if not (0.0 < float(args.val_frac) < 1.0):
        raise ValueError(f"--val-frac must be in (0,1); got {args.val_frac}")

    # ell sanity (let the data generator assert more strictly)
    if args.ell >= args.m:
        # keep but warn; your generator will fallback or assert depending on implementation
        print(f"[warn] --ell ({args.ell}) >= --m ({args.m}); "
              f"generator may ignore ell or raise. Consider setting ell < m.")

    return args

# ----------------------------------- Main -------------------------------------
def main(arg_list=None):
    import copy

    # -------------------- args & CPU setup --------------------
    args = parse_args(arg_list)
    configure_cpu(args)

    device = torch.device("cpu")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # -------------------- data & split ------------------------
    # use ell + val_frac
    X_cpu = generate_dataset(
        args.N, args.m,
        noise_sigma=args.noise_sigma,
        ell=args.ell,
        rotate=False
    )
    N = X_cpu.shape[0]
    val_frac = float(args.val_frac)
    assert 0.0 < val_frac < 1.0, "--val-frac must be in (0,1)"
    n_val = max(1, int(val_frac * N))
    perm = torch.randperm(N)
    val_idx = perm[:n_val]
    tr_idx  = perm[n_val:]

    X_tr  = X_cpu[tr_idx]
    X_val = X_cpu[val_idx]

    def _mk_loader(Xt):
        return DataLoader(
            TensorDataset(Xt),
            batch_size=args.batch_size,
            shuffle=True,
            drop_last=False,
            num_workers=args.num_workers,
            pin_memory=False,  # CPU-only
            persistent_workers=(args.num_workers > 0),
            prefetch_factor=(args.prefetch_factor if args.num_workers > 0 else None),
        )

    dl_tr  = _mk_loader(X_tr)
    dl_val = _mk_loader(X_val)

    # -------------------- models ------------------------------
    enc_eager = Encoder(args.m, args.k).to(device)
    dec_eager = Decoder(args.k, args.m).to(device)

    # compile only for Phase-1 training (no higher-order autograd)
    enc_mse, dec_mse = maybe_compile_for_training(
        enc_eager, dec_eager, enable_compile=bool(args.compile)
    )

    optE = torch.optim.AdamW(enc_eager.parameters(), lr=1e-3)
    optD = torch.optim.AdamW(dec_eager.parameters(), lr=1e-3)

    # -------------------- eval helper -------------------------
    def eval_epoch(enc, dec, loader, lam_dcae: float = 0.0, use_dcae: bool = False, dcae_probes: int = 1):
        """
        Returns dict: {"loss", "mse", "dcae"} over loader.
        Uses first-order autograd w.r.t. inputs if DCAE is on, but no optimizer/backward().
        """
        enc.eval(); dec.eval()
        mse_sum = 0.0
        dcae_sum = 0.0
        n = 0
        for (xb,) in loader:
            xb = xb.to(device, non_blocking=False)
            # MSE part
            with torch.no_grad():
                z = enc(xb)
                xhat = dec(z)
                mse = F.mse_loss(xhat, xb, reduction="mean")

            # DCAE part (no create_graph needed for eval)
            if use_dcae and lam_dcae > 0.0:
                xb_req = xb.detach().clone().requires_grad_(True)
                z2 = enc(xb_req)
                B, _ = z2.shape
                pen = xb_req.new_tensor(0.0)
                for _ in range(max(1, int(dcae_probes))):
                    v = torch.empty_like(z2).bernoulli_(0.5).mul_(2.0).sub_(1.0)
                    vz = (z2 * v).sum()
                    gx = torch.autograd.grad(
                        vz, xb_req,
                        create_graph=False, retain_graph=False, only_inputs=True
                    )[0]
                    pen = pen + (gx.pow(2).sum() / B)
                pen = pen / max(1, int(dcae_probes))
            else:
                pen = xb.new_tensor(0.0)

            bsz = xb.shape[0]
            mse_sum  += float(mse) * bsz
            dcae_sum += float(pen) * bsz
            n += bsz

        mse_mean  = mse_sum / max(1, n)
        dcae_mean = dcae_sum / max(1, n)
        return {"loss": (mse_mean + lam_dcae * dcae_mean), "mse": mse_mean, "dcae": dcae_mean}

    # -------------------- plotting helper ---------------------
    def plot_curves(xs, train_vals, val_vals, title, ylabel, out_path):
        Path(os.path.dirname(out_path)).mkdir(parents=True, exist_ok=True)
        plt.figure(figsize=(6.0, 4.2))
        plt.plot(xs, train_vals, label="train")
        plt.plot(xs, val_vals,   label="val")
        plt.xlabel("epoch")
        plt.ylabel(ylabel)
        plt.title(title)
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_path, dpi=300)
        plt.close()

    color = X_cpu[:, 0].numpy()
    color = _as_np(color)
    # ==================== Phase 1: MSE-only ====================
    t0 = time.time()
    print(f"[Phase 1] MSE pretraining for {args.epochs_mse} epochs")

    p1_train_mse, p1_val_mse = [], []

    # early-stopping trackers (Phase-1 monitors val MSE)
    best_val_mse = float("inf")
    best_epoch_p1 = 0
    patience_p1 = 0
    mse_best_state = (copy.deepcopy(enc_eager.state_dict()),
                      copy.deepcopy(dec_eager.state_dict()))

    for ep in range(1, args.epochs_mse + 1):
        # train step (compiled wrappers ok here)
        tr_stats = train_epoch(enc_mse, dec_mse, optE, optD, dl_tr, device,
                               lam_dcae=0.0, use_dcae=False)
        # val step (no DCAE in phase-1)
        va_stats = eval_epoch(enc_eager, dec_eager, dl_val, lam_dcae=0.0, use_dcae=False)

        p1_train_mse.append(tr_stats["mse"])
        p1_val_mse.append(va_stats["mse"])

        print(f"[MSE  {ep:03d}] train_mse={tr_stats['mse']:.6f} | val_mse={va_stats['mse']:.6f}")

        # early stopping on val MSE
        improved = (best_val_mse - va_stats['mse']) > args.min_delta
        if improved:
            best_val_mse = va_stats['mse']
            best_epoch_p1 = ep
            patience_p1 = 0
            mse_best_state = (copy.deepcopy(enc_eager.state_dict()),
                              copy.deepcopy(dec_eager.state_dict()))
            if args.early_stop or getattr(args, "early_stop_p1", True):
                print(f"[P1] ✨ new best val_mse={best_val_mse:.6f} at epoch {ep}")
        else:
            patience_p1 += 1
            if (args.early_stop or getattr(args, "early_stop_p1", True)) and patience_p1 >= args.patience:
                print(f"[P1] Early stop at epoch {ep} "
                      f"(no val MSE improvement > {args.min_delta:g} for {args.patience} epochs; "
                      f"best={best_val_mse:.6f} @ {best_epoch_p1}).")
                break

    print(f"[Phase 1] done in {time.time() - t0:.1f}s")

    # restore best Phase-1 before exporting/plotting
    enc_eager.load_state_dict(mse_best_state[0])
    dec_eager.load_state_dict(mse_best_state[1])

    # Phase-1 curves
    xs1 = np.arange(1, len(p1_train_mse) + 1)
    plot_curves(xs1, p1_train_mse, p1_val_mse,
                title=f"Phase-1 MSE (m={args.m}, k={args.k})",
                ylabel="MSE",
                out_path=os.path.join(args.outdir, "phase1_mse_train_val.png"))

    with torch.no_grad():
        Z_mse = embed_all(enc_eager, X_cpu, device)
    plot_latent(
        Z_mse, color, s=1,
        title=f"Latent (k={args.k}) after MSE-only, m={args.m}",
        out_path=os.path.join(args.outdir, f"latent_mse_only_m{args.m}.png"),
    )
    hist_latent_component(
        Z_mse, 0,
        title=f"Histogram of z1 (MSE only) — m={args.m}",
        out_path=os.path.join(args.outdir, f"hist_z1_mse_m{args.m}.png")
    )

    # ---------- Export MSE snapshot as NumPy-callable + arrays ----------
    X_input_np  = X_cpu.numpy().astype(np.float32, copy=False)
    enc_mse_fn  = make_np_encoder_fn(enc_eager)
    dec_mse_fn  = make_np_decoder_fn(dec_eager)
    ae_mse_fn   = make_np_autoencoder_fn(enc_eager, dec_eager)
    Z_mse_np    = enc_mse_fn(X_input_np)
    Xhat_mse_np = ae_mse_fn(X_input_np)

    # ==================== Phase 2: MSE + DCAE ==================
    use_dcae = (args.lambda_dcae is not None) and (args.lambda_dcae > 0.0) and (args.epochs_dcae > 0)
    if not use_dcae:
        print("[INFO] lambda_dcae=0 or epochs_dcae=0 → DCAE is bypassed (MSE-only).")
        # Final = MSE snapshot
        enc_final_fn, dec_final_fn, ae_final_fn = enc_mse_fn, dec_mse_fn, ae_mse_fn
        Z_final_np, Xhat_final_np = Z_mse_np, Xhat_mse_np

        result = {
            "X_input": X_input_np,
            "mse": {
                "encoder_fn": enc_mse_fn,
                "decoder_fn": dec_mse_fn,
                "autoencoder_fn": ae_mse_fn,
                "Z": Z_mse_np,
                "Xhat": Xhat_mse_np,
            },
            "final": {
                "encoder_fn": enc_final_fn,
                "decoder_fn": dec_final_fn,
                "autoencoder_fn": ae_final_fn,
                "Z": Z_final_np,
                "Xhat": Xhat_final_np,
            },
        }
        print(f"[DONE] Figures saved to: {args.outdir}")
        return result

    print(f"[Phase 2] MSE + DCAE for {args.epochs_dcae} epochs (λ={args.lambda_dcae}, probes={args.dcae_probes})")
    print("[Phase 2] switching to EAGER mode to support higher-order autograd.")

    p2_train_total, p2_train_mse, p2_train_dcae = [], [], []
    p2_val_total,   p2_val_mse,   p2_val_dcae   = [], [], []

    # early-stopping trackers (Phase-2 monitors TOTAL by default)
    monitor_total = (args.monitor in ("auto", "total"))
    best_val_metric = float("inf")
    best_epoch_p2 = 0
    patience_p2 = 0
    final_best_state = (copy.deepcopy(enc_eager.state_dict()),
                        copy.deepcopy(dec_eager.state_dict()))

    t1 = time.time()
    for ep in range(1, args.epochs_dcae + 1):
        # train (eager only due to second-order autograd)
        tr_stats = train_epoch(
            enc_eager, dec_eager, optE, optD, dl_tr, device,
            lam_dcae=args.lambda_dcae, use_dcae=True, dcae_probes=args.dcae_probes
        )
        # val
        va_stats = eval_epoch(
            enc_eager, dec_eager, dl_val,
            lam_dcae=args.lambda_dcae, use_dcae=True, dcae_probes=args.dcae_probes
        )

        p2_train_total.append(tr_stats["loss"]); p2_val_total.append(va_stats["loss"])
        p2_train_mse.append(tr_stats["mse"]);   p2_val_mse.append(va_stats["mse"])
        p2_train_dcae.append(tr_stats["dcae"]); p2_val_dcae.append(va_stats["dcae"])

        print(
            f"[DCAE {ep:03d}] "
            f"train: total={tr_stats['loss']:.6f} mse={tr_stats['mse']:.6f} dcae={tr_stats['dcae']:.6f} | "
            f"val: total={va_stats['loss']:.6f} mse={va_stats['mse']:.6f} dcae={va_stats['dcae']:.6f}"
        )

        # early stopping on chosen val metric
        val_metric = va_stats["loss"] if monitor_total else va_stats["mse"]
        improved = (best_val_metric - val_metric) > args.min_delta
        if improved:
            best_val_metric = val_metric
            best_epoch_p2 = ep
            patience_p2 = 0
            final_best_state = (copy.deepcopy(enc_eager.state_dict()),
                                copy.deepcopy(dec_eager.state_dict()))
            if args.early_stop or getattr(args, "early_stop_p2", True):
                name = "total" if monitor_total else "mse"
                print(f"[P2] ✨ new best val_{name}={best_val_metric:.6f} at epoch {ep}")
        else:
            patience_p2 += 1
            if (args.early_stop or getattr(args, "early_stop_p2", True)) and patience_p2 >= args.patience:
                name = "total" if monitor_total else "mse"
                print(f"[P2] Early stop at epoch {ep} "
                      f"(no val {name} improvement > {args.min_delta:g} for {args.patience} epochs; "
                      f"best={best_val_metric:.6f} @ {best_epoch_p2}).")
                break

    print(f"[Phase 2] done in {time.time() - t1:.1f}s")

    # restore best Phase-2 before exporting/plotting
    enc_eager.load_state_dict(final_best_state[0])
    dec_eager.load_state_dict(final_best_state[1])

    # phase-2 curves (overlapping train/val)
    xs2 = np.arange(1, len(p2_train_total) + 1)
    plot_curves(xs2, p2_train_total, p2_val_total,
                title="Phase-2 Total Loss (MSE + λ·DCAE)",
                ylabel="loss",
                out_path=os.path.join(args.outdir, "phase2_total_train_val.png"))
    plot_curves(xs2, p2_train_mse, p2_val_mse,
                title="Phase-2 MSE",
                ylabel="MSE",
                out_path=os.path.join(args.outdir, "phase2_mse_train_val.png"))
    plot_curves(xs2, p2_train_dcae, p2_val_dcae,
                title="Phase-2 DCAE penalty",
                ylabel="DCAE",
                out_path=os.path.join(args.outdir, "phase2_dcae_train_val.png"))

    # Latent viz after Phase-2
    with torch.no_grad():
        Z_dcae = embed_all(enc_eager, X_cpu, device)
    plot_latent(
        Z_dcae, color, s=1,
        title=f"Latent (k={args.k}) after MSE+DCAE, m={args.m}",
        out_path=os.path.join(args.outdir, f"latent_mse_dcae_m{args.m}.png"),
    )
    hist_latent_component(
        Z_dcae, 0,
        title=f"Histogram of z1 (MSE + DCAE) — m={args.m}",
        out_path=os.path.join(args.outdir, f"hist_z1_dcae_m{args.m}.png")
    )

    # ---------- Export FINAL (NumPy-callable) ----------
    enc_final_fn  = make_np_encoder_fn(enc_eager)
    dec_final_fn  = make_np_decoder_fn(dec_eager)
    ae_final_fn   = make_np_autoencoder_fn(enc_eager, dec_eager)
    Z_final_np    = enc_final_fn(X_input_np)
    Xhat_final_np = ae_final_fn(X_input_np)

    result = {
        "X_input": X_input_np,
        "mse": {
            "encoder_fn": enc_mse_fn,
            "decoder_fn": dec_mse_fn,
            "autoencoder_fn": ae_mse_fn,
            "Z": Z_mse_np,
            "Xhat": Xhat_mse_np,
        },
        "final": {
            "encoder_fn": enc_final_fn,
            "decoder_fn": dec_final_fn,
            "autoencoder_fn": ae_final_fn,
            "Z": Z_final_np,
            "Xhat": Xhat_final_np,
        },
    }
    print(f"[DONE] Figures saved to: {args.outdir}")
    return result

# ------------------------------- Entrypoint -----------------------------------
if __name__ == "__main__":
    main()



