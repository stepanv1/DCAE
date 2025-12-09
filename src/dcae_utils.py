#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AE splitting demo (CPU-only, multi-core). Shows MSE-only vs. MSE+DCAE.

Command-line usage
------------------
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

New programmatic API (three main functions)
------------------------------------------
1) Dataset generation (with train/val split):

    from ae_split_demo import generate_data

    data = generate_data(
        N=12000,
        m=15,
        ell=10,
        noise_sigma=0.0,
        rotate=False,
        val_frac=0.2,
        seed=42,
    )
    # data is a dict with keys: "X_cpu", "X_tr", "X_val"

2) Training (Phase-1 MSE, optional Phase-2 MSE+DCAE):

    from ae_split_demo import run_training

    result = run_training(
        **data,          # X_cpu, X_tr, X_val
        m=15,
        k=2,
        batch_size=4096,
        epochs_mse=20,
        epochs_dcae=20,
        lambda_dcae=3e-4,
        dcae_probes=1,
        monitor="auto",
        early_stop=True,
        early_stop_p1=True,
        early_stop_p2=True,
        patience=10,
        min_delta=1e-4,
        threads=56,
        interop_threads=8,
        num_workers=8,
        prefetch_factor=4,
        compile=True,
        seed=42,
    )

    # result["snapshots"]["mse"]["encoder_fn"](X_np)   → Z (NumPy)
    # result["snapshots"]["final"]["encoder_fn"](X_np) → Z_final
    # result["history"]["phase1"]["train_mse"]         → list of per-epoch losses, etc.

3) Plotting (uses training result; can be called independently):

    from ae_split_demo import plot_results

    plot_results(
        result,
        outdir="demo_plots_custom",
        color=None,          # default: first coordinate of X
        scatter_size=1.0,
        hist_component=0,
        hist_bins=80,
        prefix="exp1_",
    )
"""

import os
import sys
import time
import argparse
from pathlib import Path
from copy import deepcopy
from typing import Optional, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from typing import Optional
import math

# --------------------------- CPU configuration --------------------------------
def configure_cpu(args_or_threads, interop_threads: Optional[int] = None):
    """
    Configure PyTorch / BLAS thread counts.

    Can be called either as:
        configure_cpu(args)                    # args.threads, args.interop_threads
    or:
        configure_cpu(threads, interop_threads)
    """
    if hasattr(args_or_threads, "threads"):
        threads = int(args_or_threads.threads)
        interop = int(getattr(args_or_threads, "interop_threads", threads))
    else:
        threads = int(args_or_threads)
        if interop_threads is None:
            raise ValueError(
                "When calling configure_cpu with a raw thread count, "
                "`interop_threads` must also be provided."
            )
        interop = int(interop_threads)

    os.environ["OMP_NUM_THREADS"] = str(threads)
    os.environ["MKL_NUM_THREADS"] = str(threads)
    os.environ["OPENBLAS_NUM_THREADS"] = str(threads)
    os.environ["NUMEXPR_NUM_THREADS"] = str(threads)
    os.environ["VECLIB_MAXIMUM_THREADS"] = str(threads)
    try:
        torch.set_num_threads(threads)
    except Exception:
        pass
    try:
        torch.set_num_interop_threads(interop)
    except Exception:
        pass


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
    """
    assert 1 <= ell < m, "ell must satisfy 1 ≤ ell < m"

    device = device or torch.device("cpu")

    # --- Uniform on ℓ-ball via direction * radius ---
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
        Q, _ = torch.linalg.qr(torch.randn(m, m, device=device, dtype=dtype))
        if torch.det(Q) < 0:
            Q[:, 0] = -Q[:, 0]
        X = X @ Q.T

    return X


def sample_unit_ball(N: int, m: int) -> torch.Tensor:
    """Uniform in unit m-ball via Gaussian direction + U^(1/m) radius."""
    X = torch.randn(N, m)
    X = X / X.norm(dim=1, keepdim=True).clamp_min(1e-12)
    r = torch.rand(N).pow(1.0 / m)
    return X * r.unsqueeze(1)


def generate_dataset(
    N: int,
    m: int,
    noise_sigma: float = 0.0,
    ell: Optional[int] = None,
    rotate: bool = False,
) -> torch.Tensor:
    """
    Low-level generator for an N×m cloud.

    If ell is None or ell >= m: sample full m-ball (optionally with isotropic noise).
    Otherwise: sample an ℓ-ball embedded in R^m with Gaussian noise in the
    remaining coordinates, optionally rotated.
    """
    if ell is None or ell >= m:
        X = sample_unit_ball(N, m)
        if noise_sigma > 0.0:
            X = X + (noise_sigma * torch.randn(N, m))
        return X
    else:
        return sample_l_ball_with_noise(
            N, m, ell, noise_sigma=noise_sigma, rotate=rotate
        )


def train_val_split(X: torch.Tensor, val_frac: float = 0.2, seed: int = 42):
    """Random train/val split (CPU tensor in, two CPU tensors out)."""
    assert 0.0 < val_frac < 1.0, "--val-frac must be in (0,1)"
    N = X.shape[0]
    n_val = int(round(N * val_frac))
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(N, generator=g)
    idx_val = perm[:n_val]
    idx_tr = perm[n_val:]
    return X[idx_tr].contiguous(), X[idx_val].contiguous()


def make_loader(
    X_cpu: torch.Tensor,
    batch_size: int,
    num_workers: int,
    prefetch_factor: int,
    *,
    shuffle: bool = True,
) -> DataLoader:
    """Helper to build a DataLoader for a single-tensor dataset."""
    ds = TensorDataset(X_cpu)  # yields tuples (X,)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=False,  # CPU-only
        persistent_workers=(num_workers > 0),
        prefetch_factor=(prefetch_factor if num_workers > 0 else None),
    )


# --------------------------------- Models -------------------------------------
class MLP(nn.Module):
    def __init__(self, dims, act="elu", last_linear=False):
        super().__init__()
        layers = []
        Act = nn.ELU if act == "elu" else nn.ReLU
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2 or not last_linear:
                layers.append(Act())
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class Encoder(nn.Module):
    def __init__(self, m, k):
        super().__init__()
        self.core = MLP([m, 512, 256, 128, 64, k], act="elu", last_linear=True)

    def forward(self, x):
        return self.core(x)

class AnalyticEncoder(nn.Module):
    """
    Encoder with explicit 2-layer ELU + linear head, structured so that
    we can write an analytic DCAE penalty (no autograd.grad calls).

        x -> ELU(fc1) -> ELU(fc2) -> z = fc3
    """
    def __init__(self, m: int, k: int):
        super().__init__()
        h1 = 3 * m
        h2 = 2 * m
        self.fc1 = nn.Linear(m, h1)
        self.fc2 = nn.Linear(h1, h2)
        self.fc3 = nn.Linear(h2, k)
        self.act = nn.ELU()

    def forward(self, x: torch.Tensor):
        h1 = self.act(self.fc1(x))   # (B, h1)
        h2 = self.act(self.fc2(h1))  # (B, h2)
        z  = self.fc3(h2)            # (B, k), linear head
        # For analytic DCAE we need all three
        return z, h1, h2


class Decoder(nn.Module):
    def __init__(self, k, m):
        super().__init__()
        self.core = MLP([k, 64, 128, 256, 512, m], act="elu", last_linear=False)

    def forward(self, z):
        return self.core(z)


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
            vz,
            xb,
            create_graph=create_graph,
            retain_graph=retain_graph,
            only_inputs=True,
        )[0]
        pen = pen + (gx.pow(2).sum() / B)
    return pen / probes

def analytic_dcae_penalty(
    z: torch.Tensor,
    h1: torch.Tensor,
    h2: torch.Tensor,
    enc: "AnalyticEncoder",
    *,
    use_radial_potential: bool = True,
) -> torch.Tensor:
    """
    Analytic contractive penalty approximating ||∂z/∂x||_F^2 for AnalyticEncoder.

    We propagate diagonal derivatives through the weights with batched einsums,
    mimicking the TF DCAE_loss structure but without calling autograd.grad.

    Returns the *mean* penalty over the batch.
    """
    B, k = z.shape
    device = z.device
    dtype = z.dtype

    # Weight matrices in "math" layout:
    #   W1: (d, h1), W2: (h1, h2), W3: (h2, k)
    W1_T = enc.fc1.weight    # (h1, d)
    W2_T = enc.fc2.weight    # (h2, h1)
    W3_T = enc.fc3.weight    # (k, h2)

    W1 = W1_T.t()            # (d, h1)
    W2 = W2_T.t()            # (h1, h2)
    W3 = W3_T.t()            # (h2, k)

    # --- Derivatives of ELU from activations ---
    # ELU(u) = u (u>0), e^u - 1 (u<=0)
    # If h = ELU(u), then:
    #   u>0: h>0, φ'(u)=1
    #   u<=0: h<=0, h = e^u - 1 -> e^u = h+1, so φ'(u)=e^u = h+1
    d1 = torch.where(h1 > 0, torch.ones_like(h1), h1 + 1.0)  # (B, h1_dim)
    d2 = torch.where(h2 > 0, torch.ones_like(h2), h2 + 1.0)  # (B, h2_dim)

    # --- Optional radial potential (same spirit as TF) ---
    if use_radial_potential:
        r2 = (z * z).sum(dim=1)              # (B,)
        pot = (r2 - 1.0) ** 2 + 1.0          # (B,)
    else:
        pot = torch.ones(B, device=device, dtype=dtype)

    # 1) S1 = diag(d2) W3  -> (B, h2, k)
    S1 = d2.unsqueeze(2) * W3.unsqueeze(0)   # (B, h2, k)

    # 2) S2 = W2 @ S1     -> (B, h1, k)
    S2 = torch.einsum("ij,bjk->bik", W2, S1)

    # 3) S3 = diag(d1) S2 -> (B, h1, k)
    S3 = d1.unsqueeze(2) * S2               # (B, h1, k)

    # 4) J = W1 @ S3      -> (B, d, k)
    J = torch.einsum("ij,bjk->bik", W1, S3) # (B, d, k)

    # 5) radial scaling
    J = pot.view(B, 1, 1) * J

    # Frobenius norm squared per sample, then mean
    pen_per_sample = (J * J).sum(dim=(1, 2))  # (B,)
    return pen_per_sample.mean()


# ------------------------------ MMD penalty + annealing -----------------------

def sample_shell(
    npoints: int,
    a: float,
    b: float,
    ndim: int = 3,
    *,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    Sample points uniformly in a spherical shell between radii [a, b] in R^ndim,
    matching the TF sample_shell() logic.
    """
    if device is None:
        device = torch.device("cpu")

    # Random directions
    vec = torch.randn(npoints, ndim, device=device, dtype=dtype)
    vec = vec / vec.norm(dim=1, keepdim=True).clamp_min(1e-12)

    # Radii: r^3 ~ Uniform(a^3, b^3)  =>  r = U(a^3, b^3)^(1/3)
    u = torch.rand(npoints, device=device, dtype=dtype)
    r_inner3 = a ** 3
    r_outer3 = b ** 3
    R = (u * (r_outer3 - r_inner3) + r_inner3).pow(1.0 / 3.0)  # (npoints,)

    return vec * R.unsqueeze(1)


def compute_kernel(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    RBF kernel used in the TF code:
        exp( - mean( (x - y)^2, axis=-1 ) / dim )
    """
    # x: (B1, d), y: (B2, d)
    diff = x.unsqueeze(1) - y.unsqueeze(0)  # (B1, B2, d)
    dim = x.shape[1]
    return torch.exp(-diff.pow(2).mean(dim=2) / float(dim))


def compute_mmd(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    MMD^2(x, y) with the above kernel, same form as TF compute_mmd.
    """
    x_kernel = compute_kernel(x, x)
    y_kernel = compute_kernel(y, y)
    xy_kernel = compute_kernel(x, y)
    return x_kernel.mean() + y_kernel.mean() - 2.0 * xy_kernel.mean()


def make_rff_mmd(
    dim,
    n_features=64,
    sigma=None,
    device=None,
    dtype=torch.float32,
):
    """
    Build a fast approximate MMD^2 function using random Fourier features
    for a Gaussian kernel in R^dim.

    Returned function: mmd_rff(x, y)
        x, y: (B, dim) tensors
        -> scalar tensor (MMD^2 estimate)

    If sigma is None, we choose sigma so that the RFF kernel roughly
    matches your existing compute_kernel(), i.e.
        k(x,y) ≈ exp(-||x-y||^2 / d^2).
    That corresponds to an RBF with sigma = d / sqrt(2).
    """
    if device is None:
        device = torch.device("cpu")

    # ---- coerce dim and n_features to plain Python ints ----
    if isinstance(dim, torch.Tensor):
        dim_int = int(dim.item())
    else:
        dim_int = int(dim)

    if isinstance(n_features, torch.Tensor):
        nf_int = int(n_features.item())
    else:
        nf_int = int(n_features)

    # ---- choose bandwidth ----
    if sigma is None or sigma <= 0.0:
        # match exp(-||x-y||^2 / d^2) ≈ exp(-||x-y||^2 / (2 sigma^2))
        # -> sigma^2 = d^2 / 2  -> sigma = d / sqrt(2)
        sigma = dim_int / math.sqrt(2.0)

    # ---- sample random Fourier features ONCE ----
    # For Gaussian RBF with this sigma, we want ω ~ N(0, I / sigma^2)
    omega = torch.randn(dim_int, nf_int, device=device, dtype=dtype) / float(sigma)

    b = 2.0 * math.pi * torch.rand(nf_int, device=device, dtype=dtype)
    scale = math.sqrt(2.0 / nf_int)

    def mmd_rff(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # x, y: (B, dim_int)
        proj_x = x @ omega          # (B, nf_int)
        proj_y = y @ omega          # (B, nf_int)

        phi_x = scale * torch.cos(proj_x + b)   # (B, nf_int)
        phi_y = scale * torch.cos(proj_y + b)

        mu_x = phi_x.mean(dim=0)               # (nf_int,)
        mu_y = phi_y.mean(dim=0)               # (nf_int,)

        return (mu_x - mu_y).pow(2).sum()

    return mmd_rff

def frange_anneal(
    num_steps: int,
    *,
    ratio: float = 0.9,
    start: float = 0.0,
    end: float = 1.0,
) -> torch.Tensor:
    """
    Simple annealing schedule similar to frange_anneal in TF code.

    For the first (1 - ratio) fraction of steps, the value is `start`.
    Over the remaining steps, it ramps linearly from `start` to `end`.
    """
    num_steps = int(num_steps)
    if num_steps <= 0:
        return torch.zeros(0, dtype=torch.float32)

    ratio = float(ratio)
    ratio = max(0.0, min(1.0, ratio))

    hold = int(round(num_steps * (1.0 - ratio)))
    hold = min(hold, num_steps)
    ramp = num_steps - hold

    vals = torch.empty(num_steps, dtype=torch.float32)
    if hold > 0:
        vals[:hold] = float(start)
    if ramp > 0:
        vals[hold:] = torch.linspace(float(start), float(end), steps=ramp)
    return vals



# -------------------------- Training / eval epochs ----------------------------
def train_epoch(
    enc,
    dec,
    optE,
    optD,
    loader,
    device,
    lam_dcae: float = 0.0,
    use_dcae: bool = False,
    dcae_probes: int = 1,
    *,
    use_mmd: bool = False,
    mmd_weight: float = 1.0,
    mmd_rff_fn=None,
    mmd_samples: int = 128,
    mse_weight_scalar: float | None = None,
    mmd_shell_a: float = 0.99,
    mmd_shell_b: float = 1.01,
    analytic_dcae: bool = True,
):
    """
    One training epoch.

    If mse_weight_scalar is None and use_mmd=False:
        loss = MSE + lam_dcae * DCAE   (original behaviour).

    Otherwise (MMD + annealing enabled):
        loss = coeffMSE * (1 - w + 0.1) * MSE
             + 0.5 * (w + 1) * mmd_weight * MMD
             + (2 * w + 0.1) * lam_dcae * DCAE
    """
    enc.train()
    dec.train()
    mse_sum = 0.0
    dcae_sum = 0.0
    mmd_sum = 0.0
    loss_sum = 0.0
    n = 0

    for (xb,) in loader:
        xb = xb.to(device, non_blocking=False)

        optE.zero_grad(set_to_none=True)
        optD.zero_grad(set_to_none=True)

        # Forward encoder (may or may not return intermediates) and decoder
        enc_out = enc(xb)
        if isinstance(enc_out, tuple):
            z, h1, h2 = enc_out
        else:
            z = enc_out
            h1 = h2 = None

        xhat = dec(z)
        mse = F.mse_loss(xhat, xb, reduction="mean")

        # DCAE penalty
        if use_dcae and lam_dcae > 0.0:
            if analytic_dcae:
                dcae_pen = analytic_dcae_penalty(
                    z, h1, h2, enc, use_radial_potential=True
                )
            else:
                dcae_pen = dcae_contractive_penalty_hutchinson_from_z(
                    z, xb, probes=dcae_probes
                )
        else:
            dcae_pen = xb.new_tensor(0.0)

        # ----------------- MMD term (RFF, linear-time) -----------------
        if use_mmd and (mmd_weight > 0.0) and (mmd_rff_fn is not None):
            B = z.shape[0]
            # Subsample batch for MMD if desired
            mB = min(B, int(mmd_samples))
            idx = torch.randperm(B, device=z.device)[:mB]
            z_sub = z[idx]

            prior_samples = sample_shell(
                mB,
                mmd_shell_a,
                mmd_shell_b,
                ndim=z_sub.shape[1],
                device=z_sub.device,
                dtype=z_sub.dtype,
            )
            mmd_raw = mmd_rff_fn(prior_samples, z_sub)
        else:
            mmd_raw = xb.new_tensor(0.0)

        # Combine loss
        if mse_weight_scalar is None and not use_mmd:
            # original behaviour
            loss = mse + lam_dcae * dcae_pen
        else:
            w = float(mse_weight_scalar) if mse_weight_scalar is not None else 0.0
            coeffMSE = 1.0
            dcae_term = lam_dcae * dcae_pen
            mmd_term = mmd_weight * mmd_raw
            loss = (
                coeffMSE * (1.0 - w + 0.1) * mse
                + 0.5 * (w + 1.0) * mmd_term
                + (2.0 * w + 0.1) * dcae_term
            )

        loss.backward()
        optE.step()
        optD.step()

        bsz = xb.shape[0]
        mse_sum += float(mse.detach()) * bsz
        dcae_sum += float(dcae_pen.detach()) * bsz
        mmd_sum += float(mmd_raw.detach()) * bsz
        loss_sum += float(loss.detach()) * bsz
        n += bsz

    n = max(1, n)
    return {
        "loss": loss_sum / n,
        "mse": mse_sum / n,
        "dcae": dcae_sum / n,
        "mmd": mmd_sum / n,
    }

def eval_epoch(
    enc,
    dec,
    loader,
    device,
    lam_dcae: float = 0.0,
    use_dcae: bool = False,
    dcae_probes: int = 1,
    *,
    use_mmd: bool = False,
    mmd_weight: float = 1.0,
    mmd_rff_fn=None,
    mmd_samples: int = 128,
    mse_weight_scalar: Optional[float] = None,
    mmd_shell_a: float = 0.99,
    mmd_shell_b: float = 1.01,
    analytic_dcae: bool = False,
):
    """
    Validation epoch.

    Uses the same combined loss formula as train_epoch, but:
      - no gradients through MSE/MMD,
      - DCAE is either analytic (no autograd.grad) or Hutchinson
        with create_graph=False (for metric only).
    """
    enc.eval()
    dec.eval()
    mse_sum = 0.0
    dcae_sum = 0.0
    mmd_sum = 0.0
    loss_sum = 0.0
    n = 0

    for (xb,) in loader:
        xb = xb.to(device, non_blocking=False)
        B = xb.shape[0]

        # ---------- forward pass (no grads) ----------
        with torch.no_grad():
            enc_out = enc(xb)
            if isinstance(enc_out, tuple):
                # AnalyticEncoder: (z, h1, h2)
                z, h1, h2 = enc_out
            else:
                # Plain Encoder: just z
                z = enc_out
                h1 = h2 = None

            xhat = dec(z)
            mse = F.mse_loss(xhat, xb, reduction="mean")

        # ---------- DCAE penalty ----------
        if use_dcae and lam_dcae > 0.0:
            if analytic_dcae:
                # Analytic penalty: pure einsum, no extra autograd passes
                with torch.no_grad():
                    dcae_pen = analytic_dcae_penalty(
                        z.detach(), h1.detach(), h2.detach(), enc,
                        use_radial_potential=True,
                    )
            else:
                # Hutchinson estimator (metric only; no higher-order graph)
                n_probes = max(1, int(dcae_probes))
                pen = xb.new_tensor(0.0)

                for _ in range(n_probes):
                    xb_req = xb.detach().clone().requires_grad_(True)
                    z2 = enc(xb_req)
                    if isinstance(z2, tuple):
                        z2 = z2[0]
                    v = torch.empty_like(z2).bernoulli_(0.5).mul_(2.0).sub_(1.0)
                    vz = (z2 * v).sum()
                    gx = torch.autograd.grad(
                        vz,
                        xb_req,
                        create_graph=False,
                        retain_graph=False,
                        only_inputs=True,
                    )[0]
                    pen = pen + gx.pow(2).sum() / B

                dcae_pen = pen / n_probes
        else:
            dcae_pen = xb.new_tensor(0.0)

        # ---------- MMD penalty (RFF if available) ----------
        if use_mmd and mmd_weight > 0.0:
            with torch.no_grad():
                if (mmd_rff_fn is not None) and (mmd_samples is not None):
                    mB = min(B, int(mmd_samples))
                    idx = torch.randperm(B, device=z.device)[:mB]
                    z_sub = z.detach()[idx]

                    prior_samples = sample_shell(
                        mB,
                        mmd_shell_a,
                        mmd_shell_b,
                        ndim=z_sub.shape[1],
                        device=z_sub.device,
                        dtype=z_sub.dtype,
                    )
                    mmd_raw = mmd_rff_fn(prior_samples, z_sub)
                else:
                    # fallback: exact MMD kernel if no RFF fn is provided
                    prior_samples = sample_shell(
                        B,
                        mmd_shell_a,
                        mmd_shell_b,
                        ndim=z.shape[1],
                        device=z.device,
                        dtype=z.dtype,
                    )
                    mmd_raw = compute_mmd(prior_samples, z.detach())
        else:
            mmd_raw = xb.new_tensor(0.0)

        # ---------- combine loss (same formula as train_epoch) ----------
        if mse_weight_scalar is None and not use_mmd:
            total = mse + lam_dcae * dcae_pen
        else:
            w = float(mse_weight_scalar) if mse_weight_scalar is not None else 0.0
            coeffMSE = 1.0
            dcae_term = lam_dcae * dcae_pen
            mmd_term = mmd_weight * mmd_raw
            total = (
                coeffMSE * (1.0 - w + 0.1) * mse
                + 0.5 * (w + 1.0) * mmd_term
                + (2.0 * w + 0.1) * dcae_term
            )

        # ---------- accumulate ----------
        bsz = xb.shape[0]
        mse_sum += float(mse) * bsz
        dcae_sum += float(dcae_pen) * bsz
        mmd_sum += float(mmd_raw) * bsz
        loss_sum += float(total) * bsz
        n += bsz

    n = max(1, n)
    return {
        "loss": loss_sum / n,
        "mse": mse_sum / n,
        "dcae": dcae_sum / n,
        "mmd": mmd_sum / n,
    }

# ------------------------------ Plot helpers ----------------------------------
def embed_all(
    encoder_eager: torch.nn.Module,
    X_cpu: torch.Tensor,
    device,
    batch: int = 8192,
) -> torch.Tensor:
    encoder_eager.eval()
    zs = []
    with torch.no_grad():
        for i in range(0, X_cpu.shape[0], batch):
            xb = X_cpu[i : i + batch].to(device, non_blocking=False)
            out = encoder_eager(xb)
            if isinstance(out, tuple):
                zb = out[0]
            else:
                zb = out
            zs.append(zb.cpu())
    return torch.cat(zs, dim=0)


def _as_np(x):
    if isinstance(x, np.ndarray):
        return x
    try:
        if torch.is_tensor(x):
            return x.detach().cpu().numpy()
    except Exception:
        pass
    return np.asarray(x)


def plot_latent(Z, color, title: str, out_path: str, s=1.0):
    Z = _as_np(Z)
    color = _as_np(color)
    Path(os.path.dirname(out_path)).mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(5.2, 4.3))
    if Z.shape[1] >= 2:
        plt.scatter(Z[:, 0], Z[:, 1], c=color, s=s, alpha=0.65)
        plt.xlabel("z1")
        plt.ylabel("z2")
    else:
        plt.scatter(np.arange(Z.shape[0]), Z[:, 0], c=color, s=s, alpha=0.65)
        plt.xlabel("index")
        plt.ylabel("z1")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def hist_latent_component(
    Z,
    idx: int,
    title: str,
    out_path: str,
    bins: int = 80,
):
    Z = _as_np(Z)
    Path(os.path.dirname(out_path)).mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(5.0, 3.8))
    plt.hist(Z[:, idx], bins=bins, density=True)
    plt.xlabel(f"z{idx+1}")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def plot_train_val_curves(
    train_vals,
    val_vals,
    ylabel: str,
    title: str,
    out_path: str,
):
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


def plot_all_losses(
    train_mse,
    val_mse,
    train_dcae,
    val_dcae,
    train_total,
    val_total,
    out_path: str,
):
    Path(os.path.dirname(out_path)).mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(7.2, 4.8))
    plt.plot(train_total, label="train total")
    plt.plot(val_total, label="val total")
    plt.plot(train_mse, label="train mse")
    plt.plot(val_mse, label="val mse")
    plt.plot(train_dcae, label="train dcae")
    plt.plot(val_dcae, label="val dcae")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.title("Train vs Val: total / mse / dcae")
    plt.legend(ncol=2)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


# ------------------ NumPy-callable snapshot wrappers ------------------
def _to_float32_np(a):
    x = np.asarray(a)
    if x.dtype != np.float32:
        x = x.astype(np.float32, copy=False)
    return x


def make_np_encoder_fn(enc_snapshot: nn.Module, batch: int = 8192):
    enc_cpu = deepcopy(enc_snapshot).to("cpu").eval()

    def f(X_np: np.ndarray):
        X = torch.from_numpy(_to_float32_np(X_np))
        outs = []
        with torch.no_grad():
            for i in range(0, X.shape[0], batch):
                out = enc_cpu(X[i : i + batch])
                if isinstance(out, tuple):
                    out = out[0]
                outs.append(out.cpu().numpy())
        return np.concatenate(outs, axis=0)

    return f



def make_np_decoder_fn(dec_snapshot: nn.Module, batch: int = 8192):
    dec_cpu = deepcopy(dec_snapshot).to("cpu").eval()

    def f(Z_np: np.ndarray):
        Z = torch.from_numpy(_to_float32_np(Z_np))
        outs = []
        with torch.no_grad():
            for i in range(0, Z.shape[0], batch):
                outs.append(dec_cpu(Z[i : i + batch]).cpu().numpy())
        return np.concatenate(outs, axis=0)

    return f


def make_np_autoencoder_fn(
    enc_snapshot: nn.Module,
    dec_snapshot: nn.Module,
    batch: int = 8192,
):
    enc_cpu = deepcopy(enc_snapshot).to("cpu").eval()
    dec_cpu = deepcopy(dec_snapshot).to("cpu").eval()

    def f(X_np: np.ndarray):
        X = torch.from_numpy(_to_float32_np(X_np))
        outs = []
        with torch.no_grad():
            for i in range(0, X.shape[0], batch):
                Z = enc_cpu(X[i : i + batch])
                # --- NEW: unwrap tuple from AnalyticEncoder ---
                if isinstance(Z, tuple):
                    Z = Z[0]          # take latent z, drop h1/h2
                # ----------------------------------------------
                outs.append(dec_cpu(Z).cpu().numpy())
        return np.concatenate(outs, axis=0)

    return f


# ==============================================================================
# 1) DATASET-GENERATION FUNCTION (user-facing)
# ==============================================================================
def generate_data(
    N: int,
    m: int,
    *,
    ell: Optional[int] = None,
    noise_sigma: float = 0.0,
    rotate: bool = False,
    val_frac: float = 0.20,
    seed: Optional[int] = 42,
):
    """
    High-level helper to generate the dataset and perform a train/val split.

    Returns a dict with:
        {
          "X_cpu": full dataset (N×m),
          "X_tr":  train subset,
          "X_val": val subset,
        }

    Parameters
    ----------
    N, m : int
        Dataset size and ambient dimension.
    ell : int or None
        Intrinsic dimension for the signal-ball. If None or >= m,
        fall back to a full m-ball.
    noise_sigma : float
        Input noise level.
    rotate : bool
        Random rotation of the signal subspace (when ell < m).
    val_frac : float
        Fraction of data to use for validation (0, 1).
    seed : int or None
        If not None, sets torch + NumPy seeds for reproducible data.
    """
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)

    X_cpu = generate_dataset(
        N=N,
        m=m,
        noise_sigma=noise_sigma,
        ell=ell,
        rotate=rotate,
    )
    X_tr, X_val = train_val_split(X_cpu, val_frac=val_frac, seed=(seed or 42))
    return {"X_cpu": X_cpu, "X_tr": X_tr, "X_val": X_val}


# ==============================================================================
# 2) TRAINING FUNCTION (user-facing)
# ==============================================================================
def run_training(
    X_cpu: torch.Tensor,
    X_tr: torch.Tensor,
    X_val: torch.Tensor,
    cell_types=None,          #  come from **data, not used inside training
    *,
    m: Optional[int] = None,  #  OPTIONAL
    k: int = 2,
    batch_size: int = 4096,
    epochs_mse: int = 25,
    epochs_dcae: int = 25,
    lambda_dcae: float = 3e-4,
    dcae_probes: int = 1,
    analytic_dcae: bool = True,
    # NEW: MMD + annealing
    use_mmd: bool = False,
    mmd_weight: float = 1.0,
    mmd_samples: int = 256,
    mmd_shell_a: float = 0.99,
    mmd_shell_b: float = 1.01,
    mse_anneal_ratio: float = 0.9,
    # ------------
    monitor: str = "auto",  # "auto" / "mse" / "total"
    early_stop: bool = True,
    early_stop_p1: bool = True,
    early_stop_p2: bool = True,
    patience: int = 10,
    min_delta: float = 1e-4,
    threads: Optional[int] = None,
    interop_threads: int = 8,
    num_workers: int = 8,
    prefetch_factor: int = 4,
    compile: bool = True,
    seed: Optional[int] = 42,
):
    """
    Train the autoencoder in two phases:

      Phase-1: MSE-only pretraining.
      Phase-2: MSE + λ * DCAE (+ optional MMD with annealing).

    If m is None, uses m = X_cpu.shape[1]; i.e. fully agnostic to the
    underlying ambient dimension / number of noisy vs non-noisy coords.
    """
    # ---------- infer / sanity-check m from data ----------
    m_data = int(X_cpu.shape[1])  # ambient dimension comes entirely from data

    if m is None:
        # architecture is agnostic: just use whatever the data has
        m = m_data
    else:
        m = int(m)
        if m != m_data:
            # we *ignore* the supplied m and trust the data, but warn you
            print(
                f"[warn] run_training: m={m} but X_cpu has {m_data} features; "
                f"using m={m_data} inferred from data."
            )
            m = m_data

    # sanity check train/val shapes
    if X_tr.shape[1] != m or X_val.shape[1] != m:
        raise ValueError(
            f"Dimension mismatch: X_cpu has {m} features, "
            f"but X_tr has {X_tr.shape[1]}, X_val has {X_val.shape[1]}."
        )

    # --- basic hygiene / normalization ---
    dcae_probes = max(1, int(dcae_probes))
    batch_size = max(8, int(batch_size))
    num_workers = max(0, int(num_workers))
    prefetch_factor = max(2, int(prefetch_factor))

    if threads is None:
        threads = min(56, os.cpu_count() or 8)

    mse_anneal_ratio = float(mse_anneal_ratio)
    if mse_anneal_ratio <= 0.0 or mse_anneal_ratio > 1.0:
        mse_anneal_ratio = 1.0

    # configure CPU threading
    configure_cpu(threads, interop_threads)

    # seeding (training-specific)
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)

    device = torch.device("cpu")

    # DataLoaders
    dl_tr = make_loader(
        X_tr,
        batch_size=batch_size,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
        shuffle=True,
    )
    dl_val = make_loader(
        X_val,
        batch_size=batch_size,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
        shuffle=False,
    )

    # Models (agnostic to “signal vs noise”; only see ambient dim m)
    # Models (choose analytic encoder if requested)
    if analytic_dcae:
        enc_eager = AnalyticEncoder(m, k).to(device)
    else:
        enc_eager = Encoder(m, k).to(device)
    dec_eager = Decoder(k, m).to(device)

    dec_eager = Decoder(k, m).to(device)

    # optional compile for Phase-1
    enc_mse, dec_mse = maybe_compile_for_training(
        enc_eager, dec_eager, enable_compile=bool(compile)
    )

    optE = torch.optim.AdamW(enc_eager.parameters(), lr=1e-3)
    optD = torch.optim.AdamW(dec_eager.parameters(), lr=1e-3)

    X_input_np = X_cpu.numpy().astype(np.float32, copy=False)

    # after enc_eager / dec_eager are created and moved to device
    mmd_rff_fn = None
    if use_mmd:
        mmd_rff_fn = make_rff_mmd(
            dim=k,
            n_features=8,  # number of random features; can tune
            sigma=None,  # or set a bandwidth if you want
            device=device,
            dtype=torch.float32,
        )

    # ------------------------------------------------------------------
    # Phase 1: MSE-only
    # ------------------------------------------------------------------
    print(f"[Phase 1] MSE pretraining for {epochs_mse} epochs")
    t0 = time.time()

    p1_train_mse, p1_val_mse = [], []

    best_val_mse = float("inf")
    best_epoch_p1 = 0
    patience_p1 = 0
    mse_best_state = (
        deepcopy(enc_eager.state_dict()),
        deepcopy(dec_eager.state_dict()),
    )

    do_es_p1 = bool(early_stop or early_stop_p1)

    for ep in range(1, int(epochs_mse) + 1):
        tr_stats = train_epoch(
            enc_mse,
            dec_mse,
            optE,
            optD,
            dl_tr,
            device,
            lam_dcae=0.0,
            use_dcae=False,
            dcae_probes=dcae_probes,
            use_mmd=False,
            mmd_weight=0.0,
            mmd_samples=mmd_samples,
            mse_weight_scalar=None,
        )
        va_stats = eval_epoch(
            enc_eager,
            dec_eager,
            dl_val,
            device,
            lam_dcae=0.0,
            use_dcae=False,
            dcae_probes=dcae_probes,
            use_mmd=False,
            mmd_weight=0.0,
            mse_weight_scalar=None,
        )

        p1_train_mse.append(tr_stats["mse"])
        p1_val_mse.append(va_stats["mse"])

        print(
            f"[MSE  {ep:03d}] train_mse={tr_stats['mse']:.6f} | "
            f"val_mse={va_stats['mse']:.6f}"
        )

        improved = (best_val_mse - va_stats["mse"]) > float(min_delta)
        if improved:
            best_val_mse = va_stats["mse"]
            best_epoch_p1 = ep
            patience_p1 = 0
            mse_best_state = (
                deepcopy(enc_eager.state_dict()),
                deepcopy(dec_eager.state_dict()),
            )
            if do_es_p1:
                print(
                    f"[P1] ✨ new best val_mse={best_val_mse:.6f} at epoch {ep}"
                )
        else:
            patience_p1 += 1
            if do_es_p1 and patience_p1 >= int(patience):
                print(
                    f"[P1] Early stop at epoch {ep} "
                    f"(no val MSE improvement > {min_delta:g} for {patience} epochs; "
                    f"best={best_val_mse:.6f} @ {best_epoch_p1})."
                )
                break

    print(f"[Phase 1] done in {time.time() - t0:.1f}s")

    enc_eager.load_state_dict(mse_best_state[0])
    dec_eager.load_state_dict(mse_best_state[1])

    with torch.no_grad():
        Z_mse = embed_all(enc_eager, X_cpu, device)
    enc_mse_snapshot = deepcopy(enc_eager).cpu().eval()
    dec_mse_snapshot = deepcopy(dec_eager).cpu().eval()

    enc_mse_fn = make_np_encoder_fn(enc_mse_snapshot)
    dec_mse_fn = make_np_decoder_fn(dec_mse_snapshot)
    ae_mse_fn = make_np_autoencoder_fn(enc_mse_snapshot, dec_mse_snapshot)

    Z_mse_np = enc_mse_fn(X_input_np)
    Xhat_mse_np = ae_mse_fn(X_input_np)

    # ------------------------------------------------------------------
    # Phase 2: MSE + DCAE (+ optional MMD)
    # ------------------------------------------------------------------
    use_dcae = (
        lambda_dcae is not None
        and float(lambda_dcae) > 0.0
    )
    have_phase2 = (int(epochs_dcae) > 0) and (use_dcae or use_mmd)

    history_phase2 = None

    if not have_phase2:
        if not use_dcae and not use_mmd:
            print(
                "[INFO] lambda_dcae=0, use_mmd=False, or epochs_dcae=0 → "
                "Phase-2 (DCAE/MMD) is bypassed (MSE-only)."
            )
        else:
            print(
                "[INFO] epochs_dcae=0 → Phase-2 (DCAE/MMD) is bypassed."
            )

        enc_final_snapshot = enc_mse_snapshot
        dec_final_snapshot = dec_mse_snapshot
        enc_final_fn = enc_mse_fn
        dec_final_fn = dec_mse_fn
        ae_final_fn = ae_mse_fn
        Z_final_np = Z_mse_np
        Xhat_final_np = Xhat_mse_np

    else:
        if use_dcae and use_mmd:
            print(
                f"[Phase 2] MSE + DCAE + MMD for {epochs_dcae} epochs "
                f"(λ={lambda_dcae}, probes={dcae_probes}, mmd_weight={mmd_weight})"
            )
        elif use_dcae:
            print(
                f"[Phase 2] MSE + DCAE for {epochs_dcae} epochs "
                f"(λ={lambda_dcae}, probes={dcae_probes})"
            )
        else:
            print(
                f"[Phase 2] MSE + MMD for {epochs_dcae} epochs "
                f"(mmd_weight={mmd_weight})"
            )


        # Annealing schedule for mixing MSE / MMD / DCAE
        mse_weight_schedule = frange_anneal(
            int(epochs_dcae),
            ratio=mse_anneal_ratio,
            start=0.0,
            end=1.0,
        )

        p2_train_total, p2_train_mse, p2_train_dcae = [], [], []
        p2_val_total, p2_val_mse, p2_val_dcae = [], [], []
        p2_train_mmd, p2_val_mmd = [], []

        monitor_total = monitor in ("auto", "total")
        best_val_metric = float("inf")
        best_epoch_p2 = 0
        patience_p2 = 0
        final_best_state = (
            deepcopy(enc_eager.state_dict()),
            deepcopy(dec_eager.state_dict()),
        )

        do_es_p2 = bool(early_stop or early_stop_p2)

        t1 = time.time()
        for ep in range(1, int(epochs_dcae) + 1):
            mse_weight_scalar = float(mse_weight_schedule[ep - 1])

            tr_stats = train_epoch(
                enc_mse,                      # use compiled module
                dec_mse,
                optE,
                optD,
                dl_tr,
                device,
                lam_dcae=float(lambda_dcae),
                use_dcae=use_dcae,
                dcae_probes=dcae_probes,
                use_mmd=bool(use_mmd),
                mmd_weight=float(mmd_weight),
                mmd_rff_fn=mmd_rff_fn,
                mmd_samples=int(mmd_samples),
                mse_weight_scalar=mse_weight_scalar,
                mmd_shell_a=float(mmd_shell_a),
                mmd_shell_b=float(mmd_shell_b),
                analytic_dcae=bool(analytic_dcae),
            )
            va_stats = eval_epoch(
                enc_mse,                      # same compiled module
                dec_mse,
                dl_val,
                device,
                lam_dcae=float(lambda_dcae),
                use_dcae=use_dcae,
                dcae_probes=dcae_probes,
                use_mmd=bool(use_mmd),
                mmd_weight=float(mmd_weight),
                mmd_rff_fn=mmd_rff_fn,
                mmd_samples=int(mmd_samples),
                mse_weight_scalar=mse_weight_scalar,
                mmd_shell_a=float(mmd_shell_a),
                mmd_shell_b=float(mmd_shell_b),
                analytic_dcae=bool(analytic_dcae),
            )

            p2_train_total.append(tr_stats["loss"])
            p2_val_total.append(va_stats["loss"])
            p2_train_mse.append(tr_stats["mse"])
            p2_val_mse.append(va_stats["mse"])
            p2_train_dcae.append(tr_stats["dcae"])
            p2_val_dcae.append(va_stats["dcae"])
            p2_train_mmd.append(tr_stats["mmd"])
            p2_val_mmd.append(va_stats["mmd"])

            print(
                f"[DCAE {ep:03d}] "
                f"train: total={tr_stats['loss']:.6f} "
                f"mse={tr_stats['mse']:.6f} dcae={tr_stats['dcae']:.6f} "
                f"mmd={tr_stats['mmd']:.6f} | "
                f"val: total={va_stats['loss']:.6f} "
                f"mse={va_stats['mse']:.6f} dcae={va_stats['dcae']:.6f} "
                f"mmd={va_stats['mmd']:.6f}"
            )

            val_metric = va_stats["loss"] if monitor_total else va_stats["mse"]
            improved = (best_val_metric - val_metric) > float(min_delta)
            if improved:
                best_val_metric = val_metric
                best_epoch_p2 = ep
                patience_p2 = 0
                final_best_state = (
                    deepcopy(enc_eager.state_dict()),
                    deepcopy(dec_eager.state_dict()),
                )
                if do_es_p2:
                    name = "total" if monitor_total else "mse"
                    print(
                        f"[P2] ✨ new best val_{name}="
                        f"{best_val_metric:.6f} at epoch {ep}"
                    )
            else:
                patience_p2 += 1
                if do_es_p2 and patience_p2 >= int(patience):
                    name = "total" if monitor_total else "mse"
                    print(
                        f"[P2] Early stop at epoch {ep} "
                        f"(no val {name} improvement > {min_delta:g} "
                        f"for {patience} epochs; "
                        f"best={best_val_metric:.6f} @ {best_epoch_p2})."
                    )
                    break

        print(f"[Phase 2] done in {time.time() - t1:.1f}s")

        enc_eager.load_state_dict(final_best_state[0])
        dec_eager.load_state_dict(final_best_state[1])

        with torch.no_grad():
            Z_dcae = embed_all(enc_eager, X_cpu, device)

        enc_final_snapshot = deepcopy(enc_eager).cpu().eval()
        dec_final_snapshot = deepcopy(dec_eager).cpu().eval()

        enc_final_fn = make_np_encoder_fn(enc_final_snapshot)
        dec_final_fn = make_np_decoder_fn(dec_final_snapshot)
        ae_final_fn = make_np_autoencoder_fn(
            enc_final_snapshot,
            dec_final_snapshot,
        )

        Z_final_np = enc_final_fn(X_input_np)
        Xhat_final_np = ae_final_fn(X_input_np)

        history_phase2 = {
            "train_total": p2_train_total,
            "val_total": p2_val_total,
            "train_mse": p2_train_mse,
            "val_mse": p2_val_mse,
            "train_dcae": p2_train_dcae,
            "val_dcae": p2_val_dcae,
            "train_mmd": p2_train_mmd,
            "val_mmd": p2_val_mmd,
            "best_val_metric": best_val_metric,
            "best_epoch": best_epoch_p2,
            "monitor_total": monitor_total,
            "mse_weight_schedule": mse_weight_schedule.tolist(),
        }

    result = {
        "config": {
            "m": int(m),   # effective m inferred from data
            "k": int(k),
            "batch_size": int(batch_size),
            "epochs_mse": int(epochs_mse),
            "epochs_dcae": int(epochs_dcae),
            "lambda_dcae": float(lambda_dcae),
            "dcae_probes": int(dcae_probes),
            "use_mmd": bool(use_mmd),
            "mmd_weight": float(mmd_weight),
            "mmd_shell_a": float(mmd_shell_a),
            "mmd_shell_b": float(mmd_shell_b),
            "mse_anneal_ratio": float(mse_anneal_ratio),
            "monitor": monitor,
            "early_stop": bool(early_stop),
            "early_stop_p1": bool(early_stop_p1),
            "early_stop_p2": bool(early_stop_p2),
            "patience": int(patience),
            "min_delta": float(min_delta),
            "threads": int(threads),
            "interop_threads": int(interop_threads),
            "num_workers": int(num_workers),
            "prefetch_factor": int(prefetch_factor),
            "compile": bool(compile),
            "seed": seed,
            "use_dcae": bool(use_dcae),
        },
        "data": {
            "X_input": X_input_np,
        },
        "history": {
            "phase1": {
                "train_mse": p1_train_mse,
                "val_mse": p1_val_mse,
                "best_val_mse": best_val_mse,
                "best_epoch": best_epoch_p1,
            },
            "phase2": history_phase2,
        },
        "snapshots": {
            "mse": {
                "encoder": enc_mse_snapshot,
                "decoder": dec_mse_snapshot,
                "encoder_fn": enc_mse_fn,
                "decoder_fn": dec_mse_fn,
                "autoencoder_fn": ae_mse_fn,
                "Z": Z_mse_np,
                "Xhat": Xhat_mse_np,
            },
            "final": {
                "encoder": enc_final_snapshot,
                "decoder": dec_final_snapshot,
                "encoder_fn": enc_final_fn,
                "decoder_fn": dec_final_fn,
                "autoencoder_fn": ae_final_fn,
                "Z": Z_final_np,
                "Xhat": Xhat_final_np,
            },
        },
    }

    return result




# ---------- NumPy-only plotting helpers ----------

def plot_latent_np(
    Z_np: np.ndarray,
    color_np: np.ndarray,
    title: str,
    out_path: str,
    s: float = 1.0,
):
    """
    Scatter of latent codes Z_np ∈ R^{N×k} colored by color_np (length N).

    If k ≥ 2 → use first two components.
    If k = 1 → scatter vs index.
    """
    Z = np.asarray(Z_np)
    color = np.asarray(color_np)

    Path(os.path.dirname(out_path)).mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(5.2, 4.3))

    if Z.shape[1] >= 2:
        plt.scatter(Z[:, 0], Z[:, 1], c=color, s=s, alpha=0.65)
        plt.xlabel("z1")
        plt.ylabel("z2")
    else:
        plt.scatter(np.arange(Z.shape[0]), Z[:, 0], c=color, s=s, alpha=0.65)
        plt.xlabel("index")
        plt.ylabel("z1")

    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def plot_hist_1d_np(
    values_np: np.ndarray,
    title: str,
    out_path: str,
    bins: int = 80,
    xlabel: str = "z",
):
    """
    1D histogram of a NumPy vector.
    """
    vals = np.asarray(values_np)
    Path(os.path.dirname(out_path)).mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(5.0, 3.8))
    plt.hist(vals, bins=bins, density=True)
    plt.xlabel(xlabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def plot_train_val_np(
    train_vals: np.ndarray,
    val_vals: np.ndarray,
    ylabel: str,
    title: str,
    out_path: str,
):
    """
    Plot train/val curves vs epoch index (1..T).

    train_vals, val_vals: 1D sequences of same length (list or np.ndarray).
    """
    train_vals = np.asarray(train_vals, dtype=float)
    val_vals = np.asarray(val_vals, dtype=float)
    epochs = np.arange(1, len(train_vals) + 1)

    Path(os.path.dirname(out_path)).mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(6.4, 4.4))

    plt.plot(epochs, train_vals, label="train")
    plt.plot(epochs, val_vals, label="val")
    plt.xlabel("epoch")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def plot_all_losses_np(
    train_mse: np.ndarray,
    val_mse: np.ndarray,
    train_dcae: np.ndarray,
    val_dcae: np.ndarray,
    train_total: np.ndarray,
    val_total: np.ndarray,
    out_path: str,
):
    """
    Combined loss plot (total / mse / dcae) vs epoch index (1..T).
    All inputs: 1D sequences (list or np.ndarray) of same length.
    """
    train_mse = np.asarray(train_mse, dtype=float)
    val_mse = np.asarray(val_mse, dtype=float)
    train_dcae = np.asarray(train_dcae, dtype=float)
    val_dcae = np.asarray(val_dcae, dtype=float)
    train_total = np.asarray(train_total, dtype=float)
    val_total = np.asarray(val_total, dtype=float)

    epochs = np.arange(1, len(train_total) + 1)

    Path(os.path.dirname(out_path)).mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(7.2, 4.8))

    plt.plot(epochs, train_total, label="train total")
    plt.plot(epochs, val_total, label="val total")
    plt.plot(epochs, train_mse, label="train mse")
    plt.plot(epochs, val_mse, label="val mse")
    plt.plot(epochs, train_dcae, label="train dcae")
    plt.plot(epochs, val_dcae, label="val dcae")

    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.title("Train vs Val: total / mse / dcae")
    plt.legend(ncol=2)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


# ==============================================================================
# 3) PLOTTING FUNCTION (user-facing)
# ==============================================================================
def plot_results(
    result: dict,
    *,
    outdir: str = "demo_plots",
    color=None,
    cell_types=None,      # NEW
    pseudotime=None,      # will use in part 3
    scatter_size: float = 1.0,
    hist_component: int = 0,
    hist_bins: int = 80,
    prefix: str = "",
):
    """
    Plot training curves and latent space visualizations using a result dict
    returned by `run_training`.

    Parameters
    ----------
    result : dict
        Output of `run_training(...)`.
    outdir : str
        Directory where PNGs will be saved.
    color : array-like or None
        Optional color vector for the latent scatter plots. If None, uses
        the first coordinate of X_input.
    scatter_size : float
        Marker size for scatter plots.
    hist_component : int
        Which latent dimension to plot in histogram.
    hist_bins : int
        Number of bins for histogram.
    prefix : str
        Optional prefix added to all filenames (e.g., "exp1_").
    """
    Path(outdir).mkdir(parents=True, exist_ok=True)

    cfg = result["config"]
    m = cfg["m"]
    k = cfg["k"]
    use_dcae = cfg["use_dcae"]

    X_input = np.asarray(result["data"]["X_input"], dtype=np.float32)

    if color is None:
        color = X_input[:, 0]
    color = np.asarray(color)

    # ----------------- Phase-1 curves ----------------------------------------
    hist1 = result["history"]["phase1"]
    if hist1 is not None and len(hist1["train_mse"]) > 0:
        plot_train_val_np(
            train_vals=hist1["train_mse"],
            val_vals=hist1["val_mse"],
            ylabel="MSE",
            title=f"Phase-1 MSE (m={m}, k={k})",
            out_path=os.path.join(outdir, f"{prefix}phase1_mse_train_val.png"),
        )

    # ----------------- Phase-2 curves (if any) -------------------------------
    hist2 = result["history"]["phase2"]
    if use_dcae and hist2 is not None and len(hist2["train_total"]) > 0:
        # total loss
        plot_train_val_np(
            train_vals=hist2["train_total"],
            val_vals=hist2["val_total"],
            ylabel="loss",
            title="Phase-2 Total Loss (MSE + λ·DCAE)",
            out_path=os.path.join(outdir, f"{prefix}phase2_total_train_val.png"),
        )
        # MSE
        plot_train_val_np(
            train_vals=hist2["train_mse"],
            val_vals=hist2["val_mse"],
            ylabel="MSE",
            title="Phase-2 MSE",
            out_path=os.path.join(outdir, f"{prefix}phase2_mse_train_val.png"),
        )
        # DCAE
        plot_train_val_np(
            train_vals=hist2["train_dcae"],
            val_vals=hist2["val_dcae"],
            ylabel="DCAE",
            title="Phase-2 DCAE penalty",
            out_path=os.path.join(outdir, f"{prefix}phase2_dcae_train_val.png"),
        )
        # combined view
        plot_all_losses_np(
            train_mse=hist2["train_mse"],
            val_mse=hist2["val_mse"],
            train_dcae=hist2["train_dcae"],
            val_dcae=hist2["val_dcae"],
            train_total=hist2["train_total"],
            val_total=hist2["val_total"],
            out_path=os.path.join(outdir, f"{prefix}phase2_all_losses.png"),
        )

    # ----------------- Latent visualizations ---------------------------------
    Z_mse = np.asarray(result["snapshots"]["mse"]["Z"], dtype=np.float32)
    Z_final = np.asarray(result["snapshots"]["final"]["Z"], dtype=np.float32)

    # -------- choose coloring for latent scatter --------
    # 1) explicit cell types
    # 2) explicit pseudotime
    # 3) explicit generic color
    # 4) fallback: first coordinate of X_input
    X_input = np.asarray(result["data"]["X_input"], dtype=np.float32)

    if cell_types is not None:
        # ---- MSE latent, colored by cell type ----
        plot_latent_celltypes_np(
            Z_np=Z_mse,
            cell_types=cell_types,
            title=f"Latent (k={k}) after MSE-only, m={m} — cell types",
            out_path=os.path.join(
                outdir,
                f"{prefix}latent_mse_only_m{m}_celltypes.png",
            ),
            s=scatter_size,
        )

        # ---- FINAL latent, colored by cell type ----
        if use_dcae:
            title_latent_final = (
                f"Latent (k={k}) after MSE+DCAE, m={m} — cell types"
            )
            suffix = "dcae"
        else:
            title_latent_final = (
                f"Latent (k={k}) final (MSE-only), m={m} — cell types"
            )
            suffix = "final"

        plot_latent_celltypes_np(
            Z_np=Z_final,
            cell_types=cell_types,
            title=title_latent_final,
            out_path=os.path.join(
                outdir,
                f"{prefix}latent_mse_{suffix}_m{m}_celltypes.png",
            ),
            s=scatter_size,
        )

    # (keep your existing histograms; they don’t depend on color)
    z_comp_mse = Z_mse[:, hist_component]
    plot_hist_1d_np(
        values_np=z_comp_mse,
        title=f"Histogram of z{hist_component+1} (MSE only) — m={m}",
        xlabel=f"z{hist_component+1}",
        out_path=os.path.join(
            outdir,
            f"{prefix}hist_z{hist_component+1}_mse_m{m}.png",
        ),
        bins=hist_bins,
    )

    z_comp_final = Z_final[:, hist_component]
    if use_dcae:
        title_hist_final = (
            f"Histogram of z{hist_component+1} (MSE + DCAE) — m={m}"
        )
        suffix = "dcae"
    else:
        title_hist_final = (
            f"Histogram of z{hist_component+1} (final MSE-only) — m={m}"
        )
        suffix = "final"

    plot_hist_1d_np(
        values_np=z_comp_final,
        title=title_hist_final,
        xlabel=f"z{hist_component+1}",
        out_path=os.path.join(
            outdir,
            f"{prefix}hist_z{hist_component+1}_mse_{suffix}_m{m}.png",
        ),
        bins=hist_bins,
    )




# ==============================================================================
# CLI glue (parse_args + main)
# ==============================================================================
def parse_args(arg_list=None):
    p = argparse.ArgumentParser(description="AE splitting demo (CPU)")

    # core problem sizes / data
    p.add_argument("--m", type=int, default=15, help="ambient dim (R^m)")
    p.add_argument(
        "--ell",
        type=int,
        default=10,
        help="intrinsic ball dim ℓ (< m) for signal subspace",
    )
    p.add_argument("--k", type=int, default=2, help="latent dim")
    p.add_argument("--N", type=int, default=12000, help="dataset size")
    p.add_argument(
        "--noise-sigma",
        type=float,
        default=0.0,
        help="Gaussian noise on inputs",
    )

    # training schedule
    p.add_argument(
        "--batch-size",
        type=int,
        default=4096,
        help="minibatch size",
    )
    p.add_argument(
        "--epochs-mse",
        type=int,
        default=25,
        help="epochs for Phase-1 (MSE only)",
    )
    p.add_argument(
        "--epochs-dcae",
        type=int,
        default=25,
        help="epochs for Phase-2 (MSE + DCAE)",
    )
    p.add_argument(
        "--lambda-dcae",
        type=float,
        default=3e-4,
        help="DCAE penalty weight (0 disables Phase-2)",
    )
    p.add_argument(
        "--dcae-probes",
        type=int,
        default=1,
        help="Hutchinson probes per batch (>=1)",
    )

    p.add_argument(
        "--use-mmd",
        action="store_true",
        help="enable MMD regularization in Phase-2",
    )
    p.add_argument(
        "--mmd-weight",
        type=float,
        default=1.0,
        help="MMD loss weight",
    )
    p.add_argument(
        "--mmd-samples",
        type=int,
        default=256,
        help="latent samples per batch used to estimate MMD",
    )
    p.add_argument(
        "--mmd-shell-a",
        type=float,
        default=0.99,
        help="inner radius of target shell prior",
    )
    p.add_argument(
        "--mmd-shell-b",
        type=float,
        default=1.01,
        help="outer radius of target shell prior",
    )
    p.add_argument(
        "--mse-anneal-ratio",
        type=float,
        default=0.9,
        help="fraction of Phase-2 epochs used to ramp annealing weight from 0 to 1",
    )


    # splits / early stopping
    p.add_argument(
        "--val-frac",
        type=float,
        default=0.20,
        help="validation fraction in (0,1)",
    )
    p.add_argument(
        "--monitor",
        type=str,
        default="auto",
        choices=["auto", "mse", "total"],
        help="Phase-2 early-stop metric: 'total' (default via 'auto') or 'mse'",
    )

    # global early-stop switch (default ON), with opt-out flag
    p.add_argument(
        "--early-stop",
        dest="early_stop",
        action="store_true",
        help="enable early stopping (default: on)",
    )
    p.add_argument(
        "--no-early-stop",
        dest="early_stop",
        action="store_false",
        help="disable all early stopping",
    )
    p.set_defaults(early_stop=True)

    # per-phase toggles (default ON), with opt-out flags
    p.add_argument(
        "--early-stop-p1",
        dest="early_stop_p1",
        action="store_true",
        help="enable early stopping in Phase-1 (val MSE)",
    )
    p.add_argument(
        "--no-early-stop-p1",
        dest="early_stop_p1",
        action="store_false",
        help="disable early stopping in Phase-1",
    )
    p.set_defaults(early_stop_p1=True)

    p.add_argument(
        "--early-stop-p2",
        dest="early_stop_p2",
        action="store_true",
        help="enable early stopping in Phase-2 (monitor per --monitor)",
    )
    p.add_argument(
        "--no-early-stop-p2",
        dest="early_stop_p2",
        action="store_false",
        help="disable early stopping in Phase-2",
    )
    p.set_defaults(early_stop_p2=True)

    p.add_argument(
        "--patience",
        type=int,
        default=10,
        help="epochs without improvement before stopping",
    )
    p.add_argument(
        "--min-delta",
        type=float,
        default=1e-4,
        help="minimum improvement to reset patience",
    )

    # system / reproducibility
    p.add_argument("--seed", type=int, default=42, help="PRNG seed")
    p.add_argument(
        "--outdir",
        type=str,
        default="demo_plots",
        help="output directory",
    )

    # CPU threading and dataloader
    p.add_argument(
        "--threads",
        type=int,
        default=min(56, os.cpu_count() or 8),
        help="intra-op threads",
    )
    p.add_argument(
        "--interop-threads",
        type=int,
        default=8,
        help="interop threads",
    )
    p.add_argument(
        "--num-workers",
        type=int,
        default=8,
        help="DataLoader workers (CPU)",
    )
    p.add_argument(
        "--prefetch-factor",
        type=int,
        default=4,
        help="DataLoader prefetch factor",
    )

    # optional torch.compile for Phase-1
    p.add_argument(
        "--compile",
        action="store_true",
        help="try torch.compile (CPU) for Phase-1",
    )

    # parse (support programmatic call via arg_list)
    args = p.parse_args(arg_list if arg_list is not None else sys.argv[1:])

    # ----------- safety guards / normalization -----------
    args.dcae_probes = max(1, int(args.dcae_probes))
    args.batch_size = max(8, int(args.batch_size))
    args.num_workers = max(0, int(args.num_workers))
    args.prefetch_factor = max(2, int(args.prefetch_factor))

    # val_frac must be in (0,1)
    if not (0.0 < float(args.val_frac) < 1.0):
        raise ValueError(f"--val-frac must be in (0,1); got {args.val_frac}")

    # ell sanity (let the data generator assert more strictly)
    if args.ell >= args.m:
        print(
            f"[warn] --ell ({args.ell}) >= --m ({args.m}); "
            f"generator may ignore ell or raise. Consider setting ell < m."
        )

    return args


def main(arg_list=None):
    """
    CLI entrypoint. For programmatic use prefer:

        data = generate_data(...)
        result = run_training(..., **data)
        plot_results(result, outdir=...)
    """
    args = parse_args(arg_list)

    # configure CPU for data generation as well
    configure_cpu(args)

    # --- data ---
    data = generate_data(
        N=args.N,
        m=args.m,
        ell=args.ell,
        noise_sigma=args.noise_sigma,
        rotate=False,
        val_frac=args.val_frac,
        seed=args.seed,
    )

    # --- training ---
    result = run_training(
        X_cpu=data["X_cpu"],
        X_tr=data["X_tr"],
        X_val=data["X_val"],
        m=args.m,
        k=args.k,
        batch_size=args.batch_size,
        epochs_mse=args.epochs_mse,
        epochs_dcae=args.epochs_dcae,
        lambda_dcae=args.lambda_dcae,
        dcae_probes=args.dcae_probes,
        use_mmd=args.use_mmd,
        mmd_weight=args.mmd_weight,
        mmd_samples=args.mmd_samples,
        mmd_shell_a=args.mmd_shell_a,
        mmd_shell_b=args.mmd_shell_b,
        mse_anneal_ratio=args.mse_anneal_ratio,
        monitor=args.monitor,
        early_stop=args.early_stop,
        early_stop_p1=args.early_stop_p1,
        early_stop_p2=args.early_stop_p2,
        patience=args.patience,
        min_delta=args.min_delta,
        threads=args.threads,
        interop_threads=args.interop_threads,
        num_workers=args.num_workers,
        prefetch_factor=args.prefetch_factor,
        compile=args.compile,
        seed=args.seed,
    )

    # --- plotting ---
    #plot_results(result, outdir=args.outdir)

    #print(f"[DONE] Figures saved to: {args.outdir}")
    return result


import pandas as pd  # make sure this is imported at top


def load_splatter_data(
    expr_path: str = "expression_data_biologically_realistic50gene.csv",
    celltype_path: str = "cell_types_biologically_realistic50gene.csv",
    *,
    val_frac: float = 0.2,
    seed: int = 42,
    standardize: bool = True,
):
    """
    Load expression + cell types generated by the R splatter pipeline and
    prepare tensors for `run_training`.

    Returns:
        {
          "X_cpu":   torch.FloatTensor [N, m],
          "X_tr":    torch.FloatTensor [N_tr, m],
          "X_val":   torch.FloatTensor [N_val, m],
          "cell_types": np.ndarray of shape [N] with string labels
        }
    """
    # expression data: rows = cells, cols = genes
    expr_df = pd.read_csv(expr_path, index_col=0)
    X_np = expr_df.to_numpy().astype(np.float32)  # (N, m)

    if standardize:
        mu = X_np.mean(axis=0, keepdims=True)
        sd = X_np.std(axis=0, keepdims=True)
        sd[sd == 0] = 1.0
        X_np = (X_np - mu) / sd

    X_cpu = torch.from_numpy(X_np)
    n_cells = X_cpu.shape[0]

    # ---- robust reading of cell types ----
    # Case 1: file has a header (R default)
    ct_df = pd.read_csv(celltype_path)

    # If it's just a single column, use that column; otherwise use the last column
    if ct_df.shape[1] == 1:
        col = ct_df.columns[0]
    else:
        col = ct_df.columns[-1]

    cell_types = ct_df[col].astype(str).to_numpy()

    # If we somehow ended up with an off-by-one (header treated as row),
    # fix it by dropping the first row when lengths differ by exactly 1.
    if cell_types.shape[0] == n_cells + 1:
        cell_types = cell_types[1:]

    if cell_types.shape[0] != n_cells:
        raise ValueError(
            f"Row mismatch after loading: X has {n_cells} cells, "
            f"cell_types has {cell_types.shape[0]}"
        )

    # use existing train/val splitter
    X_tr, X_val = train_val_split(X_cpu, val_frac=val_frac, seed=seed)

    return {
        "X_cpu": X_cpu,
        "X_tr": X_tr,
        "X_val": X_val,
        "cell_types": cell_types,
    }




def plot_latent_celltypes_np(
    Z_np: np.ndarray,
    cell_types,
    title: str,
    out_path: str,
    s: float = 1.0,
):
    """
    Scatter of latent codes Z_np ∈ R^{N×k} colored by categorical cell types.
    Adds a legend with one entry per cell type.
    """
    Z = np.asarray(Z_np)
    cell_types = np.asarray(cell_types).astype(str)

    classes, labels = np.unique(cell_types, return_inverse=True)
    cmap = plt.cm.get_cmap("tab20", len(classes))

    Path(os.path.dirname(out_path)).mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(6, 5))

    if Z.shape[1] >= 2:
        sc = plt.scatter(
            Z[:, 0],
            Z[:, 1],
            c=labels,
            s=s,
            alpha=0.7,
            cmap=cmap,
        )
        plt.xlabel("z1")
        plt.ylabel("z2")
    else:
        sc = plt.scatter(
            np.arange(Z.shape[0]),
            Z[:, 0],
            c=labels,
            s=s,
            alpha=0.7,
            cmap=cmap,
        )
        plt.xlabel("index")
        plt.ylabel("z1")

    # Legend with one marker per cell type
    handles = []
    for i, ctype in enumerate(classes):
        handles.append(
            Line2D(
                [0],
                [0],
                marker="o",
                linestyle="",
                label=ctype,
                markerfacecolor=cmap(i),
                markeredgecolor="none",
                markersize=5,
            )
        )
    plt.legend(
        handles=handles,
        bbox_to_anchor=(1.05, 1),
        loc="upper left",
        borderaxespad=0.0,
        fontsize=8,
    )

    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close()

def plot_latent_pseudotime_np(
    Z_np: np.ndarray,
    pseudotime,
    title: str,
    out_path: str,
    s: float = 1.0,
):
    """
    Scatter of latent codes colored by continuous pseudotime.
    """
    Z = np.asarray(Z_np)
    pt = np.asarray(pseudotime, dtype=float)

    if Z.shape[0] != pt.shape[0]:
        raise ValueError(
            f"Z has {Z.shape[0]} rows, pseudotime has {pt.shape[0]} values"
        )

    Path(os.path.dirname(out_path)).mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(6, 5))

    if Z.shape[1] >= 2:
        sc = plt.scatter(
            Z[:, 0],
            Z[:, 1],
            c=pt,
            s=s,
            alpha=0.7,
            cmap="viridis",
        )
        plt.xlabel("z1")
        plt.ylabel("z2")
    else:
        sc = plt.scatter(
            np.arange(Z.shape[0]),
            Z[:, 0],
            c=pt,
            s=s,
            alpha=0.7,
            cmap="viridis",
        )
        plt.xlabel("index")
        plt.ylabel("z1")

    cbar = plt.colorbar(sc)
    cbar.set_label("pseudotime")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


# ------------------------------- Entrypoint -----------------------------------
if __name__ == "__main__":
    main()
