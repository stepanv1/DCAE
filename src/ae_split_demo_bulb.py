#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import importlib, split_demo_utils
importlib.reload(split_demo_utils)
from split_demo_utils import generate_data
from split_demo_utils import run_training
from split_demo_utils import plot_results, plot_latent_np, plot_hist_1d_np, plot_train_val_np
import numpy as np

data = generate_data(
        N=12000,
        m=15,
        ell=10,
        noise_sigma=1,
        rotate=True,
        val_frac=0.2,
        seed=42,
    )

#split_demo_utils.main([
#      '--m','15','--k','2','--N','12000',
#      '--epochs-mse','20','--epochs-dcae','20','--lambda-dcae','1e-4',
#      '--batch-size','4096','--threads','56','--interop-threads','8',
#      '--num-workers','8','--prefetch-factor','4','--compile'
#  ])

# Dataset generation (with train/val split):
# data is a dict with keys: "X_cpu", "X_tr", "X_val"

#Training (Phase-1 MSE, optional Phase-2 MSE+DCAE):



result = run_training(
        **data,          # X_cpu, X_tr, X_val
        k=2,
        batch_size=4096,
        epochs_mse=20000,
        epochs_dcae=20000,
        lambda_dcae=1,
        dcae_probes=3,
        monitor="auto",
        early_stop=True,
        early_stop_p1=True,
        early_stop_p2=True,
        patience=100,
        min_delta=1e-5,
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

# Plotting (uses training result; can be called independently):


plot_results(
        result,
        outdir="/home/sgrinek/PycharmProjects/DCAE/PAPERPLOTS",
        color=None,          # default: first coordinate of X
        scatter_size=1.0,
        hist_component=0,
        hist_bins=80,
        prefix="exp1_",
    )


Z_mse = result["snapshots"]["mse"]["Z"]          # (N, k) np.ndarray
Z_final = result["snapshots"]["final"]["Z"]
X_input = result["data"]["X_input"]
color = X_input[:, 0]

plot_latent_np(Z_mse, color, "MSE latent", "/home/sgrinek/PycharmProjects/DCAE/PAPERPLOTS/custom_mse_latent.png")
plot_latent_np(Z_final, color, "FINAL latent", "/home/sgrinek/PycharmProjects/DCAE/PAPERPLOTS/custom_final_latent.png")

plot_hist_1d_np(Z_final[:, 0], "Final z1", "plots/custom_hist_z1_final.png",
                xlabel="z1", bins=100)

train_mse = np.array(result["history"]["phase1"]["train_mse"])
val_mse   = np.array(result["history"]["phase1"]["val_mse"])
plot_train_val_np(train_mse, val_mse, "MSE", "Custom P1 MSE",
                  "/home/sgrinek/PycharmProjects/DCAE/PAPERPLOTS/custom_phase1_mse.png")



