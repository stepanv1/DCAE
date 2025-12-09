#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import importlib, split_demo_utils
importlib.reload(split_demo_utils)
from split_demo_utils import generate_data
from split_demo_utils import run_training
from split_demo_utils import plot_results, plot_latent_np, plot_hist_1d_np, plot_train_val_np, plot_latent_pseudotime_np
import numpy as np
import pandas as pd
from split_demo_utils import load_splatter_data

data = load_splatter_data(
    expr_path="/home/sgrinek/PycharmProjects/DCAE/R/expression_data_biologically_realistic15gene_batchCells_20000_.csv",
    celltype_path="/home/sgrinek/PycharmProjects/DCAE/R/cell_types_biologically_realistic15gene_batchCells_20000_.csv",
    val_frac=0.2,
    seed=42,
)

# data is a dict with keys: "X_cpu", "X_tr", "X_val"

#Training (Phase-1 MSE, optional Phase-2 MSE+DCAE):
result = run_training(
        **data,          # X_cpu, X_tr, X_val
        k=2,
        batch_size=4096,
        epochs_mse=0000,
        epochs_dcae=20000,
        lambda_dcae=5e-1,
        dcae_probes=3,
        monitor="auto",
        early_stop=True,
        early_stop_p1=True,
        early_stop_p2=True,
        patience=100,
        min_delta=1e-6,
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
        prefix="exp1_15genes",
    )

plot_results(
    result,
    outdir="/home/sgrinek/PycharmProjects/DCAE/PAPERPLOTS",
    cell_types=data["cell_types"],  # <- categorical coloring
    prefix="splatter_15genes",
)

import pandas as pd
import numpy as np

# Load pseudotime (must be same cell order as expression)
pt_df = pd.read_csv("/home/sgrinek/PycharmProjects/DCAE/R/pseudotime_biologically_realistic15gene_batchCells_20000_.csv")
pseudotime = pt_df["pseudotime"].to_numpy().astype(np.float32)

Z_final = np.asarray(result["snapshots"]["final"]["Z"], dtype=np.float32)

plot_latent_pseudotime_np(
    Z_np=Z_final,
    pseudotime=pseudotime,
    title="Latent (final) colored by pseudotime",
    out_path="/home/sgrinek/PycharmProjects/DCAE/PAPERPLOTS/pseudotime_biologically_realistic15gene_batchCells_20000.png",
    s=1.0,
)


Z_mse = result["snapshots"]["mse"]["Z"]          # (N, k) np.ndarray
Z_final = result["snapshots"]["final"]["Z"]
X_input = result["data"]["X_input"]
color = X_input[:, 0]

plot_latent_np(Z_mse, color, "My custom MSE latent",
"/home/sgrinek/PycharmProjects/DCAE/PAPERPLOTS/custom_mse_latent.png",s =0.01)
plot_latent_np(Z_final, color, "My custom FINAL latent",
            "/home/sgrinek/PycharmProjects/DCAE/PAPERPLOTS/custom_final_latent.png",s =0.01)

plot_hist_1d_np(Z_final[:, 0], "Final z1", "plots/custom_hist_z1_final.png",
                xlabel="z1", bins=100)

train_mse = np.array(result["history"]["phase1"]["train_mse"])
val_mse   = np.array(result["history"]["phase1"]["val_mse"])
plot_train_val_np(train_mse, val_mse, "MSE", "Custom P1 MSE",
                  "/home/sgrinek/PycharmProjects/DCAE/PAPERPLOTS/custom_phase1_mse15.png")



