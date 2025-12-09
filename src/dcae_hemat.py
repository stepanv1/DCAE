#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import importlib, dcae_utils
importlib.reload(dcae_utils)
from dcae_utils import generate_data
from dcae_utils import run_training
from dcae_utils import plot_results, plot_latent_np, plot_hist_1d_np, plot_train_val_np, plot_latent_pseudotime_np
import numpy as np
import pandas as pd
from dcae_utils import load_splatter_data

data = load_splatter_data(
    expr_path="/home/sgrinek/PycharmProjects/DCAE/R/expression_data_biologically_realistic15gene.csv",
    celltype_path="/home/sgrinek/PycharmProjects/DCAE/R/cell_types_biologically_realistic15gene.csv",
    val_frac=0.2,
    seed=42,
)

# data is a dict with keys: "X_cpu", "X_tr", "X_val"

#Training (Phase-1 MSE, optional Phase-2 MSE+DCAE):
result = run_training(
    **data,          # X_cpu, X_tr, X_val, cell_types
    k=3,
    batch_size=4096,
    epochs_mse=0,
    epochs_dcae=20000,
    lambda_dcae=5e-2,
    use_mmd=True,
    mmd_weight=100,
    mmd_samples=512,
    mse_anneal_ratio=0.9,
    dcae_probes=3,
    analytic_dcae = True,
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


# -*- coding: utf-8 -*-
import importlib, dcae_utils
importlib.reload(dcae_utils)
from dcae_utils import (
    generate_data,
    run_training,
    plot_results,
    plot_latent_np,
    plot_hist_1d_np,
    plot_train_val_np,
    plot_latent_pseudotime_np,
    load_splatter_data,
)

import numpy as np
import pandas as pd
import plotly.io as pio
from plotly.io import to_html
from utils_evaluation import plot3D_cluster_colors, plot3D_marker_colors, table

# ---------------------------------------------------------------------
# 1) Load splatter data and train
# ---------------------------------------------------------------------
expr_path = "/home/sgrinek/PycharmProjects/DCAE/R/expression_data_biologically_realistic15gene.csv"
celltype_path = "/home/sgrinek/PycharmProjects/DCAE/R/cell_types_biologically_realistic15gene.csv"

data = load_splatter_data(
    expr_path=expr_path,
    celltype_path=celltype_path,
    val_frac=0.2,
    seed=42,
)

# Training (Phase-1 MSE, optional Phase-2 MSE+DCAE):


# ---------------------------------------------------------------------
# 2) 3D PLOTLY LATENT PLOTS (using result + splatter data)
# ---------------------------------------------------------------------
import importlib, utils_evaluation
importlib.reload(utils_evaluation)

from utils_evaluation import plot3D_cluster_colors, plot3D_marker_colors, table
# Labels = cell types
lbls = data["cell_types"]

# Expression matrix used for training (standardized)
X_input = result["data"]["X_input"]          # shape (N, m)

# Gene names (markers) from the expression CSV
expr_df = pd.read_csv(expr_path, index_col=0)
markers = list(expr_df.columns)

# Latent codes (final snapshot), shape (N, 3) since k=3
Z_final = np.asarray(result["snapshots"]["final"]["Z"], dtype=np.float32)

# Quick sanity check
if Z_final.shape[0] != X_input.shape[0]:
    raise ValueError(
        f"Row mismatch: Z_final has {Z_final.shape[0]} rows, "
        f"X_input has {X_input.shape[0]}"
    )

# Summarize labels in console (your existing helper)
table(lbls)

# Output dir for HTML
output_dir = "/home/sgrinek/PycharmProjects/DCAE/PAPERPLOTS"
epochs = int(result["config"]["epochs_dcae"])
m = int(result["config"]["m"])
k = int(result["config"]["k"])
ID = f"splatter_m{m}_k{k}_epochs{epochs}"

# Subsample size for the interactive plot (safe cap)
sub_s = min(50000, Z_final.shape[0])

# ---------- 3D latent colored by markers ----------
fig_markers = plot3D_marker_colors(
    Z_final,              # latent (N, 3)
    X_input,              # expression (N, m)
    markers,              # gene names
    sub_s=sub_s,
    lbls=lbls,
    msize=1,
)

html_str_markers = to_html(
    fig_markers,
    config=None,
    auto_play=True,
    include_plotlyjs=True,
    include_mathjax=False,
    post_script=None,
    full_html=True,
    animation_opts=None,
    default_width="100%",
    default_height="100%",
    validate=True,
)

html_path_markers = (
    f"{output_dir}/{ID}_latent3D_markers_15genes.html"
)
with open(html_path_markers, "w") as f:
    f.write(html_str_markers)
print("Saved 3D marker plot to:", html_path_markers)

# ---------- 3D latent colored by cell types (clusters) ----------
fig_clusters = plot3D_cluster_colors(Z_final, lbls=lbls)

html_str_clusters = to_html(
    fig_clusters,
    config=None,
    auto_play=True,
    include_plotlyjs=True,
    include_mathjax=False,
    post_script=None,
    full_html=True,
    animation_opts=None,
    default_width="100%",
    default_height="100%",
    validate=True,
)

html_path_clusters = (
    f"{output_dir}/{ID}_latent3D_clusters_15genes.html"
)
with open(html_path_clusters, "w") as f:
    f.write(html_str_clusters)
print("Saved 3D cluster plot to:", html_path_clusters)

# ---------------------------------------------------------------------
# old 2D / hist / curves plots (unchanged)
# ---------------------------------------------------------------------
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

# pseudotime
pt_df = pd.read_csv(
    "/home/sgrinek/PycharmProjects/DCAE/R/"
    "pseudotime_biologically_realistic15gene_batchCells_20000_.csv"
)
pseudotime = pt_df["pseudotime"].to_numpy().astype(np.float32)

plot_latent_pseudotime_np(
    Z_np=Z_final,
    pseudotime=pseudotime,
    title="Latent (final) colored by pseudotime",
    out_path=(
        "/home/sgrinek/PycharmProjects/DCAE/PAPERPLOTS/"
        "pseudotime_biologically_realistic15gene_batchCells_20000.png"
    ),
    s=1.0,
)

# Extra 2D custom plots as before
Z_mse = result["snapshots"]["mse"]["Z"]
color_1d = X_input[:, 0]

plot_latent_np(
    Z_mse,
    color_1d,
    "My custom MSE latent",
    "/home/sgrinek/PycharmProjects/DCAE/PAPERPLOTS/custom_mse_latent.png",
    s=0.01,
)
plot_latent_np(
    Z_final,
    color_1d,
    "My custom FINAL latent",
    "/home/sgrinek/PycharmProjects/DCAE/PAPERPLOTS/custom_final_latent.png",
    s=0.01,
)

plot_hist_1d_np(
    Z_final[:, 0],
    "Final z1",
    "/home/sgrinek/PycharmProjects/DCAE/PAPERPLOTS/custom_hist_z1_final.png",
    xlabel="z1",
    bins=100,
)

train_mse = np.array(result["history"]["phase1"]["train_mse"])
val_mse   = np.array(result["history"]["phase1"]["val_mse"])
plot_train_val_np(
    train_mse,
    val_mse,
    "MSE",
    "Custom P1 MSE",
    "/home/sgrinek/PycharmProjects/DCAE/PAPERPLOTS/custom_phase1_mse15.png",
)
