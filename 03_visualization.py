#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 25 10:21:17 2026

@author: gemmapostill
"""

# Install dependencies if needed:
# pip install numpy pandas matplotlib seaborn scikit-learn umap-learn joblib

from pathlib import Path
from importlib.metadata import version
import json

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import OneHotEncoder
from umap import UMAP


# ================================================================
# CUSTOMIZABLE SETTINGS - this specifies the locations from which to draw data
# ================================================================

# Specify the selected number of clusters. 
# The choice of optimal K should be based on the data outputted from "Step 2: Optimal-Cluster-Identification"
OPTIMAL_K = 5

# Folder containing results from the cluster evaluation script.
EVALUATION_DIR = Path("cluster_evaluation_outputs/overall_cohort")

# Original-scale data for descriptive profiling.
PROFILE_FILE = Path("full_cohort/preprocessed_complete.csv")

# USER: Customize the export location.
EXPORT_DIR = Path(f"cluster_visualization_outputs/overall_k{OPTIMAL_K}")
SEED = 2026

# None uses all individuals.
# Set an integer, e.g. 3000, to reduce projection computation.
# Sampling affects projections only; the heatmap uses everyone.
PROJECTION_SAMPLE_SIZE = None

TSNE_PERPLEXITY = 30
TSNE_MAX_ITER = 1000

UMAP_N_NEIGHBORS = 30
UMAP_MIN_DIST = 0.10

FIGURE_DPI = 300

# PAPER TIP:
# Select k using evaluation, clinical interpretation, and stability,
# rather than selecting the most visually separated projection.

# PAPER TIP:
# This script visualizes existing assignments; it does not refit
# clustering or change the chosen cluster membership.


# ================================================================
# IMPORT DATA AND EXISTING CLUSTER ASSIGNMENTS
# ================================================================

def read_unique_ids(path):
    data = pd.read_csv(path)
    if "person_id" not in data:
        raise ValueError(f"person_id missing from {path}")
    if data["person_id"].isna().any():
        raise ValueError(f"Missing person_id in {path}")
    if not data["person_id"].is_unique:
        raise ValueError(f"Duplicate person_id in {path}")
    return data


def load_inputs():
    assignment_file = (
        EVALUATION_DIR
        / f"k_{OPTIMAL_K:02d}"
        / "cluster_assignments.csv"
    )

    # Use the exact feature data saved during evaluation.
    feature_file = EVALUATION_DIR / "analysis_input.csv"
    configuration_file = (EVALUATION_DIR / "analysis_configuration.json")

    assignments = read_unique_ids(assignment_file)
    features = read_unique_ids(feature_file)
    profiles = read_unique_ids(PROFILE_FILE)

    with open(configuration_file) as file:
        configuration = json.load(file)

    if "cluster" not in assignments:
        raise ValueError("Cluster assignment column is missing.")

    if assignments["cluster"].isna().any():
        raise ValueError("Missing cluster assignments.")

    observed_labels = set(assignments["cluster"].unique())
    expected_labels = set(range(1, OPTIMAL_K + 1))

    if observed_labels != expected_labels:
        raise ValueError(f"Expected labels {expected_labels}; "  
                         f"found {observed_labels}.")

    if set(features["person_id"]) != set(assignments["person_id"]):
        raise ValueError(
            "Feature data and cluster assignments contain "
            "different individuals.")

    if not set(assignments["person_id"]).issubset(set(profiles["person_id"])):
        raise ValueError("Some clustered individuals are absent from PROFILE_FILE.")

    # Align all datasets to the assignment row order.
    ids = assignments["person_id"].to_numpy()
    features = (features.set_index("person_id").loc[ids].reset_index())
    profiles = (profiles.set_index("person_id").loc[ids].reset_index())

    numerical_columns = configuration["numerical_columns"]
    categorical_columns = configuration["categorical_columns"]
    gamma = float(configuration["gamma"])

    required = numerical_columns + categorical_columns

    if features[required].isna().any().any():
        raise ValueError("Missing values in projection features.")

    if gamma <= 0:
        raise ValueError("Gamma must be positive.")

    return (assignments, features, profiles, configuration, numerical_columns, categorical_columns, gamma)


# ================================================================
# 3. CONDITION-PREVALENCE HEATMAP
# ================================================================

def make_heatmap(assignments, profiles):
    """
    Heatmap values:
      cluster prevalence (%) - overall prevalence (%)

    Positive values: more prevalent than the cohort average.
    Negative values: less prevalent than the cohort average.

    Uses original 0/1 conditions, not scaled clustering values.
    """

    # The complete preprocessed dataframe contains the 32 conditions.
    # Exclude identifiers, demographics, and sex encoding explicitly.
    excluded = {"person_id", "age", "sex", "sex_female"}
    condition_columns = [col for col in profiles.columns if col not in excluded]

    if len(condition_columns) != 32:
        raise ValueError(
            "Expected 32 conditions in the complete profile file; "
            f"found {len(condition_columns)}. "
            "Specify condition_columns explicitly if your file changed.")

    for column in condition_columns:
        if not profiles[column].isin([0, 1]).all():
            raise ValueError(f"Expected complete 0/1 coding for {column}.")

    data = profiles.copy()
    data["cluster"] = assignments["cluster"].to_numpy()
    overall_prevalence = data[condition_columns].mean() * 100
    cluster_prevalence = (data.groupby("cluster")[condition_columns].mean() * 100)
    cluster_prevalence = cluster_prevalence.reindex(range(1, OPTIMAL_K + 1))
    relative_prevalence = cluster_prevalence.subtract(overall_prevalence, axis="columns")

    cluster_sizes = (data.groupby("cluster").size().reindex(range(1, OPTIMAL_K + 1)))

    # PAPER TIP:
    # Report original-scale summaries alongside relative values.
    # A large relative elevation in a rare disease can still represent
    # a small absolute prevalence. Percentage-point differences avoid
    # unstable ratios when overall prevalence is very low.
    #
    # PAPER TIP:
    # These are descriptive profiles of variables used to form
    # clusters, not independent external validation.

    overall_prevalence.rename("overall_prevalence_percent").to_csv(EXPORT_DIR / "overall_condition_prevalence.csv")
    cluster_prevalence.to_csv(EXPORT_DIR / "cluster_condition_prevalence_percent.csv")
    relative_prevalence.to_csv(EXPORT_DIR / "heatmap_difference_percentage_points.csv")
    cluster_sizes.rename("n").to_csv(EXPORT_DIR / "cluster_sizes.csv")

    # Export a tidy table for subsequent figures or reporting.
    tidy = (cluster_prevalence.rename_axis("cluster")
        .reset_index().melt(id_vars="cluster",
            var_name="condition", value_name="cluster_prevalence_percent"))

    tidy["overall_prevalence_percent"] = (tidy["condition"].map(overall_prevalence))
    tidy["difference_percentage_points"] = (tidy["cluster_prevalence_percent"] - tidy["overall_prevalence_percent"])
    tidy["cluster_n"] = tidy["cluster"].map(cluster_sizes)
    tidy.to_csv(EXPORT_DIR / "condition_profiles_long.csv", index=False)

    # Conditions as rows and clusters as columns.
    plot_data = relative_prevalence.T
    plot_data.index = [column.replace("_", " ").capitalize() for column in plot_data.index]
    plot_data.columns = [f"Cluster {cluster}\n(n={cluster_sizes.loc[cluster]:,})" for cluster in plot_data.columns]

    limit = max(float(np.abs(relative_prevalence.to_numpy()).max()), 1.0)
    fig, ax = plt.subplots(figsize=(max(8, OPTIMAL_K * 1.1), 13))
    sns.heatmap(plot_data, cmap="RdBu_r", center=0, vmin=-limit, vmax=limit, annot=True, fmt=".1f", linewidths=0.3,
                annot_kws={"fontsize": 8},cbar_kws={"label": "Difference from overall prevalence (percentage points)"}, ax=ax)
    ax.set(title=(f"Condition profiles for k={OPTIMAL_K}\n"
                  "Red: above cohort mean; blue: below cohort mean"), xlabel="", ylabel="")
    ax.tick_params(axis="x", rotation=0)
    ax.tick_params(axis="y", rotation=0)
    fig.tight_layout()
    save_figure(fig, "condition_heatmap_relative_to_mean")


# ================================================================
# REPRESENT MIXED FEATURES FOR PROJECTION
# ================================================================

def make_projection_representation(features, numerical_columns, categorical_columns,
    gamma
):
    """
    Construct a Euclidean representation preserving the evaluation
    script's mixed distance:

    d² = sum(numerical differences²)
         + gamma * categorical mismatch count

    Full one-hot encoding produces squared distance 2 for a category
    mismatch, so multiply encoded categories by sqrt(gamma / 2).

    Numerical variables retain their evaluation-input scaling.
    """

    numerical = features[numerical_columns].to_numpy(dtype=float)
    encoder = OneHotEncoder(sparse_output=False, handle_unknown="error", dtype=np.float64)
    encoded_categories = encoder.fit_transform(features[categorical_columns])
    weighted_categories = (encoded_categories * np.sqrt(gamma / 2))
    representation = np.column_stack([numerical, weighted_categories])

    if not np.isfinite(representation).all():
        raise ValueError("Non-finite values in projection matrix.")

    names = (numerical_columns + encoder.get_feature_names_out(categorical_columns).tolist())

    # PAPER TIP:
    # Encoding can implicitly change feature weights. This temporary
    # weighted encoding preserves the chosen mixed distance rather
    # than adding extra weight simply by expanding categories.
    #
    # Do not standardize these encoded columns again: that would
    # change gamma's interpretation and upweight rare categories.

    pd.DataFrame({"projection_feature": names}).to_csv(EXPORT_DIR / "projection_feature_names.csv", index=False)
    joblib.dump(encoder, EXPORT_DIR / "projection_category_encoder.joblib")

    return representation, names


# ================================================================
# FIGURE HELPERS
# ================================================================

def save_figure(fig, name):
    fig.savefig(EXPORT_DIR / f"{name}.png", dpi=FIGURE_DPI, bbox_inches="tight")
    fig.savefig(EXPORT_DIR / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def draw_projection(ax, coordinates, labels, title, x_label, y_label):
    colors = sns.color_palette("tab20", n_colors=OPTIMAL_K)

    # Randomize drawing order so the last numbered cluster does not
    # systematically obscure the others.
    rng = np.random.default_rng(SEED)
    order = rng.permutation(len(labels))
    point_colors = np.array([colors[int(cluster) - 1] for cluster in labels])
    ax.scatter(coordinates[order, 0],  coordinates[order, 1], c=point_colors[order], s=10, alpha=0.45, linewidths=0, rasterized=True)

    # Proxy handles provide a consistent legend across all methods.
    for cluster in range(1, OPTIMAL_K + 1):
        ax.scatter([], [], color=colors[cluster - 1], label=f"Cluster {cluster}", s=25)

    ax.set(title=title, xlabel=x_label, ylabel=y_label)
    ax.grid(alpha=0.15)


# ================================================================
# 6. PCA, t-SNE, AND UMAP
# ================================================================

def run_projections(representation, feature_names, assignments):
    rng = np.random.default_rng(SEED)

    sample_n = (
        len(assignments)
        if PROJECTION_SAMPLE_SIZE is None
        else min(PROJECTION_SAMPLE_SIZE, len(assignments)))

    if sample_n < 4:
        raise ValueError("At least four individuals are required.")

    indices = np.sort(rng.choice(len(assignments), size=sample_n, replace=False))

    x = representation[indices]
    sampled_assignments = (assignments.iloc[indices][["person_id", "cluster"]].reset_index(drop=True))
    sampled_assignments.to_csv(EXPORT_DIR / "projection_sample.csv", index=False)
    labels = sampled_assignments["cluster"].to_numpy()

    # PAPER TIP:
    # Two-dimensional projections compress high-dimensional data.
    # Apparent overlap does not prove clustering failed; apparent
    # separation does not establish clinically meaningful subgroups.
    #
    # PAPER TIP:
    # PCA, t-SNE, and UMAP are used here for VISUALIZATION ONLY.
    # They are not inputs to the existing clustering solution.
    #
    # t-SNE/UMAP layouts depend on tuning and random initialization.
    # Their axes have no direct clinical units; avoid interpreting
    # apparent cluster area or spacing as definitive evidence.

    print(f"Running PCA for {sample_n:,} individuals...", flush=True)

    # Fit PCA on the full cohort, then plot the common sample.
    pca = PCA(n_components=2, svd_solver="full")
    pca.fit(representation)
    pca_coordinates = pca.transform(x)

    joblib.dump(pca, EXPORT_DIR / "pca_model.joblib")

    explained = pd.DataFrame({
        "component": ["PC1", "PC2"],
        "explained_variance_ratio": pca.explained_variance_ratio_,
        "explained_variance_percent": (100 * pca.explained_variance_ratio_),
        "cumulative_explained_variance_percent": (
            100 * pca.explained_variance_ratio_.cumsum()
        )
    })

    explained.to_csv(EXPORT_DIR / "pca_explained_variance.csv", index=False)

    # PCA loadings describe the weighted representation, not the
    # original unscaled disease variables.
    pd.DataFrame(pca.components_.T, index=pd.Index(feature_names, name="feature"), columns=["PC1_loading", "PC2_loading"]).to_csv(EXPORT_DIR / "pca_loadings.csv")
    perplexity = min(float(TSNE_PERPLEXITY), max(2.0, (sample_n - 1) / 3))
    print("Running t-SNE...", flush=True)

    tsne = TSNE(n_components=2, perplexity=perplexity, learning_rate="auto", init="pca", max_iter=TSNE_MAX_ITER, metric="euclidean", random_state=SEED)
    tsne_coordinates = tsne.fit_transform(x)

    # Standard sklearn t-SNE does not provide a transform method
    # for projecting new individuals into this fitted layout.
    joblib.dump(tsne, EXPORT_DIR / "tsne_model.joblib")
    neighbors = min(UMAP_N_NEIGHBORS, sample_n - 1)
    print("Running UMAP...", flush=True)

    umap = UMAP(n_components=2, n_neighbors=neighbors, min_dist=UMAP_MIN_DIST, metric="euclidean", random_state=SEED, n_jobs=1)
    umap_coordinates = umap.fit_transform(x)
    joblib.dump(umap, EXPORT_DIR / "umap_model.joblib")
    projections = {"PCA": pca_coordinates, "tSNE": tsne_coordinates, "UMAP": umap_coordinates}
    combined = sampled_assignments.copy()

    for name, coordinates in projections.items():
        export = sampled_assignments.copy()
        export[f"{name}_1"] = coordinates[:, 0]
        export[f"{name}_2"] = coordinates[:, 1]
        export.to_csv(EXPORT_DIR / f"{name.lower()}_coordinates.csv", index=False)

        combined[f"{name}_1"] = coordinates[:, 0]
        combined[f"{name}_2"] = coordinates[:, 1]

    combined.to_csv(EXPORT_DIR / "all_projection_coordinates.csv", index=False)

    pc1_percent, pc2_percent = (100 * pca.explained_variance_ratio_)

    axis_labels = {
        "PCA": (f"PC1 ({pc1_percent:.1f}% variance)", f"PC2 ({pc2_percent:.1f}% variance)"),
        "tSNE": ("t-SNE 1", "t-SNE 2"),
        "UMAP": ("UMAP 1", "UMAP 2")
    }

    # Individual figures.
    for name, coordinates in projections.items():
        fig, ax = plt.subplots(figsize=(8, 6))
        draw_projection(ax, coordinates,  labels, title=f"{name}: k={OPTIMAL_K}, n={sample_n:,}", x_label=axis_labels[name][0], y_label=axis_labels[name][1])
        ax.legend(title="Cluster", bbox_to_anchor=(1.02, 1), loc="upper left")
        fig.tight_layout()
        save_figure(fig, f"{name.lower()}_projection")

    # Combined figure.
    fig, axes = plt.subplots(1, 3, figsize=(19, 6))

    for ax, (name, coordinates) in zip(axes, projections.items()):
        draw_projection(ax, coordinates, labels, title=name, x_label=axis_labels[name][0], y_label=axis_labels[name][1])

    handles, legend_labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, title="Cluster", loc="center right")
    fig.suptitle(f"Projections of the selected k={OPTIMAL_K} solution "
                 f"(n={sample_n:,})")
    fig.tight_layout(rect=[0, 0, 0.93, 0.94])
    save_figure(fig, "projections_three_panel")

    return {
        "projection_sample_n": sample_n,
        "tsne_perplexity_used": perplexity,
        "tsne_kl_divergence": float(tsne.kl_divergence_),
        "umap_neighbors_used": neighbors,
        "pca_two_component_variance_percent": float(
            100 * pca.explained_variance_ratio_.sum()
        )
    }


# ================================================================
# 7RUN AND SAVE ANALYSIS SETTINGS
# ================================================================

def main():
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="white", context="notebook")
    (assignments, features, profiles, evaluation_configuration, numerical_columns, categorical_columns, gamma) = load_inputs()

    # Save selected assignments alongside visualizations.
    assignments.to_csv(EXPORT_DIR / "selected_cluster_assignments.csv", index=False)

    print("Creating condition heatmap...", flush=True)
    make_heatmap(assignments, profiles)

    representation, feature_names = (make_projection_representation(features, numerical_columns, categorical_columns, gamma))
    projection_settings = run_projections(representation, feature_names, assignments)

    configuration = {
        "optimal_k": OPTIMAL_K,
        "evaluation_directory": str(EVALUATION_DIR.resolve()),
        "profile_file": str(PROFILE_FILE.resolve()),
        "n_clustered": len(assignments),
        "seed": SEED,
        "gamma_from_evaluation": gamma,
        "numerical_columns": numerical_columns,
        "categorical_columns": categorical_columns,
        "heatmap_definition": ("Cluster prevalence minus overall cohort prevalence, "
                               "in percentage points"),
        "projection_representation": ("Original evaluation numerical features plus full "
                                      "one-hot categorical features weighted by sqrt(gamma/2)"),
        "tsne_max_iter": TSNE_MAX_ITER,
        "umap_min_dist": UMAP_MIN_DIST,
        **projection_settings,
        "package_versions": {
            package: version(package)
            for package in [ "numpy", "pandas", "matplotlib",  "seaborn",  "scikit-learn", "umap-learn", "joblib"]
        }
    }

    with open(EXPORT_DIR / "visualization_configuration.json", "w") as file:
        json.dump(configuration, file, indent=2)

    # Preserve the evaluation settings defining the original solution.
    with open(EXPORT_DIR / "source_evaluation_configuration.json", "w") as file:
        json.dump(evaluation_configuration, file, indent=2)

    print(f"\nResults saved to: {EXPORT_DIR.resolve()}")


if __name__ == "__main__":
    main()