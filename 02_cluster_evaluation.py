#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 25 10:11:39 2026

@author: gemmapostill
"""

# Install dependencies if needed:
# pip install numpy pandas matplotlib scikit-learn kmodes joblib

from pathlib import Path
from importlib.metadata import version
import json
import time
import warnings

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from kmodes.kprototypes import KPrototypes
from sklearn.cluster import OPTICS
from sklearn.metrics import pairwise_distances, silhouette_score


# ================================================================
# 1. CUSTOMIZABLE FILE PATHS AND SETTINGS
# ================================================================

INPUT_FILE = Path("full_cohort/features_age_minmax_binary01.csv")

# Original-scale data used ONLY to describe the resulting clusters.
PROFILE_FILE = Path("full_cohort/preprocessed_complete.csv")

# CHANGE THIS to your preferred export location.
EXPORT_DIR = Path("cluster_evaluation_outputs/overall_cohort")

K_VALUES = range(2, 13)

SEED = 2026
N_INIT = 10
MAX_ITER = 100
N_JOBS = 1

# Explicit, fixed weighting across all k values.
# With min–max scaled age, gamma=1 means that one categorical
# mismatch contributes 1 to the objective; the squared age
# difference contributes between 0 and 1.
#
# PAPER TIP: Variable representation and weighting can change the
# clusters. Repeat the analysis with other gamma values, saving
# each analysis to a separate export folder.
GAMMA = 1.0

# Pairwise distances require O(m²) memory.
# The SAME random sample is used for silhouette at every k.
# All individuals are used to fit k-prototypes and calculate
# cost, cohesion, DBI, cluster sizes, and profiles.
EVALUATION_SAMPLE_SIZE = 2000

# OPTICS is exploratory and runs on the same sampled individuals.
# Set EVALUATION_SAMPLE_SIZE = None to use the full dataset,
# recognizing the additional memory and computational burden.
OPTICS_MIN_SAMPLES = 30
OPTICS_MIN_CLUSTER_SIZE = 0.05
OPTICS_XI = 0.05

CONDITION_COLUMNS = [
    "hypertension",
    "ischemic_heart_disease",
    "heart_failure",
    "atrial_fibrillation",
    "stroke_history",
    "peripheral_arterial_disease",
    "valvular_heart_disease",
    "cardiomyopathy",
    "rheumatoid_arthritis",
    "psoriasis",
    "psoriatic_arthritis",
    "inflammatory_bowel_disease",
    "systemic_lupus",
    "multiple_sclerosis",
    "celiac_disease",
    "autoimmune_thyroid_disease",
    "type_2_diabetes",
    "dyslipidemia",
    "chronic_kidney_disease",
    "asthma",
    "copd",
    "osteoarthritis",
    "osteoporosis",
    "chronic_back_pain",
    "depressive_disorder",
    "anxiety_disorder",
    "migraine",
    "epilepsy",
    "dementia",
    "parkinsons_disease",
    "cancer_history",
    "gastroesophageal_reflux",
]


# ================================================================
# 2. IMPORT AND VALIDATE PREPROCESSED DATA
# ================================================================

def load_data():
    data = pd.read_csv(INPUT_FILE)

    if "person_id" not in data:
        raise ValueError("Input must contain person_id.")

    if data["person_id"].isna().any():
        raise ValueError("person_id contains missing values.")

    if not data["person_id"].is_unique:
        raise ValueError("person_id must be unique.")

    features = data.drop(columns="person_id").copy()

    if features.isna().any().any():
        raise ValueError("Missing values found in clustering features.")

    # Do not infer measurement type from storage dtype.
    # Conditions remain categorical even though stored as integers.
    categorical_columns = [col for col in CONDITION_COLUMNS + ["sex_female"] if col in features.columns]
    numerical_columns = [col for col in ["age"] if col in features.columns]
    unknown = set(features.columns) - set(categorical_columns + numerical_columns)

    if unknown:
        raise ValueError(f"Unexpected features: {sorted(unknown)}")

    if not categorical_columns or not numerical_columns:
        raise ValueError(
            "K-prototypes requires both numerical and categorical features. If age was excluded during preprocessing,"
            "consider k-modes for the remaining categorical data.")

    for col in categorical_columns:
        if not features[col].isin([0, 1]).all():
            raise ValueError(f"{col} must contain only 0/1.")

    numerical = features[numerical_columns].to_numpy(dtype=float)
    categorical = features[categorical_columns].to_numpy(dtype=int)

    if not np.isfinite(numerical).all():
        raise ValueError("Non-finite numerical values found.")

    # This script assumes the selected min–max age export.
    if not ((numerical >= 0) & (numerical <= 1)).all():
        raise ValueError("Expected min–max scaled age in [0, 1]. "
                         "Check INPUT_FILE and reconsider gamma if changing scaling.")

    # Numerical columns first, categorical columns second.
    feature_order = numerical_columns + categorical_columns
    mixed = np.empty((len(features), len(feature_order)), dtype=object)
    mixed[:, :len(numerical_columns)] = numerical
    mixed[:, len(numerical_columns):] = categorical

    categorical_indices = list(range(len(numerical_columns), len(feature_order)))

    return (data, numerical,
        categorical,
  mixed, numerical_columns, categorical_columns, feature_order, categorical_indices)


# ================================================================
# 3. DEFINE A CONSISTENT MIXED DISTANCE
# ================================================================

# K-prototypes minimizes:
#
# q(x, prototype) =
#     sum of squared numerical differences
#     + gamma * number of categorical mismatches
#
# For silhouette, OPTICS, cohesion, and mixed DBI, use:
#
# d(x, y) = sqrt(q(x, y))
#
# The square root provides a Euclidean-equivalent distance:
# each category can be represented by a suitably weighted one-hot
# vector. No explicit one-hot encoding is required here.
#
# PAPER TIP: Distance and encoding choices affect cluster structure.
# Matching dissimilarity counts shared absences as agreement.
# Consider a separate sensitivity analysis if shared absences
# should contribute less to similarity.


def mixed_distances(num_a, cat_a, num_b=None, cat_b=None):
    """Pairwise square-root k-prototypes dissimilarity."""

    if num_b is None:
        num_b = num_a
        cat_b = cat_a
        same_data = True
    else:
        same_data = False

    numerical_squared = pairwise_distances(
        num_a, num_b, metric="sqeuclidean"
    )

    # Hamming is the PROPORTION of mismatching columns.
    # Multiply by the number of columns to recover mismatch counts.
    mismatches = pairwise_distances(
        cat_a, cat_b, metric="hamming"
    ) * cat_a.shape[1]

    distances = np.sqrt(
        np.maximum(numerical_squared + GAMMA * mismatches, 0)
    )

    if same_data:
        np.fill_diagonal(distances, 0.0)

    return distances


# ================================================================
# 4. MIXED-DISTANCE DBI
# ================================================================

def calculate_mixed_dbi(
    numerical, categorical, labels, proto_num, proto_cat
):
    """
    Davies–Bouldin-style index using mixed prototypes.

    S_i = mean mixed distance to prototype i.
    M_ij = mixed distance between prototypes i and j.
    DBI = mean_i max_j [(S_i + S_j) / M_ij].

    This is a mixed-distance generalization, not sklearn's
    conventional numerical-centroid DBI.
    """

    k = len(proto_num)

    assigned_squared = (
        ((numerical - proto_num[labels]) ** 2).sum(axis=1)
        + GAMMA
        * (categorical != proto_cat[labels]).sum(axis=1)
    )

    assigned_distances = np.sqrt(
        np.maximum(assigned_squared, 0)
    )

    scatter = np.array([
        assigned_distances[labels == cluster].mean()
        for cluster in range(k)
    ])

    separation = mixed_distances(
        proto_num, proto_cat
    )

    off_diagonal = ~np.eye(k, dtype=bool)

    # Identical prototypes imply zero separation.
    if np.any(separation[off_diagonal] <= 1e-12):
        return np.inf, assigned_squared, assigned_distances

    ratios = np.full((k, k), -np.inf)

    numerator = scatter[:, None] + scatter[None, :]

    ratios[off_diagonal] = (
        numerator[off_diagonal] / separation[off_diagonal]
    )

    dbi = ratios.max(axis=1).mean()

    return float(dbi), assigned_squared, assigned_distances


# ================================================================
# 5. OPTICS EXPLORATION
# ================================================================

def run_optics(distance_matrix, sampled_ids):
    """
    Generate an exploratory reachability plot.

    PAPER TIP: Density exploration can inform algorithm selection.
    Valleys suggest relatively dense regions, but do not establish
    a uniquely correct number of k-prototypes clusters.
    """

    if len(sampled_ids) <= OPTICS_MIN_SAMPLES:
        raise ValueError("OPTICS sample is too small.")

    optics = OPTICS(
        min_samples=OPTICS_MIN_SAMPLES,
        min_cluster_size=OPTICS_MIN_CLUSTER_SIZE,
        xi=OPTICS_XI,
        metric="precomputed",
        cluster_method="xi",
        n_jobs=N_JOBS
    )

    optics.fit(distance_matrix)

    order = optics.ordering_

    reachability = pd.DataFrame({
        "ordering_position": np.arange(len(order)),
        "person_id": sampled_ids[order],
        "reachability": optics.reachability_[order],
        "core_distance": optics.core_distances_[order],
        "optics_cluster": optics.labels_[order]
    })

    reachability.to_csv(
        EXPORT_DIR / "optics_reachability.csv", index=False
    )

    joblib.dump(optics, EXPORT_DIR / "optics_model.joblib")

    # Infinite reachability is retained in CSV, displayed as a gap.
    plotted_reachability = reachability["reachability"].replace(
        [np.inf, -np.inf], np.nan
    )

    fig, ax = plt.subplots(figsize=(12, 5))

    ax.plot(
        reachability["ordering_position"],
        plotted_reachability,
        color="steelblue",
        linewidth=0.8
    )

    ax.set(
        xlabel="OPTICS ordering",
        ylabel="Reachability: mixed distance",
        title=f"OPTICS reachability plot (n={len(order):,})"
    )

    ax.grid(alpha=0.2)
    fig.tight_layout()

    fig.savefig(
        EXPORT_DIR / "optics_reachability.png", dpi=300
    )
    fig.savefig(EXPORT_DIR / "optics_reachability.pdf")
    plt.close(fig)


# ================================================================
# 6. OPTIONAL ORIGINAL-SCALE CLUSTER PROFILING
# ================================================================

def save_cluster_profiles(assignments, folder):
    if not PROFILE_FILE.exists():
        warnings.warn(
            "PROFILE_FILE not found; original-scale profiles skipped."
        )
        return

    profile = pd.read_csv(PROFILE_FILE)

    if (
        "person_id" not in profile
        or profile["person_id"].isna().any()
        or not profile["person_id"].is_unique
    ):
        raise ValueError("Invalid identifiers in PROFILE_FILE.")

    joined = assignments.merge(
        profile,
        on="person_id",
        how="left",
        validate="one_to_one",
        indicator=True
    )

    if not joined["_merge"].eq("both").all():
        raise ValueError("Some clustered IDs are absent from profiles.")

    conditions = [
        col for col in CONDITION_COLUMNS if col in joined
    ]

    for col in conditions:
        if not joined[col].isin([0, 1]).all():
            raise ValueError(f"Profile condition {col} is not 0/1.")

    if conditions:
        prevalence = joined.groupby("cluster")[conditions].mean() * 100
        prevalence.to_csv(folder / "condition_prevalence_percent.csv")

    if "age" in joined:
        joined.groupby("cluster")["age"].agg(
            ["count", "mean", "std", "median", "min", "max"]
        ).to_csv(folder / "age_profiles_original_scale.csv")

    if "sex" in joined:
        sex_percent = pd.crosstab(
            joined["cluster"],
            joined["sex"],
            normalize="index"
        ) * 100

        sex_percent.to_csv(folder / "sex_profiles_percent.csv")


# ================================================================
# 7. FIT K-PROTOTYPES FOR k = 2–12
# ================================================================

def evaluate_k_values(
    data,
    numerical,
    categorical,
    mixed,
    numerical_columns,
    categorical_columns,
    feature_order,
    categorical_indices,
    sample_indices,
    sample_distances
):
    results = []
    all_assignments = data[["person_id"]].copy()

    for k in K_VALUES:
        print(f"Fitting k={k}...", flush=True)

        folder = EXPORT_DIR / f"k_{k:02d}"
        folder.mkdir(parents=True, exist_ok=True)

        started = time.perf_counter()

        # PAPER TIP: Multiple initializations reduce dependence on
        # a single starting solution, but are not a substitute for
        # bootstrap/subsample stability assessment.
        model = KPrototypes(
            n_clusters=k,
            init="Huang",
            n_init=N_INIT,
            max_iter=MAX_ITER,
            gamma=GAMMA,
            random_state=SEED + k,
            n_jobs=N_JOBS,
            verbose=0
        )

        labels = model.fit_predict(
            mixed,
            categorical=categorical_indices
        ).astype(int)

        if len(np.unique(labels)) != k:
            raise RuntimeError(
                f"k={k} produced fewer than k occupied clusters."
            )

        # kmodes returns numerical centroids first, then
        # categorical centroids.
        centroids = model.cluster_centroids_

        number_numerical = len(numerical_columns)

        proto_num = np.asarray(
            centroids[:, :number_numerical], dtype=float
        )

        proto_cat = np.asarray(
            centroids[:, number_numerical:], dtype=float
        ).astype(int)

        dbi, assigned_squared, assigned_distances = (
            calculate_mixed_dbi(
                numerical,
                categorical,
                labels,
                proto_num,
                proto_cat
            )
        )

        # Check that the implemented objective matches model cost.
        if not np.isclose(
            assigned_squared.sum(),
            model.cost_,
            rtol=1e-6,
            atol=1e-6
        ):
            raise RuntimeError(
                "Distance calculation does not match model cost."
            )

        # Same sampled people and distance matrix for every k.
        sampled_labels = labels[sample_indices]
        sample_clusters_n = len(np.unique(sampled_labels))

        if 1 < sample_clusters_n < len(sample_indices):
            silhouette = silhouette_score(
                sample_distances,
                sampled_labels,
                metric="precomputed"
            )
        else:
            silhouette = np.nan

        cluster_sizes = np.bincount(labels, minlength=k)

        results.append({
            "k": k,
            "dbi_mixed": dbi,
            "cost": float(model.cost_),
            "mean_squared_distance": float(assigned_squared.mean()),
            "silhouette_sample": float(silhouette),
            "cohesion_mean_distance": float(
                assigned_distances.mean()
            ),
            "minimum_cluster_n": int(cluster_sizes.min()),
            "minimum_cluster_percent": float(
                100 * cluster_sizes.min() / len(labels)
            ),
            "maximum_cluster_n": int(cluster_sizes.max()),
            "silhouette_sample_n": len(sample_indices),
            "clusters_represented_in_silhouette_sample": sample_clusters_n,
            "iterations": int(model.n_iter_),
            "runtime_seconds": time.perf_counter() - started
        })

        # Save assignments using 1-based labels for reporting.
        assignments = data[["person_id"]].copy()
        assignments["cluster"] = labels + 1
        assignments["distance_to_prototype"] = assigned_distances

        assignments.to_csv(
            folder / "cluster_assignments.csv", index=False
        )

        all_assignments[f"cluster_k{k}"] = labels + 1

        sizes = pd.DataFrame({
            "cluster": np.arange(1, k + 1),
            "n": cluster_sizes,
            "percent": 100 * cluster_sizes / len(labels)
        })

        sizes.to_csv(folder / "cluster_sizes.csv", index=False)

        prototype_table = pd.DataFrame(
            np.column_stack([proto_num, proto_cat]),
            columns=feature_order
        )

        prototype_table.insert(
            0, "cluster", np.arange(1, k + 1)
        )

        prototype_table.to_csv(
            folder / "prototypes_input_scale.csv", index=False
        )

        # Group-specific cohesion reveals whether a good overall
        # average conceals poorly cohesive smaller clusters.
        assignments.groupby("cluster").agg(
            n=("person_id", "size"),
            mean_distance=("distance_to_prototype", "mean"),
            median_distance=("distance_to_prototype", "median"),
            maximum_distance=("distance_to_prototype", "max")
        ).to_csv(folder / "cohesion_by_cluster.csv")

        joblib.dump(model, folder / "kprototypes_model.joblib")

        save_cluster_profiles(assignments, folder)

        # Checkpoint results after each completed k.
        pd.DataFrame(results).to_csv(
            EXPORT_DIR / "evaluation_metrics.csv", index=False
        )

        print(
            f"  cost={model.cost_:.2f}; "
            f"DBI={dbi:.3f}; "
            f"silhouette={silhouette:.3f}",
            flush=True
        )

    all_assignments.to_csv(
        EXPORT_DIR / "cluster_assignments_all_k.csv",
        index=False
    )

    return pd.DataFrame(results)


# ================================================================
# 8. SAVE EVALUATION PLOTS
# ================================================================

def plot_evaluation(metrics):
    specifications = [
        ("dbi_mixed", "Mixed-distance DBI", "Lower is better"),
        ("cost", "K-prototypes cost", "Lower; inspect elbow"),
        (
            "silhouette_sample",
            "Silhouette on fixed sample",
            "Higher is better"
        ),
        (
            "cohesion_mean_distance",
            "Cohesion: mean distance to prototype",
            "Lower is better"
        )
    ]

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    for ax, (column, title, direction) in zip(
        axes.flat, specifications
    ):
        values = metrics[column].replace(
            [np.inf, -np.inf], np.nan
        )

        ax.plot(
            metrics["k"], values,
            marker="o", color="steelblue"
        )

        ax.set(
            xlabel="Number of clusters (k)",
            ylabel=title,
            title=f"{title}\n{direction}"
        )

        ax.set_xticks(metrics["k"])
        ax.grid(alpha=0.25)

    fig.suptitle(
        "Overall cohort: k-prototypes evaluation",
        fontsize=15
    )

    fig.tight_layout()

    fig.savefig(
        EXPORT_DIR / "evaluation_four_panel.png", dpi=300
    )
    fig.savefig(EXPORT_DIR / "evaluation_four_panel.pdf")
    plt.close(fig)

    # Also save each plot individually.
    for column, title, direction in specifications:
        fig, ax = plt.subplots(figsize=(7, 5))

        values = metrics[column].replace(
            [np.inf, -np.inf], np.nan
        )

        ax.plot(metrics["k"], values, marker="o")

        ax.set(
            xlabel="Number of clusters (k)",
            ylabel=title,
            title=f"{title}: {direction}"
        )

        ax.set_xticks(metrics["k"])
        ax.grid(alpha=0.25)
        fig.tight_layout()

        fig.savefig(EXPORT_DIR / f"{column}.png", dpi=300)
        fig.savefig(EXPORT_DIR / f"{column}.pdf")
        plt.close(fig)


# ================================================================
# 9. RUN ANALYSIS
# ================================================================

def main():
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)

    if GAMMA <= 0:
        raise ValueError("GAMMA must be positive.")

    (
        data,
        numerical,
        categorical,
        mixed,
        numerical_columns,
        categorical_columns,
        feature_order,
        categorical_indices
    ) = load_data()

    if len(data) <= max(K_VALUES):
        raise ValueError("Too few individuals for requested k values.")

    rng = np.random.default_rng(SEED)

    sample_n = (
        len(data)
        if EVALUATION_SAMPLE_SIZE is None
        else min(EVALUATION_SAMPLE_SIZE, len(data))
    )

    sample_indices = np.sort(
        rng.choice(len(data), size=sample_n, replace=False)
    )

    sampled_ids = data["person_id"].to_numpy()[sample_indices]

    pd.DataFrame({
        "person_id": sampled_ids,
        "input_row_position": sample_indices
    }).to_csv(
        EXPORT_DIR / "evaluation_sample.csv", index=False
    )

    # Save exactly the input used for this analysis.
    data.to_csv(
        EXPORT_DIR / "analysis_input.csv", index=False
    )

    configuration = {
        "input_file": str(INPUT_FILE.resolve()),
        "profile_file": str(PROFILE_FILE.resolve()),
        "n": len(data),
        "k_values": list(K_VALUES),
        "seed": SEED,
        "n_init": N_INIT,
        "max_iter": MAX_ITER,
        "gamma": GAMMA,
        "numerical_columns": numerical_columns,
        "categorical_columns": categorical_columns,
        "feature_order": feature_order,
        "categorical_indices": categorical_indices,
        "evaluation_sample_n": sample_n,
        "distance": (
            "sqrt(sum squared numerical differences "
            "+ gamma * categorical mismatch count)"
        ),
        "dbi_definition": "Mixed-prototype generalization",
        "cohesion_definition": "Mean distance to assigned prototype",
        "optics_min_samples": OPTICS_MIN_SAMPLES,
        "optics_min_cluster_size": OPTICS_MIN_CLUSTER_SIZE,
        "optics_xi": OPTICS_XI,
        "package_versions": {
            name: version(name)
            for name in [
                "numpy", "pandas", "scikit-learn",
                "kmodes", "matplotlib", "joblib"
            ]
        }
    }

    with open(EXPORT_DIR / "analysis_configuration.json", "w") as file:
        json.dump(configuration, file, indent=2)

    print(
        f"Calculating mixed distances for {sample_n:,} people...",
        flush=True
    )

    sample_distances = mixed_distances(
        numerical[sample_indices],
        categorical[sample_indices]
    )

    # Sample IDs specify the row/column order of this matrix.
    np.savez_compressed(
        EXPORT_DIR / "evaluation_distance_matrix.npz",
        distances=sample_distances
    )

    run_optics(sample_distances, sampled_ids)

    metrics = evaluate_k_values(
        data,
        numerical,
        categorical,
        mixed,
        numerical_columns,
        categorical_columns,
        feature_order,
        categorical_indices,
        sample_indices,
        sample_distances
    )

    plot_evaluation(metrics)

    # PAPER TIP: Do not select k solely from one numerical index.
    # Review:
    # - Agreement across evaluation metrics;
    # - Small or highly imbalanced clusters;
    # - Clinical interpretation of condition profiles;
    # - Sensitivity to scaling, gamma, and variable selection;
    # - Stability under resampling;
    # - External validation using information not used for clustering.
    #
    # PAPER TIP: Correlated conditions can implicitly upweight a
    # clinical domain. Retain/remove them based on the research
    # question and assess sensitivity, rather than automatically
    # deleting the relationships you hope to discover.
    #
    # PAPER TIP: Cluster labels are provisional analytic constructs,
    # not established clinical categories.
    #
    # This script evaluates candidate solutions. It does not perform
    # resampling stability analysis or external validation.

    print("\nEvaluation metrics:")
    print(metrics.round(3).to_string(index=False))
    print(f"\nResults saved to: {EXPORT_DIR.resolve()}")


if __name__ == "__main__":
    main()
    
    
#  Interpretation: lower DBI and mean distance suggest more compact, separated clusters; higher silhouette suggests better separation. 
# Cost generally falls as k increases,  so inspect its elbow alongside cluster profiles. Cost and cohesion are related measures, not independent evidence. 
# With 32 mostly uncommon conditions, also check whether categorical modes remain mostly zero and whether the resulting groups primarily separate by age or overall disease burden.


## As a sensitivity analysis, we recommend separately assessing whether the same clusters are present in males or females only 
