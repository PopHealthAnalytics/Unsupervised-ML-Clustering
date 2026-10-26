#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 25 12:22:53 2026

@author: gemmapostill
"""

### Note this script is designed as an exemplar senstivity analysis to understand teh within sample generalizability of the clustering structures identified 


# Dependencies:
# pip install numpy pandas matplotlib seaborn scikit-learn \
#             kmodes umap-learn joblib

from pathlib import Path
import json

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from kmodes.kprototypes import KPrototypes
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import (
    adjusted_rand_score,
    adjusted_mutual_info_score
)
from sklearn.preprocessing import OneHotEncoder
from umap import UMAP


# ================================================================
# USER SETTINGS
# ================================================================

# USER: Selected number of clusters.
OPTIMAL_K = 5

# Optional: specify a different k for individual cohorts.
# Leave empty to use OPTIMAL_K for all cohorts.
K_BY_COHORT = {
    # "female_age18_44": 4,
    # "female_age45plus": 5,
    # "male_age18_44": 4,
    # "male_age45plus": 5,
}

PREPROCESSING_DIR = Path("")
OVERALL_EVALUATION_DIR = Path("cluster_evaluation_outputs/overall_cohort")

# Selected k for the previously fitted overall solution.
OVERALL_K = OPTIMAL_K

# USER: Customize export location.
EXPORT_DIR = Path("sensitivity_outputs/age_sex_stratified")
COHORTS = ["female_age18_44", "female_age45plus", "male_age18_44", "male_age45plus"]

SEED = 2026
N_INIT = 10
MAX_ITER = 100

# None uses everyone; specify an integer to sample projections only.
PROJECTION_SAMPLE_SIZE = None

TSNE_PERPLEXITY = 30
TSNE_MAX_ITER = 1000
UMAP_N_NEIGHBORS = 30
UMAP_MIN_DIST = 0.10

CONDITIONS = [
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
    "gastroesophageal_reflux"
]

# PAPER TIP:
# Stratification changes the population being clustered.
# Document the population definition, selected features, scaling,
# weighting, and algorithm settings.
#
# This analysis uses the independently preprocessed subgroup exports.
# Consequently, age scaling differs across subgroups and sex is
# constant. This evaluates the complete subgroup-refitting pipeline,
# rather than isolating population composition alone.


# ================================================================
# DEFINE GENERAL HELPER FUNCTIONS THAT WILL BE USED THROUGHOUT 
# ================================================================

def read_data(path):
    data = pd.read_csv(path)

    if "person_id" not in data:
        raise ValueError(f"person_id missing from {path}")

    if data["person_id"].isna().any():
        raise ValueError(f"Missing person_id in {path}")

    if not data["person_id"].is_unique:
        raise ValueError(f"Duplicate person_id in {path}")

    return data


def align(data, ids):

    if not set(ids).issubset(set(data["person_id"])):
        raise ValueError("Required individuals are missing.")

    return data.set_index("person_id").loc[ids].reset_index()

def save_figure(fig, folder, name):
    fig.savefig(folder / f"{name}.png", dpi=300, bbox_inches="tight")
    fig.savefig(folder / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)

def plot_heatmap(table, folder, name, title, relative=False):
    options = {}

    if relative:
        limit = max(float(np.abs(table.to_numpy()).max()), 1.0)
        options = {"center": 0, "vmin": -limit, "vmax": limit}

    fig, ax = plt.subplots(figsize=(max(8, table.shape[1] * 1.1), max(5, table.shape[0] * 0.34)))
    sns.heatmap(table, cmap="RdBu_r" if relative else "viridis", annot=True, fmt=".1f", linewidths=0.3, annot_kws={"fontsize": 8}, 
                cbar_kws={"label": ( "Prevalence difference (percentage points)" if relative else "Number of individuals")}, ax=ax, **options)
    ax.set_title(title)
    ax.tick_params(axis="y", rotation=0)
    fig.tight_layout()
    save_figure(fig, folder, name)


# ================================================================
# LOAD OVERALL DATA AND ASSIGNMENTS
# ================================================================

def load_overall():
    profiles = read_data(PREPROCESSING_DIR  / "full_cohort"  / "preprocessed_complete.csv")
    assignments = read_data(OVERALL_EVALUATION_DIR / f"k_{OVERALL_K:02d}" / "cluster_assignments.csv")

    if set(assignments["cluster"].unique()) != set(range(1, OVERALL_K + 1)):
        raise ValueError("Overall assignments do not match OVERALL_K.")

    if not profiles[CONDITIONS].isin([0, 1]).all().all():
        raise ValueError("Overall conditions must be complete 0/1 data.")

    profiles = align(profiles, assignments["person_id"].to_numpy())

    with open(OVERALL_EVALUATION_DIR / "analysis_configuration.json") as file:
        evaluation_config = json.load(file)
        
    #Check that teh value set for gamma is positive 
    gamma = float(evaluation_config["gamma"])
    if gamma <= 0:
        raise ValueError("Gamma must be positive.")

    assignments = assignments[["person_id", "cluster"]].rename(columns={"cluster": "overall_cluster"})

    return profiles, assignments, gamma, evaluation_config


# ================================================================
# CONDITION HEATMAPS AND PROFILES
# ================================================================

def create_profiles(profiles, labels, overall_mean, folder, cohort):
    data = profiles.copy()
    data["subgroup_cluster"] = labels
    
    #Calculate the summaries
    prevalence = (data.groupby("subgroup_cluster")[CONDITIONS].mean() * 100)
    subgroup_mean = data[CONDITIONS].mean() * 100
    relative_subgroup = prevalence.subtract(subgroup_mean, axis="columns")
    relative_overall = prevalence.subtract(overall_mean, axis="columns")
    
    #Save summaries to CSV
    prevalence.to_csv(folder / "condition_prevalence_percent.csv")
    subgroup_mean.rename("prevalence_percent").to_csv(folder / "subgroup_mean_prevalence.csv")
    relative_subgroup.to_csv(folder / "relative_to_subgroup_mean_pp.csv")
    relative_overall.to_csv(folder / "relative_to_overall_mean_pp.csv")
    
    #Check the size of the groups
    sizes = data.groupby("subgroup_cluster").size()
    pd.DataFrame({"n": sizes, "percent": 100 * sizes / len(data)}).to_csv(folder / "cluster_sizes.csv")
    
    #Age summary across clusters 
    data.groupby("subgroup_cluster")["age"].agg(
        ["count", "mean", "std", "median", "min", "max"]
    ).to_csv(folder / "age_profiles_original_scale.csv")

    # PAPER TIP:
    # Report absolute prevalences alongside relative differences.
    # Subgroup-relative heatmaps have different reference means,
    # so similar colours do not necessarily mean similar phenotypes.
    #
    # Shared overall-reference heatmaps facilitate interpretation,
    # but clusters still require clinical assessment.

    for table, reference, filename in [
        (relative_subgroup,  "subgroup",  "heatmap_relative_to_subgroup_mean"),
        (relative_overall, "overall cohort", "heatmap_relative_to_overall_mean")]:
        
        display = table.T.copy()
        display.index = [col.replace("_", " ").capitalize() for col in display.index]
        display.columns = [f"Cluster {c}\n(n={sizes.loc[c]:,})" for c in display.columns]
        plot_heatmap(display, folder, filename, f"{cohort}: relative to {reference} mean", relative=True)

    return prevalence


# ================================================================
# PCA / t-SNE / UMAP PROJECTIONS
# ================================================================

def create_projections(numerical, categorical, numerical_cols, categorical_cols, assignments, gamma, k, folder, cohort):
    # Weighted one-hot representation preserves the mixed distance:
    # d² = numerical squared differences
    #      + gamma * categorical mismatch count.
    # Full one-hot encoding gives squared distance 2 per mismatch;
    # multiplying by sqrt(gamma/2) restores the desired weighting.
    encoder = OneHotEncoder(sparse_output=False, handle_unknown="error")
    encoded = encoder.fit_transform(categorical)

    x_full = np.column_stack([numerical, encoded * np.sqrt(gamma / 2)])
    feature_names = (numerical_cols + encoder.get_feature_names_out(categorical_cols).tolist())
    rng = np.random.default_rng(SEED)
    sample_n = (len(assignments) if PROJECTION_SAMPLE_SIZE is None 
                else min(PROJECTION_SAMPLE_SIZE, len(assignments)))

    if sample_n < 4:
        raise ValueError("At least four people are needed for projections.")

    indices = np.sort(rng.choice(len(assignments), sample_n, replace=False))

    x = x_full[indices]
    output = assignments.iloc[indices].reset_index(drop=True).copy()
    labels = output["subgroup_cluster"].to_numpy()

    # PAPER TIP:
    # These projections visualize existing assignments only. Two-dimensional separation is not proof of valid clusters.
    # Separate subgroup projections have different coordinate systems: do not compare axis locations or spacing across cohorts.
    # t-SNE and UMAP depend on tuning and random initialization; repeat with alternative settings when interpretation depends
    # strongly on the apparent layout.
    
    # PCA
    pca = PCA(n_components=2, svd_solver="full")
    pca.fit(x_full)
    pca_xy = pca.transform(x)
    
    #T-SNE
    perplexity = min(float(TSNE_PERPLEXITY), max(2.0, (sample_n - 1) / 3))
    tsne = TSNE(n_components=2, perplexity=perplexity, learning_rate="auto", init="pca", max_iter=TSNE_MAX_ITER, random_state=SEED)
    tsne_xy = tsne.fit_transform(x)

    # CREATE UMAP 
    umap = UMAP(n_components=2, n_neighbors=min(UMAP_N_NEIGHBORS, sample_n - 1), min_dist=UMAP_MIN_DIST, metric="euclidean", random_state=SEED, n_jobs=1)
    umap_xy = umap.fit_transform(x)

    #CREATE DATAFRAMES TO SUMMARIZE THE RESULTS 
    pd.DataFrame({"component": ["PC1", "PC2"], "explained_variance_percent": (100 * pca.explained_variance_ratio_)}).to_csv(folder / "pca_explained_variance.csv", index=False)
    pd.DataFrame(pca.components_.T, index=pd.Index(feature_names, name="weighted_feature"), columns=["PC1_loading", "PC2_loading"]).to_csv(folder / "pca_loadings.csv")

    for name, model in [("pca", pca), ("tsne", tsne), ("umap", umap), ("category_encoder", encoder)]:
        joblib.dump(model, folder / f"{name}_model.joblib")

    projections = {"PCA": pca_xy, "tSNE": tsne_xy, "UMAP": umap_xy}
    colors = np.array(sns.color_palette("tab20", n_colors=k))
    drawing_order = rng.permutation(sample_n)

    def draw(ax, name, coordinates):
        ax.scatter(coordinates[drawing_order, 0], coordinates[drawing_order, 1], c=colors[labels[drawing_order] - 1], s=10,  alpha=0.45, linewidths=0, rasterized=True)
        for cluster in range(1, k + 1):
            ax.scatter([], [], c=[colors[cluster - 1]], label=f"Cluster {cluster}",  s=25)
        if name == "PCA":
            variance = 100 * pca.explained_variance_ratio_
            xlabel = f"PC1 ({variance[0]:.1f}% variance)"
            ylabel = f"PC2 ({variance[1]:.1f}% variance)"
        else:
            xlabel, ylabel = f"{name} 1", f"{name} 2"

        ax.set(title=name, xlabel=xlabel, ylabel=ylabel)
        ax.grid(alpha=0.15)

    for name, coordinates in projections.items():
        output[f"{name}_1"] = coordinates[:, 0]
        output[f"{name}_2"] = coordinates[:, 1]
        output[["person_id", "subgroup_cluster", f"{name}_1", f"{name}_2"]].to_csv(folder / f"{name.lower()}_coordinates.csv", index=False)

        fig, ax = plt.subplots(figsize=(8, 6))
        draw(ax, name, coordinates)
        ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
        fig.suptitle(f"{cohort}: k={k}")
        fig.tight_layout()
        save_figure(fig, folder, f"{name.lower()}_projection")

    output.to_csv(folder / "all_projection_coordinates.csv", index=False)

    # CREATE A FIGURE WITH ALL THREE DIMENSION REDUCTION TECHNIQUES
    fig, axes = plt.subplots(1, 3, figsize=(19, 6))
    for ax, (name, coordinates) in zip(axes, projections.items()):
        draw(ax, name, coordinates)
    handles, legend_labels = axes[0].get_legend_handles_labels()    
    fig.legend(handles, legend_labels, title="Subgroup cluster", loc="center right")
    fig.suptitle(f"{cohort}: k={k}, n={sample_n:,}")
    fig.tight_layout(rect=[0, 0, 0.93, 0.94])
    save_figure(fig, folder, "projections_three_panel")

    return {"projection_n": sample_n,  "tsne_perplexity_used": perplexity}


# ================================================================
# FIT AND ANALYSE EACH COHORT
# ================================================================

# Define a function that analyses one age–sex cohort.
# cohort: subgroup name, such as "female_age18_44".
# k: number of clusters to fit.
# gamma: weighting of categorical mismatches relative to numerical differences.
# overall_mean: overall condition prevalences, used for heatmaps later.
# overall_assignments: original overall-cohort labels, used for comparison later.

def analyse_cohort(cohort, k, gamma, overall_mean, overall_assignments):
    
    # Construct the export folder for this cohort and k.
    # For example: sensitivity_outputs/age_sex_stratified/female_age18_44/k_05.
    # The "/" operator joins paths; :02d formats k as a two-digit integer.
    folder = EXPORT_DIR / cohort / f"k_{k:02d}"
    
    # Create the folder and any missing parent folders.
    # exist_ok=True allows the folder to exist already.
    folder.mkdir(parents=True, exist_ok=True)

    # Locate the folder containing this cohort's preprocessed datasets.
    source = PREPROCESSING_DIR / cohort

    # Read the clustering features: scaled age and 0/1 conditions.
    # read_data() also checks that person_id is present, complete, and unique.
    features = read_data(source / "features_age_minmax_binary01.csv")
    # These values describe clusters; they are not all necessarily clustering inputs.
    profiles = read_data(source / "preprocessed_complete.csv")

    # Compare the sets of IDs to confirm both files contain the same people.
    # Sets ignore row order, which is addressed by align() below.
    if set(features["person_id"]) != set(profiles["person_id"]):
        # Stop if the two files contain different individuals.
        raise ValueError(f"{cohort}: feature/profile IDs differ.")

    # Reorder profile rows to match the order of people in features.
    # to_numpy() converts the ID column into a NumPy array.
    profiles = align(profiles, features["person_id"].to_numpy())
    
    # List the profile variables that must be complete.
    # *CONDITIONS expands the list of condition names into this list.
    required_profiles = ["age", "sex", *CONDITIONS]

    # Check for any missing values among the required profile variables.
    # isna() identifies missing cells; the two any() calls reduce the
    # result across rows and columns to one True/False answer.
    if profiles[required_profiles].isna().any().any():
        # Stop if any required profile value is missing.
        raise ValueError(f"{cohort}: missing profile values.")

    # Confirm every condition value is either 0 or 1.
    # isin() checks each cell; all().all() requires every cell to pass.
    if not profiles[CONDITIONS].isin([0, 1]).all().all():
        raise ValueError(f"{cohort}: conditions must be 0/1.")

    # Infer the expected sex from the cohort name.
    # This assumes the predefined "female_..." and "male_..." naming scheme.
    expected_sex = "Female" if cohort.startswith("female") else "Male"

    # Determine the expected age range from the cohort name.
    # between() includes both endpoints and returns one Boolean per person.
    age_valid = (profiles["age"].between(18, 44) if cohort.endswith("18_44") else profiles["age"].between(45, 120))

    # Check that every person has the expected sex and an eligible age.
    # eq() tests equality; all() requires every person to meet the condition.
    if not profiles["sex"].eq(expected_sex).all() or not age_valid.all():
        raise ValueError(f"{cohort}: unexpected age/sex membership.")

    # Constant sex does not distinguish people within a subgroup.
    features = features.drop(columns="sex_female", errors="ignore")

    # Include age as a candidate numerical feature if the column exists.
    # Otherwise, create an empty numerical-feature list.
    candidate_num = ["age"] if "age" in features else []
    
    # Select condition columns present in the feature file.
    # Conditions remain categorical even though their storage values are 0/1.
    candidate_cat = [c for c in CONDITIONS if c in features]
    
    # A non-empty set means unexpected columns were found.
    unknown = (set(features.columns) - {"person_id"} - set(candidate_num + candidate_cat))
    if unknown:
        raise ValueError(f"Unexpected features: {sorted(unknown)}")

    # PAPER TIP:
    # Keep constant conditions in descriptive profiles but exclude
    # them from clustering. Review rare/correlated conditions rather
    # than automatically removing clinically meaningful variables.

    constants = [c for c in candidate_num + candidate_cat if features[c].nunique() <= 1]
    numerical_cols = [c for c in candidate_num if c not in constants]
    categorical_cols = [c for c in candidate_cat if c not in constants]
    
    # Confirm that both numerical and categorical features remain.
    # An empty list evaluates to False, so "not list" identifies an empty list.
    if not numerical_cols or not categorical_cols:
        raise ValueError(f"{cohort}: k-prototypes requires numerical and categorical features.")

    # Define a consistent feature order: numerical first, categorical second.
    # Later code uses this order to tell k-prototypes which columns are categorical.
    feature_order = numerical_cols + categorical_cols
    if features[feature_order].isna().any().any():
        raise ValueError("Missing clustering values.")

    # Confirm that all selected categorical features contain only 0 and 1.
    if not features[categorical_cols].isin([0, 1]).all().all():
        raise ValueError("Categorical features must contain 0/1.")
    
    # ---------------------------------------------------------------
    # Validate numerical features with floating-point tolerance.
    # ---------------------------------------------------------------
    
    numerical_df = features[numerical_cols].apply(pd.to_numeric, errors="raise")
    numerical = numerical_df.to_numpy(dtype=float)
    
    if not np.isfinite(numerical).all():
        raise ValueError(f"{cohort}: numerical features contain missing or infinite values. Investigate the preprocessing exports.")
    #Reformat (transpose and ensure string values )
    print(f"\n{cohort}: numerical feature ranges:")
    print(numerical_df.agg(["min", "max"]).T.to_string())
    
    # Allow tiny rounding differences from scaling and CSV export.
    tolerance = 1e-8
    outside_tolerance = ((numerical < -tolerance) | (numerical > 1.0 + tolerance))
    if outside_tolerance.any():
        ranges = numerical_df.agg(["min", "max"]).T
        raise ValueError(
            f"\n{cohort}: numerical features are outside the expected "
            f"min–max range beyond tolerance {tolerance}.\n\n"
            f"Observed ranges:\n{ranges.to_string()}\n\n"
            "Check that the script imports "
            "'features_age_minmax_binary01.csv', rather than the "
            "unscaled or z-score export. If necessary, rerun preprocessing.")
    
    # Clip only negligible rounding differences.
    # Do not silently rescale substantially out-of-range values:
    # changing scaling would change numerical/categorical weighting.
    numerical = np.clip(numerical, 0.0, 1.0)
    
    # Update the dataframe so the saved analysis input matches the
    # numerical values actually used for clustering.
    for column_index, column in enumerate(numerical_cols):
        features[column] = numerical[:, column_index]
    
    categorical = features[categorical_cols].to_numpy(dtype=int)

    if not 2 <= k < len(features):
        raise ValueError("K must be >=2 and smaller than cohort size.")

    mixed = np.empty((len(features), len(feature_order)), dtype=object)
    mixed[:, :len(numerical_cols)] = numerical
    mixed[:, len(numerical_cols):] = categorical

    if len(pd.DataFrame(mixed).drop_duplicates()) < k:
        raise ValueError("Fewer unique profiles than requested clusters.")

    #Re-fit model - note that the specification should match what was initially used for the main analysis (for example: same init methodology)
    model = KPrototypes(n_clusters=k, init="Huang", n_init=N_INIT,  max_iter=MAX_ITER, gamma=gamma, random_state=SEED, n_jobs=1, verbose=0)
    labels = model.fit_predict(mixed, categorical=list(range(len(numerical_cols), len(feature_order)))).astype(int) + 1

    if len(np.unique(labels)) != k:
        raise RuntimeError("Fewer occupied clusters than requested.")

    assignments = features[["person_id"]].copy()
    assignments["subgroup_cluster"] = labels
    assignments.to_csv(folder / "cluster_assignments.csv", index=False)

    features[["person_id"] + feature_order].to_csv(folder / "analysis_input.csv", index=False)
    
    joblib.dump(model, folder / "kprototypes_model.joblib")

    prototypes = pd.DataFrame(model.cluster_centroids_, columns=feature_order)
    prototypes.insert(0, "subgroup_cluster", range(1, k + 1))
    prototypes.to_csv(folder / "prototypes.csv", index=False)

    cluster_profiles = create_profiles(profiles, labels, overall_mean, folder, cohort)
    projection_metadata = create_projections(numerical, categorical, numerical_cols, categorical_cols, assignments, gamma, k, folder, cohort)

    # Compare partitions of the SAME individuals.
    comparison = assignments.merge(overall_assignments, on="person_id", how="left", validate="one_to_one")
    if comparison["overall_cluster"].isna().any():
        raise ValueError("Some subgroup IDs lack overall assignments.")

    ari = adjusted_rand_score(comparison["overall_cluster"], comparison["subgroup_cluster"])
    ami = adjusted_mutual_info_score(comparison["overall_cluster"], comparison["subgroup_cluster"])

    # Cluster numbering is arbitrary. ARI and AMI do not require
    # matching Cluster 1 to Cluster 1 across independent fits.
    counts = pd.crosstab(
        comparison["overall_cluster"],
        comparison["subgroup_cluster"]
    ).reindex(
        index=range(1, OVERALL_K + 1),
        columns=range(1, k + 1),
        fill_value=0
    )

    counts.to_csv(folder / "overall_vs_subgroup_counts.csv")

    row_percent = counts.div(counts.sum(axis=1).replace(0, np.nan), axis=0) * 100

    row_percent.to_csv(folder / "overall_vs_subgroup_row_percent.csv")
    comparison.to_csv(folder / "assignments_compared_with_overall.csv", index=False)

    plot_heatmap(counts, folder, "overall_vs_subgroup_assignments", f"{cohort}: overall assignments vs subgroup refit")

    sizes = pd.Series(labels).value_counts()

    summary = {
        "cohort": cohort,
        "n": len(assignments),
        "k": k,
        "cost": float(model.cost_),
        "minimum_cluster_n": int(sizes.min()),
        "minimum_cluster_percent": float(100 * sizes.min() / len(assignments)),
        "ARI_vs_overall": float(ari),
        "AMI_vs_overall": float(ami)
    }

    metadata = {
        **summary,
        **projection_metadata,
        "gamma": gamma,
        "seed": SEED,
        "n_init": N_INIT,
        "max_iter": MAX_ITER,
        "numerical_features": numerical_cols,
        "categorical_features": categorical_cols,
        "additional_constants_removed": constants,
        "scaling": "Independently fitted during subgroup preprocessing"
    }

    with open(folder / "analysis_metadata.json", "w") as file:
        json.dump(metadata, file, indent=2)

    cluster_profiles = cluster_profiles.reset_index()
    cluster_profiles.insert(0, "cohort", cohort)

    return summary, cluster_profiles, assignments["person_id"].tolist()


# ================================================================
# RUN ALL FOUR COHORTS
# ================================================================

def main():
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="white")
    (overall_profiles, overall_assignments, gamma, overall_config) = load_overall()

    overall_mean = overall_profiles[CONDITIONS].mean() * 100

    summaries = []
    profile_tables = []
    all_ids = []

    for cohort in COHORTS:
        k = K_BY_COHORT.get(cohort, OPTIMAL_K)
        print(f"Running {cohort}: k={k}...", flush=True)
        summary, profiles, ids = analyse_cohort(cohort, k, gamma, overall_mean, overall_assignments)
        summaries.append(summary)
        profile_tables.append(profiles)
        all_ids.extend(ids)

        # Save a checkpoint after each completed subgroup.
        pd.DataFrame(summaries).to_csv(EXPORT_DIR / "sensitivity_summary.csv", index=False)

    if len(all_ids) != len(set(all_ids)):
        raise ValueError("Individuals overlap across subgroup cohorts.")

    if set(all_ids) != set(overall_profiles["person_id"]):
        raise ValueError("The four cohorts do not partition the overall cohort.")

    pd.concat(profile_tables, ignore_index=True).to_csv(EXPORT_DIR / "all_subgroup_condition_profiles_percent.csv", index=False)

    configuration = {
        "optimal_k": OPTIMAL_K,
        "k_by_cohort": K_BY_COHORT,
        "overall_k": OVERALL_K,
        "gamma": gamma,
        "seed": SEED,
        "overall_evaluation_configuration": overall_config,
        "projection_sample_size": PROJECTION_SAMPLE_SIZE,
        "tsne_perplexity": TSNE_PERPLEXITY,
        "tsne_max_iter": TSNE_MAX_ITER,
        "umap_n_neighbors": UMAP_N_NEIGHBORS,
        "umap_min_dist": UMAP_MIN_DIST
    }

    with open(EXPORT_DIR / "sensitivity_configuration.json", "w") as file:
        json.dump(configuration, file, indent=2)

    # PAPER TIP:
    # This is within-population subgroup refitting, not independent
    # external validation. The subgroup and overall derivation
    # samples overlap.
    #
    # Additional analyses are warranted:
    # - Repeated random subsets and resampling stability;
    # - Common features and overall-fitted scaling across cohorts;
    # - Alternative gamma values, feature sets, and algorithms;
    # - Temporal subsets;
    # - Independent geographic or health-system populations;
    # - External outcomes not used to derive clusters;
    # - Held-out assignment using fixed overall prototypes.
    #
    # Agreement does not establish clinical usefulness, and
    # disagreement may reflect meaningful subgroup heterogeneity.

    notes = """
This analysis assesses within-population reproducibility under independent age-sex subgroup refitting.

ARI and AMI compare assignments for the same individuals and are invariant to cluster label numbering. There is no universal threshold
that establishes adequate transportability.

Subgroup-specific age scaling and exclusion of constant sex change the clustering geometry. Repeat using common features and
overall-fitted scaling to separate these effects from composition.

Additional sensitivity and validation analyses are warranted:
random subsets, resampling stability, temporal subsets, alternative preprocessing and weighting, external populations, 
independent clinical outcomes, and held-out assignment using fixed prototypes.

Separately fitted projection coordinates cannot be compared directly.
Visual separation alone is not evidence of cluster validity.
"""

    (EXPORT_DIR / "interpretation_notes.txt").write_text(notes.strip(), encoding="utf-8")
    print("\nSummary:")
    print(pd.DataFrame(summaries).round(3).to_string(index=False))
    print(f"\nResults saved to: {EXPORT_DIR.resolve()}")


if __name__ == "__main__":
    main()