#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 25 09:59:22 2026

@author: gemmapostill
"""



from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler


# ================================================================
# THIS SCRIPT PRE-PROCESSES THE DATA FOR CLUSTERING 
# ================================================================


#Read me: in this script we will take the raw data and get it into a format suitable for clustering 
#For the purpose of learning, function definition occurs where the clusters are used 



# ================================================================
# SET_UP 
# ================================================================

INPUT_FILE = Path("synthetic_chronic_conditions_8000.csv")
OUTPUT_DIR = Path("")

AGE_CUTOFF = 45

# Include demographics in the clustering feature matrix?
#This is an important considerations - are you trying to create clusters of the people with multimorbiditiy, or solely the conditions themseleves. 

# True: clusters can reflect demographics as well as conditions.
# False: demographics remain available for descriptive profiling,
#        but do not directly determine clustering distances.


INCLUDE_AGE = True
INCLUDE_SEX = True

# Used to flag conditions for review, NOT automatically remove them.
RARE_PREVALENCE_THRESHOLD = 0.01
HIGH_CORRELATION_THRESHOLD = 0.70

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
#  VALIDATE INPUT DATA
# ================================================================

def validate_input(data):
    """Validate structure, coding, identifiers, and missingness."""
    ## Note that in the practice dataset, there is no missingness 
    data = data.copy()
    required_columns = ["person_id", "age", "sex", *CONDITION_COLUMNS]
    
    missing_columns = sorted(set(required_columns) - set(data.columns))
    if missing_columns:
        raise ValueError(f"Required columns are missing: {missing_columns}")

    # Retain only explicitly specified variables.
    data = data[required_columns].copy()
    missing_counts = data.isna().sum()
    if missing_counts.any():
        raise ValueError(
            "Missing values detected. Investigate before preprocessing:\n"
            + missing_counts[missing_counts > 0].to_string())

    if data["person_id"].duplicated().any():
        raise ValueError("person_id must uniquely identify each person.")

    data["age"] = pd.to_numeric(data["age"], errors="raise")

    if not np.isfinite(data["age"]).all():
        raise ValueError("Age contains non-finite values.")

    if not data["age"].between(18, 120).all():
        raise ValueError("All ages must be between 18 and 120.")

    # Normalize text formatting without treating unexpected values as No.
    data["sex"] = (data["sex"].astype("string").str.strip().str.title())

    if not data["sex"].isin(["Female", "Male"]).all():
        raise ValueError("Expected sex coded Female/Male for this simulated dataset.")

    for column in CONDITION_COLUMNS:
        normalized = (data[column].astype("string").str.strip().str.lower())
        if not normalized.isin(["yes", "no"]).all():
            unexpected = normalized[ ~normalized.isin(["yes", "no"])].unique().tolist()
            raise ValueError(f"Unexpected values in {column}: {unexpected}")
        data[column] = normalized.map({"yes": "Yes", "no": "No"})

    return data


# ================================================================
# INITIAL EXPLORATION OF DATA 
# ================================================================

def explore_data(data, folder):
    """Save descriptive summaries before transformations."""
    # Missingness and variable types.
    variable_summary = pd.DataFrame({
        "variable": data.columns,
        "dtype": data.dtypes.astype(str).to_numpy(),
        "missing_n": data.isna().sum().to_numpy(),
        "unique_values_n": data.nunique().to_numpy()
    })

    variable_summary.to_csv(folder / "variable_summary.csv", index=False)

    # Age distribution.
    age_summary = data["age"].describe(percentiles=[0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99]).to_frame("value")
    age_summary.loc["skewness", "value"] = data["age"].skew()
    age_summary.to_csv(folder / "age_summary.csv")
    
    age_counts = (data["age"].value_counts().sort_index().rename_axis("age").reset_index(name="n"))
    age_counts.to_csv(folder / "age_distribution.csv", index=False)

    # Flag potential age outliers; do not automatically exclude them.
    q1 = data["age"].quantile(0.25)
    q3 = data["age"].quantile(0.75)
    iqr = q3 - q1

    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    age_outliers = data.loc[(data["age"] < lower_bound) | (data["age"] > upper_bound), ["person_id", "age", "sex"]]
    age_outliers.to_csv(folder / "age_outliers_for_review.csv", index=False)

    # Sex distribution.
    sex_summary = (data["sex"].value_counts().rename_axis("sex").reset_index(name="n"))
    sex_summary["percent"] = 100 * sex_summary["n"] / len(data)
    sex_summary.to_csv(folder / "sex_distribution.csv", index=False)

    # Condition prevalence and variability.
    binary = data[CONDITION_COLUMNS].eq("Yes").astype("int8")

    prevalence = pd.DataFrame({
        "condition": CONDITION_COLUMNS,
        "yes_n": binary.sum().to_numpy(),
        "no_n": (len(data) - binary.sum()).to_numpy(),
        "prevalence": binary.mean().to_numpy(),
        "variance": binary.var(ddof=0).to_numpy(),
        "constant": binary.nunique().eq(1).to_numpy()
    })

    prevalence["percent"] = 100 * prevalence["prevalence"]
    prevalence["rare_flag"] = (prevalence["prevalence"] < RARE_PREVALENCE_THRESHOLD)
    prevalence.to_csv(folder / "condition_prevalence.csv", index=False)

    # Number of conditions per person: descriptive only.
    # Do not add this to the clustering features alongside all conditions,
    # because it would count disease burden a second time.
    condition_counts = (binary.sum(axis=1).value_counts().sort_index().rename_axis("number_of_conditions").reset_index(name="n"))
    condition_counts["percent"] = (100 * condition_counts["n"] / len(data))
    condition_counts.to_csv(folder / "condition_count_distribution.csv", index=False)

    # Identical clinical profiles can legitimately occur.
    # Flag them descriptively; do not remove them as duplicate people.
    repeated_profiles_n = data.duplicated(subset=["age", "sex", *CONDITION_COLUMNS], keep=False).sum()

    return {
        "n": int(len(data)),
        "age_min": float(data["age"].min()),
        "age_max": float(data["age"].max()),
        "age_outliers_flagged_n": int(len(age_outliers)),
        "people_in_repeated_profiles_n": int(repeated_profiles_n)
    }


# ================================================================
# ENCODE VARIABLES
# ================================================================

def encode_data(data):
    """Recode conditions Yes/No -> 1/0."""
    encoded = data.copy()
    for column in CONDITION_COLUMNS:
        encoded[column] = encoded[column].map({"Yes": 1, "No": 0}).astype("int8")

    # One binary column suffices for this two-category sex variable.
    # This is a category code, not a ranking.
    encoded["sex_female"] = encoded["sex"].map({"Female": 1, "Male": 0}).astype("int8")
    return encoded


# ================================================================
# CORRELATION EXPLORATION
# ================================================================

def explore_correlations(encoded, folder):
    """Save condition correlations and strongly correlated pairs."""

    binary = encoded[CONDITION_COLUMNS]

    # Constants have undefined correlations: exclude from this matrix and report them separately in condition_prevalence.csv.
    #A varying clustering feature is a variable with at least two different observed values in the cohort being analysed.
    varying_columns = binary.columns[binary.nunique() > 1]
    correlations = binary[varying_columns].corr()
    # Pearson correlation between binary variables is phi correlation.
    correlations.to_csv(folder / "condition_phi_correlations.csv")

    pairs = []
    for i, first in enumerate(varying_columns):
        for second in varying_columns[i + 1:]:
            correlation = correlations.loc[first, second]
            if abs(correlation) >= HIGH_CORRELATION_THRESHOLD:
                pairs.append({ "condition_1": first, "condition_2": second, "phi_correlation": correlation})

    pairs_df = pd.DataFrame(pairs, columns=[ "condition_1", "condition_2", "phi_correlation"])

    if not pairs_df.empty:
        pairs_df = pairs_df.sort_values("phi_correlation", key=lambda values: values.abs(), ascending=False)

    pairs_df.to_csv(folder / "high_correlation_pairs.csv", index=False)


# ================================================================
# FUNCTION DEFINITON FOR SCALING
# ================================================================

def save_scaler_parameters(scaler, columns, method, path):
    """Save fitted scaling parameters for reproducibility."""
    # Note that this function is set up to enable to researcher to choose the method of transformation that is most appropriate for the data included 
    if method == "minmax":
        parameters = pd.DataFrame({"feature": columns, "minimum": scaler.data_min_, "maximum": scaler.data_max_, "multiplier": scaler.scale_, "offset": scaler.min_})
        # Transform new values as:
        # transformed = original * multiplier + offset
    elif method == "standard":
        parameters = pd.DataFrame({"feature": columns, "mean": scaler.mean_, "standard_deviation": scaler.scale_})
        # Transform new values as:
        # transformed = (original - mean) / standard_deviation
    else:
        raise ValueError("Unknown scaling method.")

    parameters.to_csv(path, index=False)



# ================================================================
# FUNCTION DEFINITON FOR EXPORT
# ================================================================


def export_features(encoded, folder):
    """Export unscaled and alternative scaled feature matrices."""

    candidate_features = CONDITION_COLUMNS.copy()

    if INCLUDE_AGE:
        candidate_features.append("age")

    if INCLUDE_SEX:
        candidate_features.append("sex_female")

    # Remove constants only from clustering matrices.
    # Keep them in the complete encoded dataframe.
    constant_features = [column for column in candidate_features if encoded[column].nunique() <= 1]
    feature_columns = [column for column in candidate_features if column not in constant_features]
    pd.DataFrame({"constant_feature_removed": constant_features}).to_csv(folder / "constant_features_removed.csv", index=False)

    if not feature_columns:
        raise ValueError("No varying clustering features remain.")

    features = encoded[feature_columns].copy()

    # ID is exported for linking results back to people.
    # ID is NEVER included in scaling or clustering features.
    def save_matrix(matrix, filename):
        if matrix.isna().any().any():
            raise ValueError(f"Missing values in {filename}.")

        if not np.isfinite(matrix.to_numpy(dtype=float)).all():
            raise ValueError(f"Non-finite values in {filename}.")

        output = pd.concat(
            [encoded[["person_id"]], matrix],
            axis=1
        )

        output.to_csv(folder / filename, index=False)

    # A. Encoded features, without scaling.
    save_matrix(features, "features_unscaled.csv")

    # B. Min–max scale AGE ONLY; binary variables stay 0/1.
    minmax_features = features.copy()

    # C. Z-score standardize AGE ONLY; binary variables stay 0/1.
    age_z_features = features.copy()

    if "age" in feature_columns:
        #separate min/max based scalar
        age_minmax_scaler = MinMaxScaler()
        minmax_features["age"] = (age_minmax_scaler.fit_transform(features[["age"]]).ravel())
        save_scaler_parameters(age_minmax_scaler, ["age"], "minmax", folder / "age_minmax_parameters.csv")
        #separate z scalar
        age_standard_scaler = StandardScaler()
        age_z_features["age"] = (age_standard_scaler.fit_transform(features[["age"]]).ravel())
        save_scaler_parameters(age_standard_scaler, ["age"], "standard", folder / "age_zscore_parameters.csv")

    save_matrix(minmax_features, "features_age_minmax_binary01.csv")
    save_matrix(age_z_features, "features_age_zscore_binary01.csv")

    # D. Sensitivity analysis: standardize ALL included features.
    # This changes the relative weight of rare binary conditions.
    all_standard_scaler = StandardScaler()
    all_standardized = pd.DataFrame(all_standard_scaler.fit_transform(features),columns=feature_columns, index=features.index)
    save_matrix(all_standardized, "features_all_zscore_sensitivity.csv")
    save_scaler_parameters(all_standard_scaler, feature_columns, "standard", folder / "all_features_zscore_parameters.csv")

    return feature_columns, constant_features


# ================================================================
# COMPLETE PREPROCESSING FOR ONE COHORT
# ================================================================

def preprocess_cohort(data, cohort_name):
    """Fit exploration and preprocessing independently in one cohort."""

    if data.empty:
        raise ValueError(f"{cohort_name} contains no individuals.")

    # Reset index so exported matrices align cleanly with identifiers.
    data = data.copy().reset_index(drop=True)

    folder = OUTPUT_DIR / cohort_name
    folder.mkdir(parents=True, exist_ok=True)

    exploration_summary = explore_data(data, folder)
    encoded = encode_data(data)
    explore_correlations(encoded, folder)
    feature_columns, constant_features = export_features(encoded, folder)

    # Complete dataframe for profiling and linking cluster assignments.
    # Contains all 32 conditions, including any that are constant.
    encoded.to_csv(folder / "preprocessed_complete.csv", index=False)

    metadata = {
        "cohort": cohort_name,
        **exploration_summary,
        "age_cutoff": AGE_CUTOFF,
        "include_age_in_clustering": INCLUDE_AGE,
        "include_sex_in_clustering": INCLUDE_SEX,
        "clustering_features": feature_columns,
        "constant_features_removed_from_matrices": constant_features,
        "condition_coding": {"No": 0, "Yes": 1},
        "sex_coding": {"Male": 0, "Female": 1},
        "missing_data_action": "Fail validation; no imputation performed",
        "rare_conditions_action": "Flagged and retained",
        "correlated_conditions_action": "Reported and retained",
        "outlier_action": "Flagged and retained",
        "scaling_scope": "Fitted independently within this cohort",
        "id_is_clustering_feature": False
    }

    with open(folder / "preprocessing_metadata.json", "w") as file:
        json.dump(metadata, file, indent=2)

    print(
        f"{cohort_name}: "
        f"{len(encoded):,} people; "
        f"{len(feature_columns)} varying clustering features"
    )

    return {
        "cohort": cohort_name,
        "n": len(encoded),
        "clustering_features_n": len(feature_columns),
        "constant_features_removed": ", ".join(constant_features)
    }


# ================================================================
# RUN FULL-DATASET PREPROCESSING
# ================================================================

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

raw = pd.read_csv(INPUT_FILE)
data = validate_input(raw)

if len(data) != 8000:
    raise ValueError(f"Expected 8,000 people; found {len(data):,}.")

summaries = [preprocess_cohort(data, "full_cohort")]


# ================================================================
# REPEAT IN FOUR AGE–SEX COHORTS
# ================================================================

#Note that these cohorts are created for later sensitivity analyses for the outputted clusters 

cohorts = {
    "female_age18_44": data.loc[data["sex"].eq("Female") & data["age"].lt(AGE_CUTOFF)],
    "female_age45plus": data.loc[ data["sex"].eq("Female") & data["age"].ge(AGE_CUTOFF)],
    "male_age18_44": data.loc[data["sex"].eq("Male")  & data["age"].lt(AGE_CUTOFF)],
    "male_age45plus": data.loc[data["sex"].eq("Male") & data["age"].ge(AGE_CUTOFF)],
}

# Confirm every person belongs to exactly one stratified cohort.
stratified_ids = pd.concat([cohort["person_id"] for cohort in cohorts.values()])

assert len(stratified_ids) == len(data)
assert stratified_ids.is_unique
assert set(stratified_ids) == set(data["person_id"])

for cohort_name, cohort_data in cohorts.items():
    summaries.append(preprocess_cohort(cohort_data, cohort_name))

pd.DataFrame(summaries).to_csv(OUTPUT_DIR / "cohort_summary.csv", index=False)
print(f"\nExports saved to: {OUTPUT_DIR.resolve()}")