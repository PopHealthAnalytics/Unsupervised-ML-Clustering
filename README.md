# Clustering Health Data: A Practical Walkthrough

This repository provides an educational Python walkthrough of the clustering framework described in the accompanying paper, *Clustering health data with machine learning: a discovery-oriented approach for epidemiologic research*. Four scripts demonstrate how to prepare data, evaluate candidate clustering solutions, visualize a selected solution, and examine whether similar patterns recur within population subgroups.

The example clusters individuals according to chronic conditions and demographics using **k-prototypes**, an algorithm designed for mixed numerical and categorical data.

## Authors

This repository was created and is currently maintained by Gemma Postill and Ingrid Giesinger. 

## The workflow

| Step | Script | Purpose |
|---|---|---|
| **1. Preprocessing** | `01_preprocessing.py` | Explore variables, check data quality, encode conditions, prepare alternative scaling approaches, and create age–sex cohorts. |
| **2. Cluster evaluation** | `02_cluster_evaluation.py` | Explore density structure with OPTICS and evaluate k-prototypes solutions for k = 2–12. |
| **3. Visualization** | `03_visualization.py` | Visualize the selected solution using condition-profile heatmaps and PCA, t-SNE, and UMAP projections. |
| **4. Subgroup sensitivity analysis** | `04_subgroup_sensitivity.py` | Refit clustering within four age–sex cohorts and assess whether similar assignments and condition profiles recur. |

**The workflow is iterative.** Evaluation, clinical interpretation, and sensitivity analyses may motivate revisiting preprocessing, feature selection, weighting, or the number of clusters. Note that the coding workflow is simplified into four main chunks, these correspond to steps 2-5 in the manuscripts workflow (Figure 1). 

## Educational dataset

**The dataset is entirely synthetic and was created for educational purposes only. It contains no real patient records.**

The dataset includes 8,000 simulated individuals with:

- Age between 18 and 120 years.
- A simulated binary sex variable, coded Female/Male.
- Thirty-two chronic conditions, initially coded Yes/No.
- No missing values.

The simulation includes approximate community prevalence targets, age- and sex-related patterns, and shared factors that induce associations among cardiovascular and immune-mediated conditions.

These assumptions illustrate a clustering workflow. They do **not** constitute a validated representation of a population, disease prevalence, or clinical disease relationships. The resulting clusters should not be interpreted as established clinical phenotypes or used for clinical decisions.

## Getting started

Install the required packages:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn kmodes umap-learn scipy joblib
```

Use the following script names when adding the walkthrough code to the repository:

```bash
python 01_preprocessing.py
python 02_cluster_evaluation.py

# Review the evaluation outputs and specify your selected k
# at the top of the visualization and sensitivity scripts.

python 03_visualization.py
python 04_subgroup_sensitivity.py
```

Before running each script, update the input and export paths in its settings section. Run the scripts from the repository root, or use absolute paths.

The scripts are intended to support interpretation rather than automatically identify a single “correct” solution.

## Step 1: Preprocessing

### Script: `01_preprocessing.py`

This script explores and prepares the dataset before clustering.

It:

- Validates identifiers, age ranges, condition coding, and missingness.
- Summarizes age, sex, condition prevalence, and disease burden.
- Flags rare conditions, potentially extreme ages, and strongly correlated condition pairs.
- Recodes Yes/No conditions to 1/0.
- Exports alternative numerical scaling approaches.
- Repeats preprocessing separately within four cohorts: Female 18–44, Female ≥45, Male 18–44, and Male ≥45.

Conditions are treated as **categorical variables**, even when stored as 0/1.

Constant variables are retained in the complete descriptive dataset but excluded from clustering feature matrices. For example, sex is constant within each sex-stratified cohort, and a rare condition may be absent throughout a subgroup.

### Associated outputs

| Output | Description |
|---|---|
| `preprocessed_complete.csv` | Complete dataset with all conditions coded 0/1, identifiers, and demographics. |
| `features_unscaled.csv` | Candidate clustering features without scaling. |
| `features_age_minmax_binary01.csv` | Age scaled to 0–1; binary variables retain 0/1 coding. |
| `features_age_zscore_binary01.csv` | Age standardized; binary variables retain 0/1 coding. |
| `features_all_zscore_sensitivity.csv` | All included features standardized for sensitivity analysis. |
| Descriptive summaries | Missingness, distributions, condition prevalence, and correlations. |
| Scaling parameters and metadata | Records of transformations, feature selection, and cohort definitions. |

Min–max scaling and z-score standardization are **alternative representations**, rather than sequential requirements. Standardizing binary conditions changes their relative weight and can give rare conditions greater influence.

## Step 2: Cluster evaluation

### Script: `02_cluster_evaluation.py`

This script imports the overall cohort’s preprocessed feature matrix and fits **k-prototypes solutions for k = 2–12**.

K-prototypes combines numerical differences with categorical mismatches. The `gamma` parameter controls their relative contribution and is held fixed across candidate values of k.

An OPTICS reachability plot provides an additional exploratory view of density structure. It does not prescribe the number of k-prototypes clusters.

### Associated outputs

| Output | Description |
|---|---|
| `evaluation_metrics.csv` | Evaluation measures for each candidate k. |
| Evaluation figures | Individual and combined plots of DBI, cost, silhouette, and cohesion. |
| OPTICS outputs | Reachability plot, ordered reachability values, and fitted model. |
| `k_XX/cluster_assignments.csv` | Individual assignments and distances to prototypes for each k. |
| `cluster_assignments_all_k.csv` | Assignments for all candidate values of k in one table. |
| Cluster profiles and sizes | Condition prevalences, demographics, and cluster membership counts. |
| Models and configuration | Fitted models, analysis settings, and the evaluation sample. |

### Interpreting the evaluation measures

| Measure | Interpretation |
|---|---|
| **Mixed-distance DBI** | Lower values suggest more compact and separated clusters under the specified mixed-distance definition. |
| **Cost** | The k-prototypes objective; generally decreases as k increases. Inspect changes in its rate of improvement. |
| **Silhouette** | Higher values indicate better within-cluster similarity relative to other clusters under the selected distance. |
| **Cohesion** | Defined here as mean distance to the assigned prototype; lower values indicate greater compactness. |

DBI is a **mixed-prototype generalization**, rather than the conventional numerical-centroid DBI. Silhouette and OPTICS use a fixed sample by default to limit pairwise-distance computation; model fitting uses the full cohort.

**The script does not automatically select an optimal k.** Review the measures alongside cluster size, clinical interpretation, and sensitivity to analytic choices. Cost and cohesion are related measures and should not be treated as independent confirmation.

## Step 3: Visualization

### Script: `03_visualization.py`

Specify your selected number of clusters using `OPTIMAL_K` at the top of the script.

The script imports the saved assignments for that solution and creates condition-profile heatmaps and two-dimensional projections. It does not refit the clustering model.

### Associated outputs

| Output | Description |
|---|---|
| Condition heatmap | Cluster prevalence minus overall prevalence, expressed in percentage points. |
| Absolute condition profiles | Original-scale condition prevalences for interpreting the heatmap. |
| PCA projection | A linear projection, with explained variance and loadings exported. |
| t-SNE and UMAP projections | Nonlinear visualizations of the selected solution. |
| Projection coordinates | Individual identifiers, cluster assignments, and coordinates. |
| Models and settings | Fitted projection objects and configuration records. |

For example, a heatmap value of **+10** means that a condition is 10 percentage points more prevalent in that cluster than in the overall cohort.

The projections use a weighted representation that preserves the mixed distance defined in the evaluation script.

**Projection plots are descriptive tools, not validation on their own.** Apparent separation or overlap depends on dimensionality reduction, tuning, and the information lost when compressing the data into two dimensions.

## Step 4: Subgroup sensitivity analysis

### Script: `04_subgroup_sensitivity.py`

This script imports the four age–sex cohorts and independently refits k-prototypes within each cohort.

Specify `OPTIMAL_K` at the top. The same k is used across cohorts by default; optional cohort-specific overrides are available.

The aim is to examine whether similar disease patterns recur within subsets of the original population.

### Associated outputs

| Output | Description |
|---|---|
| Subgroup cluster assignments | Individual assignments from each independently fitted solution. |
| Cluster sizes and profiles | Absolute condition prevalences and original-scale age summaries. |
| Subgroup-relative heatmaps | Differences from each subgroup’s own mean prevalences. |
| Overall-relative heatmaps | Differences from the overall cohort’s mean prevalences. |
| PCA, t-SNE, and UMAP outputs | Figures, coordinates, and projection objects for each subgroup. |
| Assignment cross-tabulations | Overall assignments compared with subgroup-refitted assignments for the same people. |
| `sensitivity_summary.csv` | Cohort sizes, cluster sizes, and assignment agreement measures. |
| Metadata and interpretation notes | Settings, limitations, and further validation considerations. |

Assignment agreement is summarized using **adjusted Rand index (ARI)** and **adjusted mutual information (AMI)**. These measures account for arbitrary cluster numbering: Cluster 1 in one fitted solution does not necessarily correspond to Cluster 1 in another.

Interpret agreement alongside the condition profiles. Similar clinical patterns may recur despite changes in individual assignments, while high assignment agreement does not establish clinical usefulness.

Because preprocessing was fitted independently within each subgroup, age scaling differs across cohorts and constant sex is excluded. Results therefore reflect sensitivity to the complete subgroup-refitting pipeline. Separately fitted projection coordinates should not be compared directly across cohorts.

## Results included in this repository

The repository will also include example outputs generated by the scripts, allowing users to inspect the workflow before running it themselves.

These results include evaluation curves, selected cluster assignments, condition profiles, heatmaps, projection plots, and subgroup sensitivity summaries. Configuration files document the settings used to generate them.

**All example results are derived from the synthetic educational dataset.** They illustrate how to operationalize and interpret the workflow; they are not empirical findings about patients or communities. Results may vary with software versions, initialization, preprocessing, weighting, and selected parameters.

## Relationship to the accompanying paper

This repository operationalizes selected components of the paper’s discovery-oriented framework. It demonstrates how analytic decisions can be documented and assessed through exploration, evaluation, interpretation, and sensitivity analysis.

It does not implement every possible clustering method or validation approach. Numerical evaluation should be combined with subject-matter expertise, consideration of data-generating processes, and transparent reporting.

## Further sensitivity analysis and validation

Age–sex subgroup refitting provides one assessment of **within-population reproducibility**. It does not, on its own, establish transportability, particularly because subgroup and overall derivation samples overlap.

Additional analyses are warranted, including:

- Repeated random subsets and bootstrap or subsampling stability analyses.
- Common feature sets and overall-fitted scaling across subgroups.
- Alternative preprocessing, categorical weighting, algorithms, and feature selections.
- Temporal subsets to assess whether patterns persist over time.
- Held-out assignment using fixed prototypes from the original solution.
- External validation in independent geographic or health-system populations.
- Associations with clinically relevant outcomes not used to derive the clusters.

Clusters should be treated as provisional, data-driven groupings whose meaning and usefulness depend on the research question, population, and analytic choices.
