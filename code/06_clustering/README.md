# Plant and Leaf Dataset Clustering

PCA-based analysis and clustering pipeline for plant phenotyping time-series data. The notebook performs PCA at two levels (individual timepoints and flattened tracks), generates animated scatter-plot frames colored by cluster or ecotype, and runs KMeans clustering in track PCA space.

## Notebook

### `plant_leaf_clustering.ipynb`

The notebook is organized into three main sections. Each analysis section is self-contained and can be run independently after executing the shared cells above it.

#### Shared Utilities

Color palettes, PCA animation frame generation, track flattening, and plotting helpers.

#### 01 — Plant-Level Analysis

Loads feature data from a pickle file containing per-genotype arrays shaped `(time, instance, feature)`, reshapes into tall format, and runs the full PCA + clustering pipeline.

**Input:** `input/all_stats_ds{N}_sigma2.pickle`

**Outputs:**
- `weights_per_each_track_timepoints_PCA.csv` — PCA loadings (per-timepoint analysis)
- `tall_data_clustered_by_inter_track_distance.csv` — per-timepoint PCA scores with cluster labels
- `weights_per_track_PCA.csv` — PCA loadings (track-level analysis)
- `cl_by_dist_in_PCA_space.png` — tracks colored by distance-based clusters
- `cl_after_track_PCA.png` — tracks colored by KMeans clusters
- `clustering_tracks_no_distance.csv` — final track-to-cluster assignments
- `PCA_{i}_{j}_clustered/` — animation frames colored by cluster
- `PCA_{i}_{j}_ecotypes/` — animation frames colored by ecotype

#### 02 — Leaf-Level Analysis

Loads aggregated leaf data from CSV, drops standard-deviation columns, and runs the same pipeline with the addition of interactive 3D PCA scatter plots.

**Input:** `input/leaf_ds{N}_aggregated.csv`

**Outputs:**
- `weights_per_each_track_timepoints_PCA.csv` — PCA loadings (per-timepoint analysis)
- `leaves_DS{N}_tall_data_clustered_by_inter_track_distance.csv` — PCA scores with cluster labels
- `Leaves_DS{N}_weights_combined_track_distances.csv` — PCA loadings (track-level analysis)
- `cl_by_dist_in_PCA_space.png`, `cl_after_track_PCA.png` — cluster scatter plots
- `clustering_tracks_no_distance.csv` — final track-to-cluster assignments
- `PCA_{i}_{j}_clustered/`, `PCA_{i}_{j}_ecotypes/` — animation frames

## External Dependencies

Both analysis sections expect three variables to be defined before running:

| Variable | Type | Description |
|---|---|---|
| `cluster_info` | `pd.DataFrame` | Has a `"name"` column with `"genotype\|instance"` strings, indexed by cluster ID |
| `cl_names` | `list[str]` | Display names for each cluster (used in plot legends) |
| `n_clusters` | `int` | Number of clusters for KMeans |

## Requirements

- Python 3.8+
- numpy
- pandas
- matplotlib
- scikit-learn
