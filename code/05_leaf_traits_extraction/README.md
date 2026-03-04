# Leaf Traits Extraction

This module extracts morphological traits from segmented leaf and stem masks, computes plant-level metrics, and generates visualisations of trait dynamics over time.

## Overview

The pipeline takes binary leaf/stem mask sequences (produced by earlier segmentation steps) and computes a comprehensive set of geometric and shape-based traits at the individual leaf, replicate, accession, and dataset levels. Results are exported to multi-sheet Excel files and plotted as time-series charts.

## Notebooks

### 01_leaf_traits_calculations.ipynb

Core computation notebook. Reads leaf and stem mask image sequences and calculates:

**Leaf traits:**
| Trait | Description |
|---|---|
| `l_area (mm^2)` | Leaf area (calibrated) |
| `l_perimeter (mm)` | Leaf perimeter |
| `l_roundness` | 4 pi Area / Perimeter^2 |
| `l_circumference (mm)` | Circumference of minimum enclosing circle |
| `l_eccentricity` | Eccentricity from region properties |
| `l_compactness` | Area / Convex hull area |
| `l_extent` | Area / Rotated bounding box area |
| `l_surface_coverage` | Area / Bounding circle area |
| `l_RMA` | Rotational Mass Asymmetry from 2nd central moment ellipse |
| `l_angle (deg)` | Leaf angle relative to plant centre |
| `l_length (mm)` | Petiole length + lamina length |
| `l_lamina_length (mm)` | Length along the estimated leaf vein |
| `l_petiole_length (mm)` | Distance from plant centre to leaf base |
| `l_width (mm)` | Maximum diameter perpendicular to the leaf vein |
| `SOL` | Slenderness of Leaf (lamina_length^2 / area) |
| `l_overlapping (%)` | Percentage of leaf area overlapped by other leaves |

**Plant traits requiring leaf masks:**
| Trait | Description |
|---|---|
| `Isotropy` | Circularity of the plant's combined convex hull |
| `LAI` | Leaf Area Index (total leaf area / pot area) |
| `SOL` | Plant-level Slenderness of Leaves |

**Key algorithms:**
- Plant centre detection via stem mask overlap accumulation
- Leaf vein estimation by interpolating midpoints between left/right contour halves
- Leaf width as the widest perpendicular cross-section to the vein
- Isotropy from pairwise consecutive-leaf convex hulls
- Date alignment to Days After Sowing (DAS) with dataset-specific offsets

**Output structure:** Multi-sheet Excel files at dataset, accession, and replicate levels containing `Leaf_Traits`, `Plant_Traits`, `Rep_Leaf_Traits_AVG`, and `Accession_Leaf_Traits_AVG` sheets.

### 02_calculated_leaf_traits_plots.ipynb

Generates per-replicate visualisation images from leaf/stem masks and an associated `plant_centers.xlsx` file. For each leaf at each time-step it produces:

- **Lobe mask** - coloured leaf region
- **Lobe contour** - leaf contour overlaid on mask
- **Leaf length** - estimated leaf vein drawn from stem to tip
- **Lobe width** - maximum perpendicular diameter
- **Bounding circle** - minimum enclosing circle overlay

### 03_plotting_leaf_traits.ipynb

Batch plotting notebook. Reads the Excel files produced by notebook 01 and generates publication-quality PNG plots:

- **Plant-level traits** per accession: Isotropy, LAI, SOL over DAS (one line per replicate)
- **Leaf-level traits** per replicate: area, perimeter, roundness, length, width, compactness, extent, eccentricity, RMA, SOL, surface coverage over DAS (one line per leaf)
- **Leaf overlapping** bar charts per individual leaf
- **Leaf angle** polar plots per replicate and per individual leaf
- **Accession averages** (across replicates) for all leaf traits over DAS

Short-lived leaves/replicates (< 9 days, fallback 4 days) are excluded from plots.


## Input Requirements

- **Leaf masks:** Binary mask image sequences organised as `<dataset>/<accession>/<replicate>/Leaf_NNN/[leaf seq/]hidden leaf mask seq/*.png`
- **Stem masks:** Binary mask image sequences at `<dataset>/<accession>/<replicate>/Leaf_NNN/stem seq/hidden stem mask seq/*.png`
- **Calibration factors:** Pixel-to-mm conversion factors per dataset (configured in notebook 01)
- **Plant centres file:** `plant_centers.xlsx` with Date, Time, X, Y columns (for notebook 02)

## Dependencies

- numpy, pandas, opencv-python (cv2), scipy, scikit-image, matplotlib, joblib, distinctipy

## Configuration

Key parameters in notebook 01:

```python
calibration_factors = {'leaf_dataset1': 0.13715, 'leaf_dataset2': 0.14690}  # px to mm
pot_areas_calibrated = {...}  # pot area in mm^2 for LAI calculation
parallelize = True  # enable joblib parallelisation
leaf_seq_in_path = False  # whether path includes 'leaf seq' subfolder
```
