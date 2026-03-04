## GUI for extraction and visualisation of basic and derived rosette traits

GUI is a fast, desktop-based application designed for quantitative analysis of plant image datasets. It sits at the intersection of **computer vision**, **plant phenotyping**, and **machine learning data preparation**. It is not a general-purpose image editor — it expects pre-processed input (binary masks + segmented images) and produces structured quantitative outputs.

Its four core purposes are:
1. **Geometrical trait calculation and display** — visual overlays on plant images (e.g., perimeter, convex hull, bounding shapes)
2. **Numerical trait evolution tracking** — time-series plots and bar charts of plant metrics
3. **Batch/dataset automation** — processing entire ecotype datasets with multiple replicate folders in one pass
4. **Ground truth dataset creation** — for training Machine Learning models on plant analysis tasks

---

## How It Fits the Broader Project

This manual is almost certainly **one document in a larger ecosystem** that includes:

- **Pre-processing pipeline** — GUI requires input data to already be structured into `masks/` and `segmented_images/` subfolders per replicate. The pre-processing step that creates this structure is not described here, implying it is covered elsewhere (possibly another manual, script, or tool in the project).
- **Two plant datasets** — the GUI explicitly supports switching between "plant dataset 1" and "plant dataset 2", suggesting the project involves at least two distinct experimental setups with different calibration factors and sowing/screening dates.
- **Machine Learning pipeline** — the app is described as a source of training/testing/validation data for ML models. The ML models themselves are separate components. This manual positions GUI as an upstream data-generation tool for those models.
- **Output files consumed downstream** — results are saved to structured subfolders (`_Plots/`, `_PlotsHtml/`, `_Bars/`, `_Excels/`, `_Saved_Lists/`). These outputs likely feed into downstream analysis scripts, reports, or ML workflows that are part of the broader project but not described here.

---

## Key Technical Details

### Input Data Format
- Folder hierarchy: `dataset/ → ecotype/ → replicate/ → masks/ + segmented_images/`
- Both `masks/` and `segmented_images/` share the same filenames (time-series images)
- Input path must point to the dataset root folder exactly

### GUI Structure (4 Regions)
| Region | Role |
|--------|------|
| A | Root/output path selection, dataset type toggle (Dataset 1 / Dataset 2), current file display |
| B | Ecotype/replicate/time-point selection; calibration factor; sowing & screening dates; display tab control (Image Type, Plots, Bar Charts, Raw Data); scope of processing; Process button |
| C | Main canvas — displays images, overlaid traits, plots, bar charts, raw data tables |
| D | Trait selection, display colour controls, plot/bar image size, replicate colours, Save Image and Exit buttons |

### Computed Geometrical Traits (visualised as overlays)
- Perimeter, Convex Hull, Convex Hull Equivalent Circle
- Width, Bounding Box, Bounding Circle, Bounding Ellipse

### Computed Numerical Traits (time-series)
- Rosette Area, Perimeter, Eccentricity, Roundness, Rotational Mass Asymmetry
- Area Growth Rate, Convex Hull Area Growth Rate (supports negative values for leaf loss/occlusion)

### Output Files Generated
| Subfolder | Content |
|-----------|---------|
| `_Plots/` | PNG line plots per trait |
| `_PlotsHtml/` | Interactive HTML plots (pannable/zoomable, built with Plotly or similar) |
| `_Bars/` | PNG bar charts (ecotype-level and replicate-level) |
| `_Excels/` | Raw numerical trait data as Excel files |
| `_Saved_Lists/` | Geometrical traits and analysis parameters |

Output is cached: if the output folder already contains results, they are loaded directly without reprocessing.

