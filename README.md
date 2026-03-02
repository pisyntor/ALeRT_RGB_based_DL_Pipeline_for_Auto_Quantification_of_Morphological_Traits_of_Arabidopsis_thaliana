# ALeRT: An RGB-based deep learning pipeline for automated phenotyping of rosette and leaf traits in *A. thaliana*

![Python](https://img.shields.io/badge/Python-≥3.7-blue?logo=python&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green?logo=opencv&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-latest-orange?logo=scikit-learn&logoColor=white)
![License](https://img.shields.io/badge/license-MIT-brightgreen)
[![run with conda](https://img.shields.io/badge/run%20with-conda-3EB049?labelColor=000000&logo=anaconda)](https://docs.conda.io/en/latest/)

# Introduction

This repository provides the full source code for an RGB-based image pipeline for the extraction and analysis of morphological traits in *A. thaliana* at both the rosette and leaf levels.

![image](images/img1_cbr.png)

# Description

![image](images/rgb_pipeline2b.png)

**Figure 1.** The complete workflow of the developed RGB image analysis pipeline for extracting and analysing various morphological traits of *A. thaliana* plants.

The pipeline consists of several stages:

- **A** – Data collection
- **B** – Data preparation using auto-tray and auto-pot cropping
- **C** – Annotation of whole plant canopies and individual leaves
- **D** – Analysis of ecotype replicates based on growth and plant/soil colour distribution
- **E** – Data splitting
- **F** – Data augmentation based on affine transformations and colour jittering
- **G** – Deep Learning-based rosette segmentation
- **H** – Deep Learning-based leaf segmentation and tracking
- **I** – Basic geometrical plant traits used to calculate basic and derived numerical traits
- **J** – Basic geometrical leaf traits used to calculate basic and derived numerical traits
- **K** – Storage of extracted numerical plant- and leaf-level traits for each dataset
- **L** – K-means clustering on the PCA-transformed datasets

# Modules

## 01 — Colour and Area Analysis (Stages A–D)

Pre-processing of raw RGB images to isolate soil backgrounds, merge time-series images into composite strips, and visualise colour distributions.

The consolidated notebook `01_pre_processing/pre_processing.ipynb` is the recommended entry point. It combines all four pre-processing stages into a single, sequential workflow:

| Step | Description | Original scripts |
|------|-------------|------------------|
| 1 — Inpainting | Remove plant canopies via `cv2.inpaint` (Telea) using binary masks, with morphological dilation/erosion | `opencv_inpaint.py` |
| 2 — Merge plant images | Stack segmented plant images into horizontal composite strips | `merge_plant.py`, `pre_process1.py` |
| 3 — Merge soil images | Crop and stack inpainted soil images into composite strips | `merge_soil.py`, `pre_process_2.py` |
| 4 — Colour distribution | Generate K-means pie charts of colour proportions | `pie_chart.py` |

`02_example_output/` contains sample inpainted output images for reference.

## 02 — Rosette Segmentation (Stages E–G)

Deep Learning-based segmentation of whole plant rosettes. Located in `02_segmentation_new/`. Includes training and inference for classical encoder-decoder models and SAM:

| Notebook | Description |
|----------|-------------|
| `training_and_SAM_fine_tuning.ipynb` | **Part 1** — Train U-Net / DeepLab / SegFormer models with `segmentation_models_pytorch`. **Part 2** — Fine-tune the Segment Anything Model (SAM) via HuggingFace |
| `mask_generation_inference.ipynb` | **Part A** — Run classical model inference to generate masks. **Part B** — Run fine-tuned SAM inference to generate masks |

Pre-trained weights are stored under `02_segmentation_new/models/` (`classic_models/` and `SAM1_models/`). See `02_segmentation_new/notebooks/readme.md` for detailed configuration instructions.

## 03 — Leaf Segmentation and Tracking (Stage H)

Training and inference for leaf-level instance segmentation and multi-object tracking. Covers classical encoder-decoder models and SAM fine-tuning. All paths and options are set in configuration cells at the top of each notebook.

| Notebook | Description |
|----------|-------------|
| `01_03_merged_training_and_SAM_fine_tuning.ipynb` | **Part 1** — Train U-Net / DeepLab / SegFormer models with `segmentation_models_pytorch`. **Part 2** — Fine-tune SAM via HuggingFace |
| `02_04_merged_mask_generation_inference.ipynb` | **Part A** — Run classical model inference to generate masks. **Part B** — Run fine-tuned SAM inference to generate masks |
| `03_inference.ipynb` | Leaf segmentation inference and tracking (YOLO, SAM2, Detectron2, BoxMOT) |

**Quick reference:**

| Goal | Notebook | Section |
|------|----------|---------|
| Train U-Net / DeepLab / SegFormer | `01_03` | Part 1 — Configuration |
| Fine-tune SAM | `01_03` | Part 2 — Configuration |
| Run classical model on images | `02_04` | Part A — Configuration |
| Generate masks with SAM | `02_04` | Part B — Configuration |

See `03_readme.md` for detailed configuration parameters and usage instructions.

## 04 — PlantInspector: GUI-Based Plant Traits Extraction (Stage I)

PlantInspector is a Windows desktop application for quantitative analysis of plant image datasets. It expects pre-processed input (binary masks + segmented images from earlier pipeline stages) and produces structured quantitative outputs.

**Core capabilities:**
- **Geometrical trait visualisation** — overlays on plant images (perimeter, convex hull, bounding box/circle/ellipse)
- **Numerical trait tracking** — time-series plots and bar charts of rosette area, eccentricity, roundness, growth rates, etc.
- **Batch processing** — process entire ecotype datasets with multiple replicates in one pass
- **Ground truth creation** — structured output for training ML models

**Input format:** `dataset/ → ecotype/ → replicate/ → masks/ + segmented_images/`

**Output structure:**

| Subfolder | Content |
|-----------|---------|
| `_Plots/` | PNG line plots per trait |
| `_PlotsHtml/` | Interactive HTML plots (pannable/zoomable) |
| `_Bars/` | PNG bar charts (ecotype-level and replicate-level) |
| `_Excels/` | Raw numerical trait data as Excel files |
| `_Saved_Lists/` | Geometrical traits and analysis parameters |

See `04_gui_summary.md` for a detailed summary of the PlantInspector user manual.

## 05 — Leaf Traits Extraction (Stage J)

Calculation of basic geometrical leaf traits and derived numerical measurements from segmented leaf instances. Located in `05_leaf_traits_extraction/`.

| Notebook | Description |
|----------|-------------|
| `01_leaf_traits_calculations.ipynb` | Calculate leaf-level morphological traits from segmentation masks |
| `02_calculated_leaf_tratis_plots.ipynb` | Plot calculated leaf trait data |
| `02b_SAMPLE_leaf_traits_output.ipynb` | Sample output demonstrating leaf trait results |
| `03_plotting_leaf_traits.ipynb` | Additional leaf trait visualisations |
| `03b_SAMPLE_leaf_trait_plots.ipynb` | Sample output demonstrating leaf trait plots |

## 06 — Clustering (Stages K–L)

K-means clustering on PCA-transformed trait datasets to identify phenotypic groupings at both plant and leaf levels.

| Notebook | Description |
|----------|-------------|
| `06_plant_leaf_clustering.ipynb` | Cluster plant-level and leaf-level trait datasets |

# Usage

## Requirements

- [Python >= 3.7](https://www.python.org/downloads/)
- [PyTorch >= 1.4](https://pytorch.org/get-started/locally/)
- [matplotlib](https://pypi.org/project/matplotlib/)
- [numpy](https://pypi.org/project/numpy/)
- [pandas](https://pypi.org/project/pandas/)
- [Pillow](https://pypi.org/project/pillow/)
- [opencv-python](https://pypi.org/project/opencv-python/)
- [scikit-learn](https://pypi.org/project/scikit-learn/)
- [natsort](https://pypi.org/project/natsort/)
- [tqdm](https://pypi.org/project/tqdm/)
- [Jupyter Notebook](https://jupyter.org/)
- [Visual Studio Code 2022](https://code.visualstudio.com/download)
- [.NET Framework >= 4.8](https://dotnet.microsoft.com/en-us/download/dotnet-framework/net48)


## Setting Up

We recommend using [Conda](https://conda.io/projects/conda/en/latest/user-guide/install/index.html) to manage environments.

```bash
conda create -n env_name python=3.10
conda activate env_name
pip install -r requirements.txt
```

# Structure

```console
├── 01_pre_processing/                      # Module 01 — Colour & area analysis
│   ├── pre_processing.ipynb                #   Consolidated notebook (recommended)
│   ├── opencv_inpaint.py                   #   Legacy script — inpainting
│   ├── merge_plant.py                      #   Legacy script — merge plant strips
│   ├── merge_soil.py                       #   Legacy script — merge soil strips
│   └── pie_chart.py                        #   Legacy script — colour pie charts
│
├── 02_example_output/                      # Module 01 — Sample inpainted images
│
├── 02_segmentation_new/                    # Module 02 — Rosette segmentation
│   ├── notebooks/
│   │   ├── training_and_SAM_fine_tuning.ipynb
│   │   └── mask_generation_inference.ipynb
│   ├── models/
│   │   ├── classic_models/
│   │   └── SAM1_models/
│   └── input_data/
│
├── 03_inference.ipynb                      # Module 03 — Leaf segmentation inference
│
├── 04_gui_summary.md                       # Module 04 — PlantInspector summary
│
├── 05_leaf_traits_extraction/              # Module 05 — Leaf traits
│   ├── 01_leaf_traits_calculations.ipynb
│   ├── 02_calculated_leaf_tratis_plots.ipynb
│   ├── 02b_SAMPLE_leaf_traits_output.ipynb
│   ├── 03_plotting_leaf_traits.ipynb
│   └── 03b_SAMPLE_leaf_trait_plots.ipynb
│
├── 06_plant_leaf_clustering.ipynb          # Module 06 — Clustering
│
├── input/                                  # Raw images and segmentation masks
│   ├── raw_input/
│   └── segmented_imgs/
│
├── soil_and_plant_crop1.py                 # Standalone crop script
└── requirements.txt
```

# License

This repository is distributed under [LICENSE](LICENSE).
