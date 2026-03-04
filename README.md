# ALeRT: An RGB-based deep learning pipeline for automated phenotyping of rosette and leaf traits in *Arabidopsis thaliana*

![Python](https://img.shields.io/badge/Python-≥3.7-blue?logo=python&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green?logo=opencv&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-latest-orange?logo=scikit-learn&logoColor=white)
![License](https://img.shields.io/badge/license-MIT-brightgreen)
[![run with conda](https://img.shields.io/badge/run%20with-conda-3EB049?labelColor=000000&logo=anaconda)](https://docs.conda.io/en/latest/)

This repository provides the full source code for an RGB-based image pipeline for the extraction and analysis of morphological traits in *A. thaliana* at both the rosette and leaf levels.

![image](images/img1_cbr.png)

# Description

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

![image](images/rgb_pipeline2b.png)

**Figure 1.** The complete workflow of the developed ALeRT RGB image analysis pipeline for extracting and analysing various morphological traits of *A. thaliana* plants.

# Modules

## 01 — Colour and Area Analysis (Stages A–E)

Pre-processing of raw RGB images to isolate soil backgrounds, merge time-series images into composite strips, and visualise colour distributions.

The consolidated notebook `01_pre_processing/pre_processing.ipynb` is the recommended entry point. It combines all four pre-processing stages into a single, sequential workflow:

| Step | Description | 
|------|-------------|
| 1 — Inpainting | Remove plant canopies via `cv2.inpaint` (Telea) using binary masks, with morphological dilation/erosion | 
| 2 — Merge plant images | Stack segmented plant images into horizontal composite strips | 
| 3 — Merge soil images | Crop and stack inpainted soil images into composite strips | 
| 4 — Colour distribution | Generate K-means pie charts of colour proportions |

`02_example_output/` contains sample inpainted output images for reference.

## 02 — Rosette Segmentation (Stages F–G)

Deep Learning-based segmentation of whole plant rosettes. Located in `02_segmentation_new/`. Includes training and inference for classical encoder-decoder models and SAM:

| Notebook | Description |
|----------|-------------|
| `training_and_SAM_fine_tuning.ipynb` | **Part 1** — Train U-Net / DeepLab / SegFormer models with `segmentation_models_pytorch`. **Part 2** — Fine-tune the Segment Anything Model (SAM) via HuggingFace |
| `mask_generation_inference.ipynb` | **Part A** — Run classical model inference to generate masks. **Part B** — Run fine-tuned SAM inference to generate masks |

Pre-trained weights are stored under `02_segmentation_new/models/` (`classic_models/` and `SAM1_models/`). See `02_segmentation_new/notebooks/readme.md` for detailed configuration instructions.

**Quick reference:**

| Goal | Notebook |
|------|----------|
| Train U-Net / DeepLab / SegFormer | Part 1 — Configuration |
| Fine-tune SAM | Part 2 — Configuration |
| Run classical model on images | Part A — Configuration |
| Generate masks with SAM | Part B — Configuration |

## 03 — Leaf Segmentation and Tracking (Stage H)

Leaf-level instance segmentation and multi-object tracking. The module is split into three areas: **data preparation**, **model training**, and **inference**. All paths and options are set in configuration cells at the top of each notebook.

### Inference

| Notebook | Description |
|----------|-------------|
| `inference.ipynb` | **Key notebook.** End-to-end leaf segmentation inference and tracking. Sections 1–5 run a single pipeline (YOLO or YOLO+SAM2). Section 6 performs a multi-model tracking sweep across YOLOv8, YOLOv11, SAM2, and Detectron2 with BoxMOT trackers (ByteTrack, DeepOcSort, BotSort, StrongSort) |

### Data Preparation (`data_prep/`)

| Notebook | Description |
|----------|-------------|
| `yolo_labels/yolo_ds_labels.ipynb` | Build train/val/test splits from raw images and masks, extract leaf contours, and save YOLO-format segmentation labels and a `ds.csv` dataset file |
| `coco_labels/convert_yolo_to_coco_labels.ipynb` | Convert YOLO annotations to COCO JSON format for Detectron2 training |

### Model Training (`seg_train_models/`)

| Notebook | Description |
|----------|-------------|
| `train_yolo_v8.ipynb` | Train a YOLOv8 segmentation model |
| `train_yolo_v11.ipynb` | Train a YOLOv11 segmentation model |
| `train_detectron2.ipynb` | Train a Detectron2 Mask R-CNN model |
| `train_sam2.ipynb` | Fine-tune SAM 2 for leaf segmentation |

**Quick reference:**

| Goal | Notebook |
|------|----------|
| Run inference on new images | `inference.ipynb` — Section 1 (parameters) |
| Prepare a YOLO training dataset | `data_prep/yolo_labels/yolo_ds_labels.ipynb` |
| Convert labels to COCO format | `data_prep/coco_labels/convert_yolo_to_coco_labels.ipynb` |
| Train YOLO v8 / v11 | `seg_train_models/train_yolo_v8.ipynb` or `train_yolo_v11.ipynb` |
| Train Detectron2 | `seg_train_models/train_detectron2.ipynb` |
| Fine-tune SAM 2 | `seg_train_models/train_sam2.ipynb` |
| Run multi-model tracking sweep | `inference.ipynb` — Section 6 |

## 04 — GUI-Based Rosette Traits Extraction (Stage I)

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
ALeRT RGB-based deep learning pipeline stages (1 -6):
├── 01_colour_n_area_analysis                # Module 01 — Colour & area analysis
│    └── colour_analysis_pp1.ipynb           #   Consolidated notebook (recommended)
│
├── 02_rosette_segmentation/                    # Module 02 — Rosette segmentation
│   ├── notebooks/
│   │   ├── training_and_SAM_fine_tuning.ipynb
│   │   └── mask_generation_inference.ipynb
│   └── models/
│       ├── classic_models/
│       └── SAM1_models/
│
├── 03_leaf_segmentation_n_tracking        # Module 03 — Leaf instance segmentation and tracking
│   ├── inference.ipynb                    #  Key notebook — inference & tracking
│   ├── helpers.py                         #  Required for running inference.ipynb 
│   ├── data_prep/
│   │   ├── yolo_labels/
│   │   │   └── yolo_ds_labels.ipynb                #   Build splits & YOLO labels
│   │   └── coco_labels/
│   │       └── convert_yolo_to_coco_labels.ipynb   #   YOLO → COCO conversion
│   └── seg_train_models/
│       ├── train_yolo_v8.ipynb
│       ├── train_yolo_v11.ipynb
│       ├── train_detectron2.ipynb
│       └── train_sam2.ipynb
│
├── 04_GUI_based_plant_traits_extraction   # Module 04 — GUI for calculation and visualisation
│                                                         of basic and derived rosette traits 
│
├── 05_leaf_traits_extraction/              # Module 05 — basic and derived leaf traits extraction
│   ├── 01_leaf_traits_calculations.ipynb
│   ├── 02_calculated_leaf_tratis_plots.ipynb
│   ├── 02b_SAMPLE_leaf_traits_output.ipynb
│   ├── 03_plotting_leaf_traits.ipynb
│   └── 03b_SAMPLE_leaf_trait_plots.ipynb
│
└── 06_plant_leaf_clustering.ipynb          # Module 06 — Clustering of Arabidopsis ecotypes using 
                                                          extracted rosette and leaf traits
```

# License

This repository is distributed under [LICENSE](LICENSE).
