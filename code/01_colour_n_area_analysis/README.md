# Colour Analysis Pipeline

Pre-processing pipeline for RGB image analysis of *Arabidopsis thaliana* plant phenotyping data. The notebook processes raw pot images and segmentation masks to produce merged image strips and K-means colour distribution pie charts.

## Directory Structure

```
01_colour_n_area_analysis
├── colour_analysis_pp1.ipynb      # Main pipeline notebook
├── requirements.txt
├── output/                        # Inpainted soil images + merged strips
│   ├── Col_0_DS2/
│   │   ├── rep_XX/                # Individual inpainted soil images
│   │   ├── rep_XX_soil_merged.png # Merged soil strip per replicate
│   │   └── rep_XX_plant_merged.png# Merged plant strip per replicate
│   └── Ede-1_DS1/
└── out_pie_charts/
    ├── plant_pie_chart/           # K-means colour pie charts for merged plant strips
    │   ├── Col_0_DS2/
    │   └── Ede-1_DS1/
    └── soil_pie_chart/            # K-means colour pie charts for soil images
        ├── Col_0_DS2/
        └── Ede-1_DS1/
```

## Pipeline Steps

The notebook runs five steps in order:

1. **Inpainting** -- Uses segmentation masks to remove plant canopies from raw images via OpenCV inpainting (`cv2.INPAINT_TELEA`), producing soil-only images in `output/`.

2. **Merge plant images** -- Stacks all segmented plant images for each replicate into a single horizontal strip, saved to `output/<ecotype>/rep_XX_plant_merged.png`.

3. **Plant colour distribution** -- Generates K-means pie charts from the merged plant strips, saved to `out_pie_charts/plant_pie_chart/<ecotype>/`.

4. **Merge soil images** -- Stacks all inpainted soil images for each replicate into a horizontal strip, saved to `output/<ecotype>/rep_XX_soil_merged.png`.

5. **Soil colour distribution** -- Generates K-means pie charts from each individual soil image, saved to `out_pie_charts/soil_pie_chart/<ecotype>/`.

## Configuration

Key parameters in the notebook's configuration cell:

| Parameter | Default | Description |
|---|---|---|
| `ECOTYPE` | `None` | Set to a specific ecotype name to process only that one, or `None` for all |
| `PIE_CHART_CLUSTERS` | `8` | Number of K-means clusters for colour analysis |
| `CROP_FRACTION` | `0.18` | Fraction of width/height cropped from each edge during inpainting |

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Core dependencies: `opencv-python`, `scikit-learn`, `matplotlib`, `Pillow`, `numpy`, `natsort`, `tqdm`.

## Usage

Open `colour_analysis_pp1.ipynb` in JupyterLab or VS Code and run cells sequentially. Each pipeline step can be run independently after the imports, configuration, and utility function cells have been executed.
