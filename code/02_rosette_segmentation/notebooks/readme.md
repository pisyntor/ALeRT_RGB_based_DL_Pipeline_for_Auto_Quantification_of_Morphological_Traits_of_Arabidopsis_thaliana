# Rosette Segmentation Notebooks

This module provides two main notebooks for training rosette segmentation models and generating masks. All paths and options are set in **configuration cells** at the top of each part so you can run experiments without editing code throughout the file.

---

## 1. `training_and_SAM_fine_tuning.ipynb`

**Training: classical segmentation models and SAM fine-tuning.**

- **Setup (run once)**
  Shared imports and utilities for both parts.

- **Part 1 — Classical segmentation**
  Train U-Net / DeepLabv3+ / PSPNet / SegFormer-style models with `segmentation_models_pytorch` on a single dataset:
  - **Part 1 — Configuration:** `ROOT_PATH`, `MODEL_SAVE_DIR`, `TMP_SAVE_DIR`, `PART1_SPLIT_JSON`, `PART1_BASE_DIR`, `PART1_BASE_DIR_LABEL`, encoder/decoder lists, `PART1_BATCH_SIZE`, `PART1_AUGMENTATION`
  - `MODEL_SAVE_DIR` and `TMP_SAVE_DIR` are derived from `ROOT_PATH` — change only `ROOT_PATH` to relocate all outputs.
  - Custom dataset, data loaders, training loop, evaluation, and optional **SegFormer** section (different encoders/decoders)
  - Training results are saved to an Excel file (`excel_file`) after each run.

- **Part 2 — SAM fine-tuning**
  Fine-tune the Segment Anything Model (HuggingFace) on one or two datasets:
  - **Part 2 — Configuration:** `SAM_MODEL_NAME`, `SAM_SAVE_PATH_BASE`, `SAM_BATCH_SIZE`, split JSONs and base dirs for Dataset-1/2 (`SAM_SPLIT_JSON_DS*`, `SAM_BASE_DIR_DS*`), checkpoint paths (`SAM_CHECKPOINT_DS1`, `SAM_CHECKPOINT_DS2`)
  - SAM dataset/dataloaders, training, evaluation, and visualization

**How to use:** Run Setup, then run **Part 1 — Configuration** or **Part 2 — Configuration** and the sections you need (Part 1 only, Part 2 only, or both).

---

## 2. `mask_generation_inference.ipynb`

**Inference: generate segmentation masks from trained models.**

- **Part A — Classical model inference**
  Run a saved `.pt` model (from the training notebook) on a folder of images:
  - **Part A — Configuration:** `CLASSICAL_MODEL_PATH`, `CLASSICAL_INPUT_DIR`, `CLASSICAL_OUTPUT_DIR`, `CLASSICAL_IMAGE_SIZE`, normalization mean/std, `INFERENCE_THRESHOLD`, `NOISE_REMOVAL_THRESHOLD`, `CLASSICAL_DEVICE`
  - Load model, iterate over images, optional contour-based noise removal, write masks and segmented images to disk

- **Part B — SAM mask generation**
  Run a fine-tuned SAM checkpoint on test dataloaders and save predicted masks:
  - **Part B — Configuration:** `SAM_MODEL_NAME`, `SAM_CHECKPOINT_PATH`, split JSONs and base dirs for Dataset-1/2/3, `OUTPUT_NAME_DS1`/`DS2`/`DS3`, `SAM_BATCH_SIZE`
  - Load SAM model and checkpoint, run `save_predicted_masks` for each dataset

**How to use:** Run **Part A — Configuration** or **Part B — Configuration**, then the corresponding load-model and inference cells. You can run Part A only, Part B only, or both.

---

## Quick reference

| Goal                          | Notebook                          | Section to configure    | What you set |
|-------------------------------|-----------------------------------|-------------------------|--------------|
| Train U-Net/DeepLab/SegFormer | `training_and_SAM_fine_tuning`    | Part 1 — Configuration  | Paths, encoders/decoders, batch size, split JSON |
| Fine-tune SAM                 | `training_and_SAM_fine_tuning`    | Part 2 — Configuration  | SAM model name, dataset paths, save path, checkpoint paths |
| Run classical model on images | `mask_generation_inference`       | Part A — Configuration  | Model path, input/output dirs, thresholds, device |
| Generate masks with SAM       | `mask_generation_inference`       | Part B — Configuration  | SAM checkpoint, dataset paths, output folder names |

---

## Environment for Running Mask Generation 

Please use `smp==0.3.3` for running mask generation. As the weights are trained using version `0.3.3` of the segmentation library, you will need to have a matching version to run the models. You can install the requirements by running `pip install -r mask_generation_requirements.txt`.

## Encoder naming: the `tu-` prefix

In `segmentation_models_pytorch`, encoders sourced from the [`timm`](https://github.com/huggingface/pytorch-image-models) (PyTorch Image Models) library are accessed using the **`tu-`** prefix. This gives access to hundreds of state-of-the-art architectures beyond the encoders bundled natively with smp.

**When you need it:** any encoder name that begins with `tu-` (e.g. `tu-efficientnet_b5`, `tu-regnetx_160`) must be specified with that prefix — without it, smp will not recognise the model.

**Older `timm-` prefix:** the encoder lists in the notebook were originally written with the legacy `timm-` prefix (e.g. `timm-efficientnet-b5`). If you encounter a `KeyError` or "encoder not found" error on a `timm-` name, replace the prefix with `tu-` and verify the exact model name against the timm model registry:

```python
import timm
timm.list_models('efficientnet*')   # find the canonical name
```

Then use it in smp as:

```python
encoders = ['tu-efficientnet_b5', 'tu-regnetx_160', ...]
```

> **Note:** `tu-` encoder names also tend to use underscores rather than hyphens within the model name (e.g. `tu-efficientnet_b5`, not `tu-efficientnet-b5`). Check the timm registry for the exact string.

---

## Dependencies

This project works best with Python `3.10`, Pytorch `2.1.1`, and CUDA `12.1.1`.

- PyTorch, torchvision, `segmentation_models_pytorch`
- OpenCV, NumPy, Pandas, scikit-learn, imgaug, tqdm, matplotlib
- For **Part 2 (SAM)** in either notebook: `transformers`, `datasets`, MONAI (install via the optional pip cell in the notebook)
