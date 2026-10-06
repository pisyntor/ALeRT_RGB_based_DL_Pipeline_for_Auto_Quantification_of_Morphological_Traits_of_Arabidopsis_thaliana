# Rosette Segmentation Notebooks

This module provides notebooks for training rosette segmentation models, generating masks, and scoring a fine-tuned SAM-1 checkpoint against existing reference masks. All paths and options are set in **configuration cells** at the top of each part so you can run experiments without editing code throughout the file.

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
  - **Metrics:** the test metrics are foreground IoU based — `TP / (TP + FP + FN)`, so agreement on background is never counted. Both the pooled figure (`IoU (Jaccard Index)`) and the mean per image (`IoU_plant`) are reported. They diverge sharply on a germination time series, where the pooled figure is dominated by large rosettes and a missed seedling is nearly invisible to it.

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

## 3. `SAM1_seg_eval.ipynb`

**Evaluation: score a fine-tuned SAM-1 checkpoint against existing reference masks.**

For each test image, a fixed classical model predicts a mask, a bounding box is taken around that prediction, and the original RGB image plus that box are passed to a fixed SAM-1 checkpoint. SAM's mask is then scored against the reference mask that already exists for the image. Nothing is trained here, and neither model sees a reference mask before it predicts.

**Order of use.** Each code cell sits under the heading that describes it. Run them top to bottom the first time: the imports and helpers cell, then **Shared model configuration**, **Define the dataset runner**, and **Load the two fixed checkpoints**. Those four are run once per session; run **Load the two fixed checkpoints** again after changing anything in the configuration. Then run **Predict and score DS1**, **Predict and score DS2**, or both, and finally **Evaluation metrics**, which reads saved results only and so also runs in a fresh kernel.

### Shared model configuration

| Setting | What it is |
|---------|------------|
| `CLASSICAL_MODEL_PATH` | the trained classical checkpoint whose prediction supplies the box prompt |
| `CLASSICAL_ENCODER_NAME` | encoder name matching that checkpoint's keys — see the encoder prefix note below |
| `CLASSICAL_CHECKPOINT_FORMAT` | `full_model` for `torch.save(model)`, `state_dict` for weights only. The `.pth` extension does not identify which one you have |
| `CLASSICAL_IMAGE_SIZE`, `CLASSICAL_MEAN`, `CLASSICAL_STD` | resize and normalisation used when that model was trained |
| `INFERENCE_THRESHOLD` | probability cutoff for the classical mask |
| `SAM_CHECKPOINT_PATH`, `SAM_MODEL_NAME` | the fine-tuned SAM state dict, and the base configuration and processor it matches |
| `SAM_THRESHOLD` | probability cutoff for SAM's mask |
| `SAM1_OUTPUT_ROOT` | parent results folder; each dataset writes to its own `sam1_ds*` subfolder |
| `LABEL_MASK_SUFFIX`, `LABEL_RESIZE` | how reference masks are named and sized — see below |

### Predict and score DS1 / DS2

Each dataset cell needs only its own `SPLIT_JSON_DS*`, `BASE_DIR_DS*` (RGB images) and `BASE_DIR_LABEL_DS*` (reference masks). Run one, the other, or both. The split JSON assigns each `<ecotype>/<replicate>` to train, valid or test; every assigned folder must be present, but only `test` images are predicted and scored.

`LABEL_FOREGROUND_THRESHOLD` can also be set at the top of a dataset cell, which is how one dataset can use strict masks while another uses thresholded ones.

Both datasets are scored with whichever pair of checkpoints **Load the two fixed checkpoints** loaded. If you want each dataset scored with its own SAM checkpoint, change `SAM_CHECKPOINT_PATH` and run that cell again between them — otherwise the second dataset is a test of how well the first dataset's model transfers. `summary.json` records the checkpoint path and its SHA-256 for exactly this reason.

### SAM-1 evaluation metrics

Set `METRICS_RUN_DIR` to one completed dataset output folder (`.../sam1_ds1` or `.../sam1_ds2`) and run this section once per dataset you want to report. It reads that folder's saved CSV and summary, so it needs no checkpoints and runs in a fresh kernel.

### Reference masks

Masks are white plant on black soil; non-zero is plant. Three settings control how one is read. They affect scoring only — no model ever sees a reference mask.

| Setting | Default | Meaning |
|---------|---------|---------|
| `LABEL_MASK_SUFFIX` | `'_mask'` | naming under `<label_root>/<ecotype>/<replicate>/masks/`. `'_mask'` for `<image_stem>_mask.png`, `''` for `<image_stem>.png`. A missing mask reports the names the folder actually holds |
| `LABEL_FOREGROUND_THRESHOLD` | `None` | `None` requires every pixel to be 0, 1 or 255 and counts non-zero as plant. An integer thresholds instead, for masks with anti-aliased edges |
| `LABEL_RESIZE` | `'nearest'` | rescales a mask that is not the same size as its photo. Nearest neighbour adds no new grey values, so a binary mask stays binary. `None` stops the run instead |

A file whose values are none of 0, 1 or 255 **stops the run** rather than being thresholded into a plausible score. Such a file is more often the wrong file than a soft-edged mask: a `segmented_images` file or an RGB photo reached through the wrong path looks exactly like that, and thresholding one produces a mask-shaped result that scores non-zero and looks fine.

`inspect_labels.py` tells the two apart without stopping on a bad file:

```bash
python inspect_labels.py <label_root> <image_root>
```

It prints each mask's stored mode, size, value profile and a verdict, reports the naming in use, and says whether a mask's size matches its photo. A real mask keeps its largest group of foreground pixels at exactly 255, however soft its edges; a photograph peaks somewhere mid-range and has no 255 group at all.

Rescaling is recorded, not announced: nothing is printed per image, and the count appears as `label_resized` per row and `resized_label_images` in the summary. Note that resampling a reference mask does cost accuracy at the boundary, which matters most on small plants, so a dataset that needed rescaling is not strictly comparable with one that did not.

### Outputs

Each dataset writes to `SAM1_OUTPUT_ROOT/sam1_ds*/`:

| Output | Content |
|--------|---------|
| `<ecotype>/<replicate>/unet_masks`, `sam_masks`, `segmented_images` | original-resolution PNGs. The source extension is kept in the output name so two images with the same stem cannot collide |
| `per_image.csv` | one row per test image: image and mask paths, the box, empty-box flag, SAM TP/FP/FN/TN, SAM IoU, the classical model's IoU and its TP/FP/FN, `gt_ambiguous_px`, `sam_iou_gt_nonzero`, `label_resized` |
| `summary.json` | status, image and empty-box counts, mean and pooled foreground IoU for both models, the label settings actually applied, and SHA-256 hashes of both checkpoints and the split JSON |
| `additional_metrics.json` / `.csv` | written by **Additional SAM evaluation metrics**: pooled precision, recall, F1, MCC, specificity and G-mean for one completed run |

IoU throughout is **foreground IoU**, `TP / (TP + FP + FN)`: agreement on background is never counted, so an empty prediction against a real plant scores 0 rather than scoring well on a mostly-soil photo. Two empty masks score 1.0.

Report the pooled and per-image figures separately, and say which one a headline number is. `sam_iou_gt_nonzero` is the same prediction scored against the most lenient reading of the mask; if it and `sam_iou` agree, the threshold setting did not affect the result.

### When a run stops

That is intended. The notebook stops rather than skipping an image or substituting data, so a finished run always covers every test image, and a stopped run is marked `failed` in `summary.json` and cannot be mistaken for a complete one. The message names the file that caused it.

**How to use:** fill in **Shared model configuration**, run everything down to **Load the two fixed checkpoints**, then run the dataset section you want — **Predict and score DS1**, **Predict and score DS2**, or both — and finally **Evaluation metrics**.

---

## Quick reference

| Goal                          | Notebook                          | Section to configure    | What you set |
|-------------------------------|-----------------------------------|-------------------------|--------------|
| Train U-Net/DeepLab/PSPNet/SegFormer | `training_and_SAM_fine_tuning`    | Part 1 — Configuration  | Paths, encoders/decoders, batch size, split JSON |
| Fine-tune SAM                 | `training_and_SAM_fine_tuning`    | Part 2 — Configuration  | SAM model name, dataset paths, save path, checkpoint paths |
| Run classical model on images | `mask_generation_inference`       | Part A — Configuration  | Model path, input/output dirs, thresholds, device |
| Generate masks with SAM       | `mask_generation_inference`       | Part B — Configuration  | SAM checkpoint, dataset paths, output folder names |
| Score SAM-1 against reference masks | `SAM1_res_evaluation`       | Shared model configuration, then the dataset section | Both checkpoints, dataset paths, mask naming and sizing |

---

## Environment for Running Mask Generation 

Please use `smp==0.3.3` for running mask generation. As the weights are trained using version `0.3.3` of the segmentation library, you will need to have a matching version to run the models. You can install the requirements by running `pip install -r mask_generation_requirements.txt`.

## Encoder naming: the specific prefix

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
