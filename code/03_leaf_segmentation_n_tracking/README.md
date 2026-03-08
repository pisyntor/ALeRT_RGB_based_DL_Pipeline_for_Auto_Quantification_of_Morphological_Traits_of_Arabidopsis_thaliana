# Leaf Segmentation and Tracking

This directory contains the codebase for leaf segmentation and tracking, combining functionality from multiple codebases with organised training notebooks for different models.

## Project Structure

```
Leaf instance segmentation and tracking
|
|── models/ # Place your .pt model files here.
├── seg_train_models/       # Training notebooks for different models
│   ├── train_sam2.ipynb
│   ├── train_yolo_v11.ipynb
│   ├── train_yolo_v8.ipynb
│   └── train_detectron2.ipynb
├── data_prep/              # Data preparation scripts
├── thirdparty/             # Third-party dependencies (SAM2, etc.)
├── inference_merged.ipynb  # Unified inference notebook
├── src_helpers.py          # Helper functions
└── README.md               # This file
```

## Installation

### 1. Create Virtual Environment

```bash
python3 -m venv .env
source .env/bin/activate  # On Windows: .env\Scripts\activate
```

### 2. Install Requirements

```bash
python3 -m pip install -r requirements.txt
python3 -m ipykernel install --user
```

### 3. Download Third-Party Dependencies (SAM2)

If you need to train or use SAM2 models, download the SAM2 repository and pretrained checkpoints:

```bash
# Create thirdparty directory if it doesn't exist
mkdir -p thirdparty
cd thirdparty

# Clone SAM2 repository
git clone https://github.com/facebookresearch/segment-anything-2.git
cd segment-anything-2

# Install SAM2
python3 -m pip install -e .

# Download pretrained checkpoints (required for training SAM2)
cd checkpoints
bash download_ckpts.sh
```

After installation, the SAM2 path should be `thirdparty/segment-anything-2/`.

## Training

Training notebooks are organized by model type in the `seg_train_models/` directory:

### Available Training Notebooks:

#### 1. **`train_sam2.ipynb`** - SAM2 Model Training
- **Description**: Fine-tune Segment Anything Model 2 for leaf segmentation
- **Requirements**: SAM2 thirdparty installation (see above)
- **Architecture**: SAM2.1 Hiera Base+
- **Output**: `sam2_training_run/checkpoints/checkpoint.pt`
- **Features**:
  - Custom dataset loaders for leaf segmentation
  - Video object segmentation (VOS) training approach
  - Configurable training epochs and hyperparameters

#### 2. **`train_yolo_v11.ipynb`** - YOLO v11 Training
- **Description**: Train YOLO v11 instance segmentation model
- **Framework**: [Ultralytics](https://docs.ultralytics.com/)
- **Model**: yolo11m-seg (large variant)
- **Output**: `yolo_training_run/weights/best.pt`
- **Features**:
  - Extensive data augmentation 
  - 300 epochs default training
  - Sanity check visualization included

#### 3. **`train_yolo_v8.ipynb`** - YOLOv8 Training
- **Description**: Train YOLOv8 instance segmentation model
- **Framework**: [Ultralytics](https://docs.ultralytics.com/)
- **Model**: yolov8m-seg (medium variant)
- **Output**: `runs/segment/yolov8/yolo_v8_training_run/weights/best.pt`
- **Features**:
  - Similar augmentation to YOLO v11
  - Early stopping included
  - 300 epochs default training

#### 4. **`train_detectron2.ipynb`** - Detectron2 Training
- **Description**: Train Detectron2 Mask R-CNN for instance segmentation
- **Framework**: [Detectron2](https://github.com/facebookresearch/detectron2)
- **Architecture**: Mask R-CNN R-101 FPN 3x
- **Output**: `leaf/mask_rcnn_R_101_FPN_3x/{timestamp}/model_final.pth`
- **Features**:
  - COCO-format dataset registration
  - Configurable hyperparameters
  - Visualization tools for dataset and predictions

### Training Steps:
1. Prepare your dataset with proper directory structure and annotations
2. Navigate to `seg_train_models/` directory
3. Choose the appropriate training notebook for your model
4. Open and run the notebook, following the instructions
5. Note the output path of your trained model
6. Update the inference notebook with paths to your trained models

## Inference

The `inference_merged.ipynb` notebook provides a unified interface for leaf segmentation and tracking.

### Key Features:

#### Segmentation Models:
- **YOLO Models**: YOLOv8/v11 instance segmentation
- **SAM2**: Segment Anything Model 2 for refinement
- **Detectron2**: Optional Mask R-CNN support

#### Tracking Algorithms:
- **BoxMOT Trackers**: Advanced multi-object tracking
  - ByteTrack
  - DeepOcSort
  - StrongSort

#### Analysis Tools:
- **MOTA/MOTP Metrics**: Tracking performance analysis
- **IOU Calculation**: Frame-based intersection over union
- **Visualization**: Image and mask visualization functions

### Inference Methods:
- **"yolo"**: YOLO model only (faster, good for most cases)
- **"yolo_and_sam2"**: YOLO + SAM2 model (more accurate, more complex)

### Usage:
1. Open `inference_merged.ipynb`
2. Set `PARAMS` with your inference configuration
   - Choose segmentation method: "yolo" or "final"
   - Specify model paths (use your trained models)
   - Configure tracking options
3. Set `DATA_PARAMS` with your data paths
   - Input images directory
   - Output directory for results
4. Run all cells to perform inference and tracking

## Dependencies

### Core (Required):
- ultralytics (YOLO)
- torch
- numpy
- pandas
- imageio
- matplotlib
- scikit-learn
- opencv-python

### Model-Specific:
- **SAM2**: Installed from thirdparty (see installation section)
- **Detectron2**: `pip install detectron2` (optional)

### Tracking (Optional):
- boxmot
- filterpy

## Notes

- All training notebooks preserve existing functionality
- Notebooks include sanity checks for model validation
- Output paths are configurable in each notebook
- For best results with SAM2, ensure you have downloaded the pretrained checkpoints
- The inference notebook supports both trained and pretrained models
- In the `output` folder you can expect the results of trackers combined with models, while `output_v2` contains metrics of YOLO & SAM2 models.
