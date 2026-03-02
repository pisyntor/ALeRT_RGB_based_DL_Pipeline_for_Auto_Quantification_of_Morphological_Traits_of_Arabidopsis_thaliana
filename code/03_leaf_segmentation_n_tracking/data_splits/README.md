# data_splits — Dataset Split Configurations

This directory contains CSV train/val/test split
definitions used by the leaf segmentation and tracking pipeline.


### CSV Split Files

All CSVs contain the same 17,082 images with columns: `plant, rep, image_num,
image_path, mask_path, nn_role`. They differ only in the `nn_role` column
(train/val/test assignment):

| File | Train | Val | Test | Purpose |
|------|-------|-----|------|---------|
| `data_split_v1.csv` | 11,724 | 4,047 | 1,311 | Split optimized for max MOTA |
| `data_split_v2.csv` | 11,688 | 4,519 | 875 | Split optimized for max MOTP |
