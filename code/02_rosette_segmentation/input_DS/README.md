# plant_ds1

Plant phenotyping image dataset containing raw RGB images and corresponding segmentation labels for 38 Arabidopsis thaliana accessions.

## Directory Structure

```
plant_ds1/
├── 01_accessions_dataset1_raw/
│   └── <accession>/            # 38 accessions (e.g., Col-0, Ler-1, Ei-6, ...)
│       └── rep_XX/             # Up to 10 replicates per accession (rep_01 – rep_10)
│           └── *.png           # Raw RGB plant images
│
└── 02_accessions_dataset1_labels/
    └── <accession>/            # Same 38 accessions as above
        └── rep_XX/             # Matching replicates
            ├── masks/
            │   └── *_mask.png  # Binary segmentation masks
            └── segmented_images/
                └── *_seg.png   # Segmented plant images
```

## Image Naming Convention

Raw images and labels follow the pattern:

```
<pot_id>_<YYYY>_<MM>_<DD>_<HH>_<MM>_<SS>-<session>-<index>-<camera>_pot_<position>_<accession>-<rep>.png
```

For example: `1000066_2022_05_14_12_00_09-6-5-TA02-RGB1_pot_C4_Ei-6-01.png`

Label files append `_mask` or `_seg` before the extension.

## Accessions

Ba4-1, Ba5-1, Bch-4, Br-0, C24, Can-0, Col-0, Cvi-0, Ede-1, Edr-1, Ei-6, Go-0, Hovdala-2, Hs-0, Hsm, Is-1, Jm-0, Kz-9, Ler-1, Li-5-2, Lz-0, MIB-28, Or-0, Ors-2, PHW-2, Pro-0, Ren-11, Sav-0, TOU-I-17, TOU-J-3, Udul-1-34, Uk-1, Uk-4, Utrecht, Wil-2, Ws-2, Wt-5, Zdr-1
