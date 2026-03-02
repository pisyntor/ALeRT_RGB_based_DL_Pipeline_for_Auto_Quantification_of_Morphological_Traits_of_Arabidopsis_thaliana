import pathlib

PROJECT_ROOT = pathlib.Path(".").resolve()

# ── Standard library imports ───────────────────────────────────

import json
import logging
import math
import os
import random
import re
import sys
import tempfile
from collections import Counter, namedtuple
from typing import Any, Optional

# ── Third-party imports ────────────────────────────────────────

import cv2
import imageio
import numpy as np
import pandas as pd
import torch
from PIL import Image
from ultralytics import YOLO

# ── SAM2 setup ─────────────────────────────────────────────────

THIRDPARTY_PATH = str((PROJECT_ROOT / "thirdparty").resolve())
SAM2_PATH = str(pathlib.Path(THIRDPARTY_PATH) / "segment-anything-2")
sys.path.append(SAM2_PATH)

# fmt: off
from sam2.build_sam import build_sam2, build_sam2_video_predictor  # noqa: E402
from sam2.sam2_image_predictor import SAM2ImagePredictor  # noqa: E402
# fmt: on

# ── Path configuration ─────────────────────────────────────────

DEFAULT_DATA_PATH = str(PROJECT_ROOT.parent / "leaf_dataset")
DEFAULT_IMAGE_ROOT = str(
    PROJECT_ROOT.parent / "leaf_dataset" / "01_raw_dataset"
)
DEFAULT_MASK_ROOT = str(PROJECT_ROOT.parent / "leaf_dataset" / "02_leaf_labels")
DEFAULT_DS_PATH = str(PROJECT_ROOT / "data" / "ds.csv")


# ── Color constants ────────────────────────────────────────────

DEFAULT_COLORS = [
    (244, 64, 14),
    (48, 57, 249),
    (234, 250, 37),
    (24, 193, 65),
    (245, 130, 49),
    (231, 80, 219),
    (0, 182, 173),
    (115, 0, 218),
    (191, 239, 69),
    (255, 250, 200),
    (250, 190, 212),
    (66, 212, 244),
    (155, 99, 36),
    (220, 190, 255),
    (69, 158, 220),
    (255, 216, 177),
    (98, 2, 37),
    (227, 213, 12),
    (79, 159, 83),
    (170, 23, 101),
    (170, 255, 195),
    (169, 169, 169),
    (181, 111, 119),
    (144, 121, 171),
    (9, 125, 244),
    (184, 70, 30),
    (154, 35, 246),
    (229, 225, 238),
    (141, 254, 82),
    (31, 200, 209),
    (194, 217, 105),
    (91, 20, 124),
    (181, 220, 171),
    (37, 3, 193),
]
OUT_OF_LIST_COLOR = (255, 255, 255)


# ── Mask drawing functions ─────────────────────────────────────


def _draw_masks_on_canvas(canvas: np.ndarray, masks: dict[int, dict]) -> None:
    """Draw masks on the canvas in-place."""
    for i, mask_entry in masks.items():
        color = (
            DEFAULT_COLORS[i] if i < len(DEFAULT_COLORS) else OUT_OF_LIST_COLOR
        )
        canvas[mask_entry["segmentation"]] = color


def draw_joined_masks_on_image(
    image: np.ndarray, masks: dict[int, dict], on_black_background: bool
) -> np.ndarray:
    """Draw masks on the image or on a black background.

    Args:
        image: RGB image array.
        masks: masks in SAM2 format {id: {"segmentation": bool_array}}.
        on_black_background: if True, draw on black; otherwise on image copy.

    Returns:
        Image with masks drawn.
    """
    canvas = np.zeros_like(image) if on_black_background else image.copy()
    _draw_masks_on_canvas(canvas, masks)
    return canvas


def mask_joined_to_masks_dict(mask: np.ndarray) -> dict[int, dict]:
    """Split a joined color mask image into separate boolean masks.

    Args:
        mask: RGB image where each color represents a different mask.

    Returns:
        Dict mapping mask index to SAM2-format mask dict.
    """
    masks = {}

    all_masks_colors = set(
        tuple(x.tolist()) for x in np.unique(mask.reshape(-1, 3), axis=0)
    )
    for color in all_masks_colors:
        if not (
            color in DEFAULT_COLORS
            or color == OUT_OF_LIST_COLOR
            or color == (0, 0, 0)
        ):
            print(
                f"Unexpected color {color}, adding it to the used ones.\n"
                f"New set size: {len(DEFAULT_COLORS) + 1}"
            )
            DEFAULT_COLORS.append(color)

    for i, color in enumerate(DEFAULT_COLORS + [OUT_OF_LIST_COLOR]):
        if color not in all_masks_colors:
            continue

        mask_i = np.all(mask == color, axis=-1)
        if mask_i.sum() > 0:
            masks[i] = {"segmentation": mask_i, "_detection_index": i}
    return masks


def leaf_mask_to_image(
    image: np.ndarray, leaf_mask: np.ndarray, binary: bool
) -> np.ndarray:
    """Extract a single leaf from an image using its mask.

    Args:
        image: RGB image.
        leaf_mask: boolean mask for the leaf.
        binary: if True, return white-on-black; otherwise crop the leaf pixels.

    Returns:
        Image with just the leaf visible.
    """
    if binary:
        color_mask = np.zeros_like(image)
        assert color_mask.dtype == np.uint8
        color_mask[leaf_mask] = 255
    else:
        color_mask = np.zeros_like(image)
        color_mask[leaf_mask] = image[leaf_mask]
    return color_mask


# ── Mask correspondence / mapping ──────────────────────────────


def build_correspondence(
    masks_reference: list[np.ndarray], masks_predicted: list[np.ndarray]
) -> tuple[list[int], list[float]]:
    """Find best-IoU correspondence between reference and predicted masks.

    Args:
        masks_reference: list of reference boolean masks.
        masks_predicted: list of predicted boolean masks.

    Returns:
        Tuple of (indices, ious) where indices[i] is the index in
        masks_predicted best matching masks_reference[i].
    """
    indices = []
    ious = []
    for mask_ref in masks_reference:
        max_iou = -1.0
        max_iou_index = None

        for j, mask_pred in enumerate(masks_predicted):
            intersection = np.logical_and(mask_ref, mask_pred)
            union = np.logical_or(mask_ref, mask_pred)
            iou = np.sum(intersection) / np.sum(union)
            if iou > max_iou:
                max_iou = iou
                max_iou_index = j

        assert max_iou_index is not None, "Error, no predicted_masks found"
        indices.append(max_iou_index)
        ious.append(max_iou)

    if len(set(indices)) != len(indices):
        logging.debug(f"Some masks are not unique, see {indices}")

    assert len(indices) == len(masks_reference)
    return indices, ious


def _greedy_bipartite_match(
    keys_ref: list[int], keys_exp: list[int], similarity: np.ndarray
):
    """Build greedy mapping between two key sets using a similarity matrix.

    Args:
        keys_ref: keys from the reference set.
        keys_exp: keys from the experimental set.
        similarity: IoU matrix of shape [len(keys_ref), len(keys_exp)].

    Returns:
        Dict mapping ref keys to exp keys (best greedy matches).
    """
    assert isinstance(keys_ref, list), "keys_ref should be a list"
    assert isinstance(keys_exp, list), "keys_exp should be a list"
    assert (
        similarity.min(initial=0) > -1
    ), "similarity should be a matrix of nonegative values"

    mapping = {}
    similarity = similarity.copy()
    for _i in range(min(len(keys_ref), len(keys_exp))):
        # pylint: disable=unbalanced-tuple-unpacking
        ref_i, exp_i = np.unravel_index(np.argmax(similarity), similarity.shape)
        mapping[keys_ref[ref_i]] = keys_exp[exp_i]
        similarity[ref_i, :] = -1
        similarity[:, exp_i] = -1

    return mapping


def _compute_iou(mask1: np.ndarray, mask2: np.ndarray) -> float:
    """Calculate IoU between two boolean masks."""
    intersection = np.logical_and(mask1, mask2)
    union = np.logical_or(mask1, mask2)
    return np.sum(intersection) / np.sum(union)


def build_mask_mapping_greedy_lists(
    masks_reference: list[np.ndarray], masks_predicted: list[np.ndarray]
) -> dict[int, tuple[int, float]]:
    """Build greedy IoU-based mapping between two lists of masks.

    Returns:
        Dict mapping reference index -> (predicted index, iou).
    """
    ious_mat = np.zeros((len(masks_reference), len(masks_predicted)))
    for i, mask_ref in enumerate(masks_reference):
        for j, mask_pred in enumerate(masks_predicted):
            ious_mat[i, j] = _compute_iou(mask_ref, mask_pred)

    mapping = _greedy_bipartite_match(
        list(range(len(masks_reference))),
        list(range(len(masks_predicted))),
        ious_mat,
    )

    return {k: (mk, ious_mat[k, mk]) for k, mk in mapping.items()}


def build_mask_mapping_greedy_dicts(
    masks_reference: dict[int, np.ndarray],
    masks_predicted: dict[int, np.ndarray],
) -> dict[int, tuple[int, float]]:
    """Build greedy IoU-based mapping between two dicts of masks.

    Returns:
        Dict mapping reference_id -> (predicted_id, iou).
    """

    def _make_list(masks):
        keys = list(masks.keys())
        return keys, [masks[k] for k in keys]

    keys_ref, masks_ref_list = _make_list(masks_reference)
    keys_exp, masks_pred_list = _make_list(masks_predicted)
    mapping = build_mask_mapping_greedy_lists(masks_ref_list, masks_pred_list)

    return {
        keys_ref[kr]: (keys_exp[ke], iou) for kr, (ke, iou) in mapping.items()
    }


# ── Dataset management ─────────────────────────────────────────


def string_nums_sorting_key(s: str) -> list[str | tuple[int, str]]:
    """Sorting key that handles embedded numbers naturally.

    Splits string on digit boundaries and converts numeric parts
    to (int, str) tuples for correct numeric ordering.
    """
    parts = re.split(r"(\d+)", s)
    return [(int(part), part) if part.isdigit() else part for part in parts]


def save_dataset(
    dataset: pd.DataFrame,
    path: str,
    images_root: str,
    masks_root: Optional[str],
) -> None:
    """Save dataset to CSV with paths relative to their roots."""
    dataset = dataset.copy()
    assert not pathlib.Path(path).exists(), f"File already exists: {path}"

    def _to_relative_image(p: str) -> str:
        return os.path.relpath(p, images_root)

    def _to_relative_mask(p: str) -> str:
        if masks_root is None:
            assert len(p) == 0, "Mask path is not empty, but masks_root is None"
        return "" if len(p) == 0 else os.path.relpath(p, masks_root)

    dataset["image_path"] = dataset["image_path"].apply(_to_relative_image)
    dataset["mask_path"] = dataset["mask_path"].apply(_to_relative_mask)
    dataset.to_csv(path, index=False)


def load_dataset(
    path: Optional[str] = None,
    images_root: str = DEFAULT_IMAGE_ROOT,
    masks_root: Optional[str] = DEFAULT_MASK_ROOT,
) -> pd.DataFrame:
    """Load dataset from CSV and resolve relative paths to absolute.

    Args:
        path: path to CSV file; uses DEFAULT_DS_PATH if None.
        images_root: root directory for images.
        masks_root: root directory for masks (None if unavailable).
    """
    if path is None:
        path = DEFAULT_DS_PATH

    dataset = pd.read_csv(path)

    def _to_global_image(p: str) -> str:
        return str((pathlib.Path(images_root) / p).resolve())

    def _to_global_mask(p: str) -> str:
        if len(p) == 0:
            assert (
                masks_root is None
            ), "Mask path is empty, but masks_root is not None"
            return ""
        return (
            ""
            if masks_root is None
            else str((pathlib.Path(masks_root) / p).resolve())
        )

    dataset["image_path"] = dataset["image_path"].apply(_to_global_image)
    dataset["mask_path"] = dataset["mask_path"].apply(_to_global_mask)
    return dataset


def load_or_build_dataset(
    path: Optional[str] = "default",
    images_root: str = DEFAULT_IMAGE_ROOT,
    masks_root: Optional[str] = DEFAULT_MASK_ROOT,
) -> pd.DataFrame:
    """Load dataset from CSV, or build one from directory structure if no CSV.

    Args:
        path: CSV path, "default" for DEFAULT_DS_PATH, or None to build from dirs.
        images_root: root dir with images.
        masks_root: root dir with masks (None if unavailable).
    """
    if path == "default":
        path = DEFAULT_DS_PATH

    if path is not None:
        dataset = load_dataset(path, images_root=images_root, masks_root=masks_root)
    else:
        logging.warning(
            "No path is provided, building the dataset with "
            f"images in {images_root} and masks in {masks_root}."
        )
        dataset = _parse_dataset_files(
            images_root=images_root, masks_root=masks_root
        )
        logging.warning(f"Loaded ds with {len(dataset)} test images.")
        dataset["nn_role"] = "test"

    return dataset


class ProjectPaths:
    """Handles directory traversal for the plant image dataset structure."""

    def __init__(
        self,
        images_root: str = DEFAULT_IMAGE_ROOT,
        masks_root: Optional[str] = DEFAULT_MASK_ROOT,
    ):
        self.images_root = pathlib.Path(images_root)
        self.masks_root = (
            None if masks_root is None else pathlib.Path(masks_root)
        )

    def get_plants(self) -> list[str]:
        """Get sorted list of plant directory names."""
        return sorted(
            (
                p.name
                for p in self.images_root.glob("*")
                if not p.name.startswith(".")
            ),
            key=string_nums_sorting_key,
        )

    def get_reps(self, plant_name: str) -> list[str]:
        """Get sorted list of repetition directory names for a plant."""
        return sorted(
            (p.name for p in self.images_root.glob(f"{plant_name}/*")),
            key=string_nums_sorting_key,
        )

    def get_images(self, plant_name: str, rep_name: str) -> list[str]:
        """Get sorted list of image filenames for a plant/rep combo."""
        return sorted(
            (
                p.name
                for p in self.images_root.glob(f"{plant_name}/{rep_name}/*.png")
            ),
            key=string_nums_sorting_key,
        )

    def get_image_masks(
        self, plant_name: str, rep_name: str
    ) -> list[tuple[str, str]]:
        """Get sorted list of (image_path, mask_path) tuples.

        Mask paths are "" when masks_root is None.
        """

        def _get_image_path(image):
            return str(self.images_root / plant_name / rep_name / image)

        def _get_mask_path(image):
            return (
                str(self.masks_root / plant_name / rep_name / image)
                if self.masks_root
                else ""
            )

        return sorted(
            (
                (
                    _get_image_path(image),
                    _get_mask_path(image),
                )
                for image in self.get_images(plant_name, rep_name)
            ),
            key=lambda x: string_nums_sorting_key(x[0]),
        )


class OutputProjectPaths:
    """Build paths for output project files.

    out_dir
    - metrics.csv
    - params.json
    - <nn_role>
    -- <plant>
    --- <rep>
    ---- colour_leaf_masks/   (masks on black)
    ---- colour_leaf_masks_raw/  (masks on image)
    ---- leaf_num_masks/leaf_N/  (binary per-leaf)
    ---- leaf_seg_imgs/leaf_N/   (extracted per-leaf)
    """

    def __init__(self, output_directory: str, separate_nn_role: bool = True):
        self.data_path = pathlib.Path(output_directory)
        self.separate_nn_role = separate_nn_role

    def make_params_path(self) -> str:
        """Path for the params.json file."""
        return str(self.data_path / "params.json")

    def _rep_path(self, dataset_row: pd.Series) -> pathlib.Path:
        if self.separate_nn_role:
            role_path = self.data_path / dataset_row["nn_role"]
        else:
            role_path = self.data_path
        return role_path / dataset_row["plant"] / dataset_row["rep"]

    def _basename(self, dataset_row: pd.Series) -> str:
        return pathlib.Path(dataset_row["image_path"]).name

    def make_joined_mask_path(
        self, dataset_row: pd.Series, on_black_background: bool
    ) -> str:
        """Path for a joined color mask image.

        Args:
            dataset_row: row from the dataset DataFrame.
            on_black_background: True for masks on black, False for on image.
        """
        subdir = (
            "colour_leaf_masks"
            if on_black_background
            else "colour_leaf_masks_raw"
        )
        return str(
            self._rep_path(dataset_row)
            / subdir
            / f"{self._basename(dataset_row)}"
        )

    def make_leaf_i_mask_path(
        self, dataset_row: pd.Series, i: int, binary: bool
    ) -> str:
        """Path for a single leaf mask image."""
        subdir = "leaf_num_masks" if binary else "leaf_seg_imgs"
        return str(
            self._rep_path(dataset_row)
            / subdir
            / f"leaf_{i+1}"
            / f"{self._basename(dataset_row)}"
        )

    def make_metrics_path(self, per_sequence: bool) -> str:
        """Path for the metrics CSV file."""
        name = "metrics_per_sequence" if per_sequence else "metrics"
        return str(self.data_path / f"{name}.csv")

    def make_config_path(self) -> str:
        """Path for the config.json file."""
        return str(self.data_path / "config.json")


def _parse_dataset_files(
    images_root: str = DEFAULT_IMAGE_ROOT,
    masks_root: Optional[str] = DEFAULT_MASK_ROOT,
) -> pd.DataFrame:
    """Parse directory structure to build a dataset DataFrame."""
    dataset_items = []
    paths = ProjectPaths(images_root=images_root, masks_root=masks_root)

    plants = paths.get_plants()
    for plant in plants:
        reps = paths.get_reps(plant)
        for rep in reps:
            for image_number, (image_path, mask_path) in enumerate(
                paths.get_image_masks(plant, rep)
            ):
                assert pathlib.Path(
                    image_path
                ).exists(), f"Image not found: {image_path}"
                assert (
                    len(mask_path) == 0 or pathlib.Path(mask_path).exists()
                ), f"Mask not found: {mask_path}"
                dataset_items.append(
                    {
                        "plant": plant,
                        "rep": rep,
                        "image_num": image_number,
                        "image_path": image_path,
                        "mask_path": mask_path,
                        "nn_role": None,
                    }
                )

    return pd.DataFrame(dataset_items)


def _assign_roles(
    objects: list[object],
    random_generator: random.Random,
    minimal_test_share: float,
    minimal_val_share: float,
) -> dict[object, str]:
    """Assign train/val/test roles to objects based on share ratios."""
    objects = objects.copy()
    random_generator.shuffle(objects)

    num_test = math.ceil(minimal_test_share * len(objects))
    num_val = math.ceil(minimal_val_share * len(objects))
    assert num_test + num_val < len(objects), "Not enough objects for train"

    test_objects = objects[:num_test]
    val_objects = objects[num_test : num_test + num_val]
    train_objects = objects[num_test + num_val :]

    objects_roles = {}
    for objects_group, group_role in [
        (test_objects, "test"),
        (val_objects, "val"),
        (train_objects, "train"),
    ]:
        for r in objects_group:
            objects_roles[r] = group_role

    return objects_roles


def _split_dataset_rep_based(
    dataset: pd.DataFrame,
    minimal_test_share: float,
    minimal_val_share: float,
    seed: int = 1,
) -> pd.DataFrame:
    """Split dataset into train/val/test based on the rep column."""
    assert minimal_test_share > 0, "Test share should be positive"

    random_generator = random.Random(seed)
    dataset = dataset.copy()
    assert (
        dataset["nn_role"].isna().all()
    ), "Dataset should not have nn_role assigned yet"

    plants = dataset["plant"].unique()
    for plant in plants:
        reps = list(dataset[dataset["plant"] == plant]["rep"].unique())
        rep_roles = _assign_roles(
            reps, random_generator, minimal_test_share, minimal_val_share
        )
        for rep, role in rep_roles.items():
            dataset.loc[
                (dataset["plant"] == plant) & (dataset["rep"] == rep), "nn_role"
            ] = role

    return dataset


def build_dataset(
    images_root: str = DEFAULT_IMAGE_ROOT,
    masks_root: str = DEFAULT_MASK_ROOT,
    minimal_test_share: float = 0.1,
    minimal_val_share: float = 0.2,
    seed: int = 1,
) -> pd.DataFrame:
    """Build dataset from the project directory structure.

    Args:
        images_root: path to image directory.
        masks_root: path to mask directory (None if unavailable).
        minimal_test_share: minimum share of test sequences.
        minimal_val_share: minimum share of validation sequences.
        seed: random seed for splitting.
    """
    dataset = _parse_dataset_files(images_root=images_root, masks_root=masks_root)
    assert len(dataset) > 0, (
        f"No images found in the dataset with IMG_ROOT={images_root}. "
        "Possible reasons: incorrect directory; "
        "not following the structure 'IMG_ROOT/<group>/<rep>/<image.png>'."
    )
    dataset = _split_dataset_rep_based(
        dataset, minimal_test_share, minimal_val_share, seed
    )
    return dataset


# ── Model interface ────────────────────────────────────────────


class AbstractModel:
    """Abstract model class, setting the interface for model classes."""

    def __init__(self, config: dict, device: str):
        self.config = config
        self.device = device

    def predict_masks(self, images: list) -> list[dict[int, dict]]:
        """Predict masks for a sequence of images.

        Args:
            images: sequence of RGB images.

        Returns:
            List of mask dicts, one per image. Each dict maps
            mask_id -> {"segmentation": bool_array}.
        """
        raise NotImplementedError()

    def get_config(self):
        """Return the model configuration dict."""
        return self.config

    @staticmethod
    def _renumerate_masks(
        masks: list[dict[int, dict]]
    ) -> list[dict[int, dict]]:
        """Assign new sequential IDs to masks, ordered by area (largest first)."""
        start_id = 0
        result_masks = []
        old_ids_to_new_ids = {}

        for current_masks in masks:
            not_added_ids_to_size: dict[int, int] = {}
            for old_id, mask_entry in current_masks.items():
                if old_id not in old_ids_to_new_ids:
                    not_added_ids_to_size[old_id] = mask_entry[
                        "segmentation"
                    ].sum()

            for old_id, _ in sorted(
                not_added_ids_to_size.items(), key=lambda x: x[1], reverse=True
            ):
                old_ids_to_new_ids[old_id] = start_id
                start_id += 1

            new_masks = {}
            for old_id, mask_entry in current_masks.items():
                new_masks[old_ids_to_new_ids[old_id]] = mask_entry
            result_masks.append(new_masks)

        return result_masks

    @staticmethod
    def _filter_edge_areas(
        mask: np.ndarray,
        edge_share: float = 0.125,
        area_share_threshold: float = 0.5,
    ) -> bool:
        """Return False if a mask's area is mostly on the image edge."""
        h, w = mask.shape
        h_small = int(h * edge_share)
        h_large = int(h * (1 - edge_share))
        w_small = int(w * edge_share)
        w_large = int(w * (1 - edge_share))

        mask_area = mask.sum()
        for region in [
            mask[:h_small, :],
            mask[h_large:, :],
            mask[:, :w_small],
            mask[:, w_large:],
        ]:
            if region.sum() / mask_area > area_share_threshold:
                return False
        return True


def log_debug_warning(debug_mode: bool) -> None:
    """Log warning if debug mode is on."""
    if debug_mode:
        logging.error("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        logging.error("!!!DEBUG MODE IS ON, MASKS ARE INCORRECT!!!")
        logging.error("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        logging.error("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")


def set_device(device: str) -> torch.device:
    """Set the device for the model."""
    torch_device = torch.device(device)
    if torch_device.type == "cuda:1":
        # use bfloat16 for the entire notebook
        torch.autocast("cuda:1", dtype=torch.bfloat16).__enter__()
        # turn on tfloat32 for Ampere GPUs
        if torch.cuda.get_device_properties(0).major >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

    return torch_device


# ── Metrics ────────────────────────────────────────────────────


def _sorted_dict_values(objects: dict) -> list:
    """Return dict values sorted by their keys."""
    return [objects[key] for key in sorted(objects.keys())]


def _extract_mask_arrays(masks: dict[int, dict]) -> list[np.ndarray]:
    """Extract segmentation arrays from SAM2-format mask dicts, sorted by key."""
    return [m["segmentation"] for m in _sorted_dict_values(masks)]


class AbstractMetric:
    """Abstract class for metrics."""

    def __init__(self, *args: Any, **kwargs: Any):
        self.args = args
        self.kwargs = kwargs

    def add_sequence(
        self,
        masks_reference: list[dict[int, dict]],
        masks_predicted: list[dict[int, dict]],
        name: str = "",
    ):
        """Update metric with a new sequence of masks."""
        raise NotImplementedError

    def get_aggregate_metrics(
        self, per_sequence: bool = False
    ) -> pd.DataFrame:
        """Get aggregate metrics, optionally broken down per sequence."""
        raise NotImplementedError

    def get_name(self):
        """Get the name of the metric."""
        return self.__class__.__name__

    def reset(self) -> None:
        """Reset the metric for a new dataset."""
        raise NotImplementedError()


class SimpleMetric(AbstractMetric):
    """Base metric that aggregates per-sequence statistics.

    Subclasses implement _calc_sequence_statistics and _calc_aggregate_metrics.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.seq_stats = {}

    def _calc_sequence_statistics(
        self,
        masks_reference: list[dict[int, dict]],
        masks_predicted: list[dict[int, dict]],
        name: str = "",
    ) -> Any:
        """Calculate metric data for a single sequence."""
        raise NotImplementedError()

    def _calc_aggregate_metrics(self, results_list: list) -> dict:
        """Aggregate per-sequence data into final metric values."""
        raise NotImplementedError()

    def add_sequence(
        self,
        masks_reference: list[dict[int, dict]],
        masks_predicted: list[dict[int, dict]],
        name: str = "",
    ):
        assert (
            self.seq_stats.get(name) is None
        ), "Sequence {name} already added, please specify another one"
        self.seq_stats[name] = self._calc_sequence_statistics(
            masks_reference, masks_predicted, name
        )

    def get_aggregate_metrics(
        self, per_sequence: bool = False
    ) -> pd.DataFrame:
        if not per_sequence:
            all_values = list(self.seq_stats.values())
            res = pd.DataFrame([self._calc_aggregate_metrics(all_values)])
        else:
            metrics_by_seq = []
            for name, value in self.seq_stats.items():
                metrics = self._calc_aggregate_metrics([value])
                metrics["rep"] = name
                metrics_by_seq.append(metrics)

            res = pd.DataFrame(metrics_by_seq).set_index("rep")

        return res

    def reset(self):
        self.seq_stats = {}


class FrameBasedIOU(SimpleMetric):
    """Frame-based Intersection over Union metric.

    1. Find IoU-best correspondence between reference and predicted masks.
    2. Calculate share of correctly corresponded pixels.
    """

    @staticmethod
    def _calc_one_frame(
        masks_reference_dicts: dict[int, dict],
        masks_predicted_dicts: dict[int, dict],
    ) -> float:
        masks_reference = _extract_mask_arrays(masks_reference_dicts)
        masks_predicted = _extract_mask_arrays(masks_predicted_dicts)

        if len(masks_reference) == 0 or len(masks_predicted) == 0:
            return 0

        indices, _ = build_correspondence(masks_reference, masks_predicted)

        corresponded_pixels = 0
        for mask_ref, index in zip(masks_reference, indices):
            assert index is not None, "Error, no masks found"
            corresponded_pixels += np.logical_and(
                mask_ref, masks_predicted[index]
            ).sum()

        all_pixels_mask = np.zeros_like(masks_reference[0])
        for mask in list(masks_reference) + list(masks_predicted):
            all_pixels_mask = np.logical_or(all_pixels_mask, mask)

        return corresponded_pixels / all_pixels_mask.sum()

    def _calc_sequence_statistics(
        self, masks_reference: list, masks_predicted: list, name: str = ""
    ) -> list[float]:
        values = []
        for masks_ref, masks_pred in zip(masks_reference, masks_predicted):
            values.append(self._calc_one_frame(masks_ref, masks_pred))
        return values

    def _calc_aggregate_metrics(self, results_list: list[list[float]]) -> dict:
        flatten_list = [x for sublist in results_list for x in sublist]
        if len(flatten_list) == 0:
            return {"FrameBasedIOU": np.nan}
        return {"FrameBasedIOU": np.mean(flatten_list)}


class MultiObjectTrackingPrecision(SimpleMetric):
    """Multi-object tracking precision metric (average IoU of matched objects)."""

    def __init__(self, overlap_threshold: float = 0.5):
        super().__init__(overlap_threshold=overlap_threshold)
        self.overlap_threshold = overlap_threshold

    def _calc_one_frame(
        self,
        masks_reference_dicts: dict[int, dict],
        masks_predicted_dicts: dict[int, dict],
    ) -> list[float]:
        masks_reference = _extract_mask_arrays(masks_reference_dicts)
        masks_predicted = _extract_mask_arrays(masks_predicted_dicts)

        if len(masks_reference) == 0 or len(masks_predicted) == 0:
            return []

        _indices, ious = build_correspondence(masks_reference, masks_predicted)
        return [x for x in ious if x > self.overlap_threshold]

    def _calc_sequence_statistics(
        self, masks_reference: list, masks_predicted: list, name: str = ""
    ) -> list[list[float]]:
        frames_ious = []
        for masks_ref, masks_pred in zip(masks_reference, masks_predicted):
            frames_ious.append(self._calc_one_frame(masks_ref, masks_pred))
        return frames_ious

    def _calc_aggregate_metrics(self, results_list) -> dict:
        expanded = []
        for sequence_ious in results_list:
            for frame_ious in sequence_ious:
                for iou_value in frame_ious:
                    expanded.append(iou_value)
        return {"MultiObjectTrackingPrecision": np.mean(expanded)}


MOTAAccumulator = namedtuple(
    "MOTAAccumulator",
    ["false_positives", "false_negatives", "id_switches", "ground_truth_count"],
)


class MultiObjectTrackingAccuracy(SimpleMetric):
    """Multi-object tracking accuracy metric (MOTA)."""

    def __init__(
        self, compatible_mode: bool = False, overlap_threshold: float = 0.5
    ):
        super().__init__(overlap_threshold=overlap_threshold)
        self.overlap_threshold = overlap_threshold
        self.compatible_mode = compatible_mode

    def _calc_sequence_statistics(
        self, masks_reference: list, masks_predicted: list, name: str = ""
    ) -> list[MOTAAccumulator]:
        accumulators = []
        previous_id_map: dict[int, int] = {}
        for masks_ref, masks_pred in zip(masks_reference, masks_predicted):
            accumulator, previous_id_map = self._calc_one_frame(
                masks_ref, masks_pred, previous_id_map
            )
            accumulators.append(accumulator)
        return accumulators

    def _calc_aggregate_metrics(
        self, results_list: list[list[MOTAAccumulator]]
    ) -> dict:
        flat_list = []
        for sequence_accs in results_list:
            for acc in sequence_accs:
                flat_list.append(acc)

        if len(flat_list) == 0:
            sum_stat = MOTAAccumulator(np.nan, np.nan, np.nan, np.nan)
        else:
            sum_stat = MOTAAccumulator(*np.array(flat_list).sum(axis=0))

        metrics = {
            "FalsePositives": sum_stat.false_positives,
            "FalseNegatives": sum_stat.false_negatives,
            "IDSwitches": sum_stat.id_switches,
            "GroundTruthMasksCount": sum_stat.ground_truth_count,
            "MultiObjectTrackingAccuracy": 1
            - (
                sum_stat.false_positives
                + sum_stat.false_negatives
                + sum_stat.id_switches
            )
            / sum_stat.ground_truth_count,
        }

        if self.compatible_mode:
            metrics = {k + "-deprecated": v for k, v in metrics.items()}

        return metrics

    @staticmethod
    def _calc_id_switches(previous_id_map: dict, id_map: dict):
        """Count ID switches between two consecutive frames' ID maps."""
        id_switches = 0
        for k in set(previous_id_map.keys()) & set(id_map.keys()):
            i1 = previous_id_map[k]
            i2 = id_map[k]
            if i1 is not None and i2 is not None and i1 != i2:
                id_switches += 1
        return id_switches

    def _calc_one_frame(
        self,
        masks_reference_dicts: dict[int, dict],
        masks_predicted_dicts: dict[int, dict],
        previous_id_map: dict[int, int],
    ) -> tuple[MOTAAccumulator, dict[int, int]]:
        # TODO: more accurate id switches calculation

        def _to_list_dicts(masks_dict, compat_mode):
            keys = sorted(masks_dict.keys())
            res = []
            for i, k in enumerate(keys):
                res.append(
                    {
                        "segmentation": masks_dict[k]["segmentation"],
                        "key": i if compat_mode else k,
                    }
                )
            return res

        masks_reference = _to_list_dicts(
            masks_reference_dicts, compat_mode=self.compatible_mode
        )
        masks_predicted = _to_list_dicts(
            masks_predicted_dicts, compat_mode=self.compatible_mode
        )

        if len(masks_reference) == 0 or len(masks_predicted) == 0:
            return MOTAAccumulator(0, 0, 0, 0), {}

        inds_unfiltered, ious = build_correspondence(
            masks_reference=[m["segmentation"] for m in masks_reference],
            masks_predicted=[m["segmentation"] for m in masks_predicted],
        )

        id_map = {}
        correct_correspondences = 0
        for i, (ind, iou) in enumerate(zip(inds_unfiltered, ious)):
            if iou > self.overlap_threshold:
                new_ind = ind
                correct_correspondences += 1
            else:
                new_ind = None

            id_map[masks_reference[i]["key"]] = (
                masks_predicted[new_ind]["key"] if new_ind is not None else None
            )

        accumulator = MOTAAccumulator(
            false_positives=len(masks_predicted) - correct_correspondences,
            false_negatives=len(masks_reference) - correct_correspondences,
            id_switches=self._calc_id_switches(previous_id_map, id_map),
            ground_truth_count=len(masks_reference),
        )
        return accumulator, id_map

    def get_name(self):
        if self.compatible_mode:
            return "MOTA-deprecated"
        return self.__class__.__name__


# ── Resolution utilities ───────────────────────────────────────


def _build_sam_format_mask_mapping(
    masks_reference: dict[int, dict],
    masks_predicted: dict[int, dict],
) -> dict[int, tuple[int, float]]:
    """Calculate correspondence between masks given in SAM format."""
    ref_masks_np = {k: v["segmentation"] for k, v in masks_reference.items()}
    pred_masks_np = {k: v["segmentation"] for k, v in masks_predicted.items()}
    return build_mask_mapping_greedy_dicts(ref_masks_np, pred_masks_np)


def change_mask_resolution(
    mask: np.ndarray, new_size: tuple[int, int]
) -> np.ndarray:
    """Resize a boolean mask using nearest-neighbor interpolation.

    Args:
        mask: 2D boolean/uint8 mask.
        new_size: target (height, width).
    """
    assert len(new_size) == 2, "New size should be 2D"
    assert len(mask.shape) == 2, "Mask should be 2D"
    new_size_wh = new_size[::-1]  # PIL uses (width, height)
    return np.array(
        Image.fromarray(mask).resize(
            new_size_wh, resample=Image.Resampling.NEAREST
        )
    )


def _resize_image(
    image: np.ndarray, new_size: tuple[int, int]
) -> np.ndarray:
    """Resize an RGB image using bicubic interpolation.

    Args:
        image: 3D image array (H, W, C).
        new_size: target (height, width).
    """
    assert len(new_size) == 2, "New size should be 2D"
    assert len(image.shape) == 3, "Image should be 3D"
    new_size_wh = new_size[::-1]  # PIL uses (width, height)
    return np.array(
        Image.fromarray(image).resize(
            new_size_wh, resample=Image.Resampling.BICUBIC
        )
    )


def ensure_same_image_sizes(
    images: list[np.ndarray], description: str
) -> list[np.ndarray]:
    """Ensure all images have the same size, resizing to the most common if not.

    Args:
        images: list of images.
        description: label for warning messages.
    """
    sizes = Counter([img.shape for img in images])
    if len(sizes) <= 1:
        return images

    print(f"Warning! Sizes of images {description} are different.")
    print(
        f"    There are {len(sizes)} different sizes: {sizes.most_common(3)}..."
    )
    print(
        "    Resizing images to the most common size: "
        f"{sizes.most_common(1)[0][0]}"
    )

    new_size = sizes.most_common(1)[0][0][:2]
    return [_resize_image(img, new_size) for img in images]


# ── Data I/O (Save/Load) ──────────────────────────────────────


def _save_image(path: str, image: np.ndarray) -> None:
    imageio.imwrite(path, image)


def _imread_func(path: str) -> np.ndarray:
    return imageio.imread(path)


def read_image(dataset_row: pd.Series, key="image_path") -> np.ndarray:
    """Read an image from a dataset row."""
    return _imread_func(dataset_row[key])


def read_masks(dataset_row: pd.Series, key="mask_path") -> list[dict]:
    """Read and parse masks from a dataset row."""
    joined_masks = _imread_func(dataset_row[key])
    return mask_joined_to_masks_dict(joined_masks)


def _ensure_parent_dirs(path):
    """Create parent directories for a file path if they don't exist."""
    pathlib.Path(path).parent.mkdir(parents=True, exist_ok=True)


class Saver:
    """Save output masks in specific formats.

    Saves: joined color masks (on black + on image), per-leaf masks.
    """

    def __init__(self, output_directory: str, save_leaves: bool = True):
        self.output_directory = output_directory
        self.save_leaves = save_leaves
        self.saved_leaf_indices: list[int] = []
        self.out_paths = OutputProjectPaths(output_directory)

    def save_masks(
        self,
        image: np.ndarray,
        dataset_row: pd.Series,
        out_masks: dict[int, dict],
    ) -> None:
        """Save output masks for one image."""
        for on_black_background in [True, False]:
            image_with_masks = draw_joined_masks_on_image(
                image, out_masks, on_black_background=on_black_background
            )
            joined_masks_path = self.out_paths.make_joined_mask_path(
                dataset_row, on_black_background=on_black_background
            )
            _ensure_parent_dirs(joined_masks_path)
            imageio.imwrite(joined_masks_path, image_with_masks)

        if not self.save_leaves:
            return

        for ind, mask in out_masks.items():
            self.saved_leaf_indices.append(ind)
            for binary in [True, False]:
                leaf_image = leaf_mask_to_image(
                    image, mask["segmentation"], binary
                )
                leaf_path = self.out_paths.make_leaf_i_mask_path(
                    dataset_row, ind, binary=binary
                )
                _ensure_parent_dirs(leaf_path)
                imageio.imwrite(leaf_path, leaf_image)

    def finalize_sequence(self, row: pd.Series):
        """Finalize a sequence (placeholder for future video saving)."""
        self.saved_leaf_indices = []

    def save_metrics(
        self, metrics: pd.DataFrame, per_sequence: bool = False
    ):
        """Save metrics DataFrame to CSV."""
        metrics_path = self.out_paths.make_metrics_path(per_sequence)
        _ensure_parent_dirs(metrics_path)
        metrics.to_csv(metrics_path, index=True)

    def save_configs(self, configs: dict):
        """Save model config to JSON."""
        configs_path = self.out_paths.make_config_path()
        _ensure_parent_dirs(configs_path)
        with open(configs_path, "w", encoding="utf-8") as f:
            json.dump(configs, f)


class EmptySaver(Saver):
    """No-op saver for when saving is disabled."""

    def save_masks(self, image, dataset_row, out_masks):
        pass

    def finalize_sequence(self, row):
        pass

    def save_metrics(self, metrics, per_sequence=False):
        pass

    def save_configs(self, configs):
        pass


# ── YOLO tracker model ─────────────────────────────────────────


class YoloTrackerModel(AbstractModel):
    """Model using YOLO's built-in tracker for detection and tracking."""

    CONFIG_KEYS = {
        "yolo_model",
        "yolo_threshold",
        "debug_mode",
    }

    def __init__(self, config: dict, device: str):
        super().__init__(config, device)
        self.yolo_model = YOLO(config["yolo_model"]).to(self.device)
        self.yolo_model.track(
            (np.random.random((416, 416, 3)) * 255).astype("uint8"),
            persist=True,
        )  # init tracker

        assert (
            set(self.config.keys()) == self.CONFIG_KEYS
        ), f"Expected keys: {self.CONFIG_KEYS}, got {self.config.keys()}"

    def predict_masks(self, images: list) -> list[dict[int, dict]]:
        """Predict masks for a sequence of images using YOLO tracking."""
        masks_log: list[dict[int, dict]] = []

        if self.yolo_model.predictor is not None:
            self.yolo_model.predictor.trackers[0].reset()

        for _i, image in enumerate(images):
            # Duplicate tracks slightly improve results on the dataset.
            yolo_res = self.yolo_model.track(
                image[:, :, ::-1],
                persist=True,
                verbose=False,
                retina_masks=True,
            )
            yolo_res = self.yolo_model.track(
                image[:, :, ::-1],
                persist=True,
                verbose=False,
                retina_masks=True,
            )

            if yolo_res[0].masks is None:
                masks_log.append({})
                continue

            over_threshold = (
                (yolo_res[0].boxes.conf > self.config["yolo_threshold"])
                .cpu()
                .detach()
                .numpy()
            )
            masks = {}

            if yolo_res[0].masks.xy is None or yolo_res[0].boxes.id is None:
                masks_log.append({})
                continue

            for contour, over, mask_id, xywh in zip(
                yolo_res[0].masks.xy,
                over_threshold,
                yolo_res[0].boxes.id,
                yolo_res[0].boxes.xywh,
            ):
                if not over:
                    continue

                binary_mask = np.zeros(image.shape[:2], np.uint8)
                contour = contour.astype(np.int32)
                contour = contour.reshape(-1, 1, 2)
                _ = cv2.drawContours(
                    binary_mask, [contour], -1, (255, 255, 255), cv2.FILLED
                )
                masks[int(mask_id)] = {
                    "segmentation": binary_mask > 0,
                    "xywh": xywh.cpu().detach().numpy(),
                }

            masks_log.append(masks)

        self.yolo_model.predictor.trackers[0].reset()
        masks_log = self._renumerate_masks(masks_log)
        return masks_log


# ── Final SAM model (VideoSAMFinal) ───────────────────────────


class VideoSAMFinal(AbstractModel):
    """Final model: YOLO tracking + SAM2 refinement."""

    # Class-level parameters to avoid too complicated config
    ensure_same_image_sizes_flag = True
    remove_edge_detections = True
    simple_sam2_mode = True
    iou_refinement_threshold = 0.55

    edge_threshold = 0.0625
    area_share_threshold = 0.25

    morphology_joining_iou_threshold = 0.7
    morphology_joining_kernel_size = 9

    remove_stems_parts = False
    remove_stems_parts_kernel_size = 3

    EXPECTED_INPUT_KEYS = {
        "yolo_model",
        "yolo_threshold",
        "sam2_cfg",
        "sam2_model",
        "debug_mode",
        "morphology_join_stems",
    }

    def __init__(self, config: dict, device: str):
        config = config.copy()

        input_keys = set(config.keys())
        assert (
            input_keys == self.EXPECTED_INPUT_KEYS
        ), f"Incorrect config, diff is {input_keys ^ self.EXPECTED_INPUT_KEYS}"

        config["ensure_same_image_sizes"] = self.ensure_same_image_sizes_flag
        config["remove_edge_detections"] = self.remove_edge_detections
        config["simple_sam2_mode"] = self.simple_sam2_mode
        config["iou_refine_thresh"] = self.iou_refinement_threshold
        config["edge_threshold"] = self.edge_threshold
        config["area_share_threshold"] = self.area_share_threshold
        config["morphology_joining_iou_thresh"] = (
            self.morphology_joining_iou_threshold
        )
        config["morphology_joining_kernel_size"] = (
            self.morphology_joining_kernel_size
        )
        config["remove_stems_parts"] = self.remove_stems_parts
        config["remove_stems_parts_kernel_size"] = (
            self.remove_stems_parts_kernel_size
        )

        assert (not config["remove_stems_parts"]) or (
            not config["morphology_join_stems"]
        ), (
            "Mostly one of remove_stems_parts and morphology_join_stems "
            "should be set to True"
        )
        super().__init__(config, device)

        self.yolo_tracker_model = YoloTrackerModel(
            {
                "yolo_model": config["yolo_model"],
                "yolo_threshold": config["yolo_threshold"],
                "debug_mode": config["debug_mode"],
            },
            device,
        )

        if self.config["simple_sam2_mode"]:
            self.sam2_model = SAM2ImagePredictor(
                build_sam2(
                    self.config["sam2_cfg"],
                    self.config["sam2_model"],
                    device=set_device(self.device),
                    apply_postprocessing=False,
                )
            )
        else:
            self.sam2_model = build_sam2_video_predictor(
                self.config["sam2_cfg"],
                self.config["sam2_model"],
                device=set_device(self.device),
                apply_postprocessing=False,
            )

        log_debug_warning(config["debug_mode"])

    def _remove_stems_morphology(self, mask):
        """Remove thin stem-like parts from a mask using morphological ops."""
        kernel_size = self.config["remove_stems_parts_kernel_size"]
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)
        )
        mask_erode = cv2.erode(mask.astype(np.uint8), kernel, iterations=1)
        mask_erode = cv2.dilate(mask_erode, kernel, iterations=2)
        mask = mask & (mask_erode == 1)
        return mask

    def _join_masks_morphology_stems(
        self,
        yolo_mask: np.ndarray,
        sam_mask: np.ndarray,
    ) -> np.ndarray:
        """Join SAM and YOLO masks, preserving stems from YOLO."""
        assert yolo_mask.shape == sam_mask.shape
        assert yolo_mask.dtype == bool
        assert sam_mask.dtype == bool

        kernel_size = self.config["morphology_joining_kernel_size"]
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)
        )

        no_stems_erode = cv2.erode(
            yolo_mask.astype(np.uint8), kernel, iterations=1
        )
        no_stems_erode = cv2.dilate(no_stems_erode, kernel, iterations=2)

        stems_only = yolo_mask & (no_stems_erode == 0) & (~sam_mask)
        joined = sam_mask | stems_only

        stems_joined = joined.astype(np.uint8)
        stems_joined = cv2.dilate(stems_joined, kernel, iterations=3)
        stems_joined = cv2.erode(stems_joined, kernel, iterations=3)

        stems_only_erode = cv2.dilate(
            stems_only.astype(np.uint8), kernel, iterations=3
        )

        stems_final = (stems_only_erode == 1) & (stems_joined == 1) & yolo_mask
        return sam_mask | stems_final

    def _sam_predict_simple(self, image, masks):
        """Refine masks using SAM2 image predictor with bounding boxes."""
        if len(masks) == 0:
            return masks

        self.sam2_model.set_image(image)
        keys = []
        boxes = []
        for ann_obj_id, mask_entry in masks.items():
            mask = mask_entry["segmentation"]
            mask_where = np.array(np.where(mask))
            if mask_where.size == 0:
                continue

            lower_bounds = mask_where.min(axis=-1)[::-1]
            upper_bounds = mask_where.max(axis=-1)[::-1]
            keys.append(ann_obj_id)
            boxes.append(np.array([lower_bounds, upper_bounds]))

        boxes = np.array(boxes)
        predicted_masks, _scores, _ = self.sam2_model.predict(
            point_coords=None,
            point_labels=None,
            box=boxes,
            multimask_output=False,
        )
        return {
            k: {"segmentation": v[0] > 0.5}
            for k, v in zip(keys, predicted_masks)
        }

    def _sam_predict_complicated(self, image, initial_masks):
        """Refine masks using SAM2 video predictor (unused when simple_sam2_mode=True)."""
        video_segments = []
        with tempfile.TemporaryDirectory() as tmpdirname:
            _save_image(str(pathlib.Path(tmpdirname) / f"{0}.jpg"), image)
            inference_state = self.predictor.init_state(video_path=tmpdirname)
            self.predictor.reset_state(inference_state)

            for ann_obj_id, mask_entry in initial_masks.items():
                mask = mask_entry["segmentation"]
                mask_where = np.array(np.where(mask))
                if mask_where.size == 0:
                    continue

                lower_bounds = mask_where.min(axis=-1)[::-1]
                upper_bounds = mask_where.max(axis=-1)[::-1]

                ann_frame_idx = 0
                _, out_obj_ids, out_mask_logits = (
                    self.predictor.add_new_points_or_box(
                        inference_state=inference_state,
                        frame_idx=ann_frame_idx,
                        obj_id=ann_obj_id,
                        box=np.array([lower_bounds, upper_bounds]),
                    )
                )

            for frame_ind, (
                out_frame_idx,
                out_obj_ids,
                out_mask_logits,
            ) in enumerate(self.predictor.propagate_in_video(inference_state)):
                assert frame_ind == out_frame_idx
                video_segments.append(
                    {
                        out_obj_id: {
                            "segmentation": (out_mask_logits[i][0] > 0.0)
                            .cpu()
                            .numpy()
                        }
                        for i, out_obj_id in enumerate(out_obj_ids)
                    }
                )

        return video_segments[0]

    def _refine_single_image_masks(
        self, image: np.ndarray, masks: dict[int, dict]
    ) -> dict[int, dict]:
        """Refine masks for a single image using SAM2."""
        sam_masks = (
            self._sam_predict_simple(image, masks)
            if self.config["simple_sam2_mode"]
            else self._sam_predict_complicated(image, masks)
        )

        mapping = _build_sam_format_mask_mapping(masks, sam_masks)
        for old_id, (new_id, iou) in mapping.items():
            if iou > self.config["iou_refine_thresh"]:
                refined_mask = sam_masks[new_id]["segmentation"]

                if self.config["remove_stems_parts"]:
                    refined_mask = self._remove_stems_morphology(refined_mask)

                if (
                    self.config["morphology_join_stems"]
                    and iou > self.config["morphology_joining_iou_thresh"]
                ):
                    refined_mask = self._join_masks_morphology_stems(
                        masks[old_id]["segmentation"], refined_mask
                    )

                masks[old_id]["segmentation"] = refined_mask

        return masks

    def _predict_masks_internal(self, images: list) -> list[dict]:
        """Run YOLO tracking + SAM refinement pipeline."""
        masks = self.yolo_tracker_model.predict_masks(images)

        if self.config["remove_edge_detections"]:
            masks = [
                {
                    k: v
                    for k, v in masks_frame.items()
                    if self._filter_edge_areas(
                        v["segmentation"],
                        self.config["edge_threshold"],
                        self.config["area_share_threshold"],
                    )
                }
                for masks_frame in masks
            ]

        refined_masks = []
        for image, masks_frame in zip(images, masks):
            if self.device.startswith("cuda"):
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    if torch.cuda.get_device_properties(0).major >= 8:
                        torch.backends.cuda.matmul.allow_tf32 = True
                        torch.backends.cudnn.allow_tf32 = True

                    refined = self._refine_single_image_masks(
                        image, masks_frame
                    )
            else:
                refined = self._refine_single_image_masks(image, masks_frame)

            refined_masks.append(refined)

        return refined_masks

    def predict_masks(self, images: list) -> list[dict[int, dict]]:
        """Predict masks for a sequence of images.

        Args:
            images: sequence of RGB images.

        Returns:
            List of mask dicts, one per image.
        """
        if self.config["ensure_same_image_sizes"]:
            images = ensure_same_image_sizes(images, "")
        masks = self._predict_masks_internal(images)
        masks = self._renumerate_masks(masks)
        return masks
