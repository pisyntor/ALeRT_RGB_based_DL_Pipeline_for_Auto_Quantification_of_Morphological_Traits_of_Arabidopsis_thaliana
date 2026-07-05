# Mikel Broström 🔥 Yolo Tracking 🧾 AGPL-3.0 license

import numpy as np
from torch import device
from pathlib import Path

from boxmot.appearance.reid_auto_backend import ReidAutoBackend
from boxmot.trackers.deepsort.sort.detection import Detection
from boxmot.trackers.deepsort.sort.tracker import Tracker
from boxmot.utils.matching import NearestNeighborDistanceMetric
from boxmot.utils.ops import xyxy2tlwh
from boxmot.trackers.basetracker import BaseTracker


class DeepSort(object):
    """
    DeepSORT Tracker: the classic appearance + motion tracker from
    "Simple Online and Realtime Tracking with a Deep Association Metric"
    (Wojke et al., 2017).

    Detections are associated to tracks with a matching cascade that combines a
    deep ReID appearance descriptor (nearest-neighbour cosine distance over a
    per-track feature gallery) with a Kalman-filter motion gate, followed by an
    IoU fallback for unconfirmed / recently-lost tracks.

    Args:
        reid_weights (Path): Path to the model weights for ReID (Re-Identification).
        device (torch.device): Device on which to run the model (e.g., 'cpu' or 'cuda').
        half (bool): Whether to use half-precision (fp16) for faster inference on compatible devices.
        per_class (bool, optional): Whether to perform per-class tracking. If True, tracks are maintained separately for each object class.
        max_dist (float, optional): Maximum cosine distance for ReID feature matching in the Nearest Neighbor Distance Metric.
        max_iou_dist (float, optional): Maximum Intersection over Union (IoU) distance for data association during the IoU fallback step.
        max_age (int, optional): Maximum number of frames to keep a track alive without any detections.
        n_init (int, optional): Number of consecutive frames required to confirm a track.
        nn_budget (int, optional): Maximum size of the appearance feature gallery per track. Oldest features are dropped once the budget is exceeded.
    """

    def __init__(
        self,
        reid_weights: Path,
        device: device,
        half: bool,
        per_class: bool = False,
        max_dist=0.2,
        max_iou_dist=0.7,
        max_age=30,
        n_init=3,
        nn_budget=100,
    ):

        self.per_class = per_class
        self.model = ReidAutoBackend(
            weights=reid_weights, device=device, half=half
        ).model

        self.tracker = Tracker(
            metric=NearestNeighborDistanceMetric("cosine", max_dist, nn_budget),
            max_iou_dist=max_iou_dist,
            max_age=max_age,
            n_init=n_init,
        )

    @BaseTracker.per_class_decorator
    def update(self, dets: np.ndarray, img: np.ndarray, embs: np.ndarray = None) -> np.ndarray:
        assert isinstance(
            dets, np.ndarray
        ), f"Unsupported 'dets' input format '{type(dets)}', valid format is np.ndarray"
        assert isinstance(
            img, np.ndarray
        ), f"Unsupported 'img' input format '{type(img)}', valid format is np.ndarray"
        assert (
            len(dets.shape) == 2
        ), "Unsupported 'dets' dimensions, valid number of dimensions is two"
        assert (
            dets.shape[1] == 6
        ), "Unsupported 'dets' 2nd dimension lenght, valid lenghts is 6"

        dets = np.hstack([dets, np.arange(len(dets)).reshape(-1, 1)])
        xyxy = dets[:, 0:4]
        confs = dets[:, 4]
        clss = dets[:, 5]
        det_ind = dets[:, 6]

        # extract appearance information for each detection
        if embs is not None:
            features = embs
        else:
            features = self.model.get_features(xyxy, img)

        tlwh = xyxy2tlwh(xyxy)
        detections = [
            Detection(box, conf, cls, det_ind, feat) for
            box, conf, cls, det_ind, feat in
            zip(tlwh, confs, clss, det_ind, features)
        ]

        # update tracker
        self.tracker.predict()
        self.tracker.update(detections)

        # output bbox identities
        outputs = []
        for track in self.tracker.tracks:
            if not track.is_confirmed() or track.time_since_update >= 1:
                continue

            x1, y1, x2, y2 = track.to_tlbr()

            id = track.id
            conf = track.conf
            cls = track.cls
            det_ind = track.det_ind

            outputs.append(
                np.concatenate(([x1, y1, x2, y2], [id], [conf], [cls], [det_ind])).reshape(1, -1)
            )
        if len(outputs) > 0:
            return np.concatenate(outputs)
        return np.array([])
