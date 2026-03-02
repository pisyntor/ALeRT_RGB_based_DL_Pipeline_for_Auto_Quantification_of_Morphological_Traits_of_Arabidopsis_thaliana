# BoxMOT - Multi-Object Tracking Library

BoxMOT (v11.0.6) is a multi-object tracking library by Mikel Brostrom that provides
several tracking algorithms sharing a common `BaseTracker` interface. Each tracker
accepts per-frame detections (bounding boxes + confidence + class) and returns
persistent track IDs across frames.

This is a local vendored copy used by the leaf segmentation and tracking pipeline.

## Trackers

All trackers live under `boxmot/trackers/` and inherit from `BaseTracker`
(`basetracker.py`), which provides:

- Abstract `update(dets, img, embs=None)` method that subclasses implement.
- Per-class tracking support (maintain separate tracks per object class).
- Visualization helpers: `plot_box_on_img()`, `plot_results()`,
  `plot_trackers_trajectories()`.
- Track history management.

### ByteTrack (`bytetrack/`)

Pure motion-based tracker (no ReID appearance model needed).

- **Association strategy**: Two-stage. First matches high-confidence detections to
  existing tracks via IoU + linear assignment, then recovers low-confidence
  detections ("bytes") in a second pass. This is its key innovation and makes it
  robust in crowded/occluded scenes.
- **Motion model**: Shared Kalman filter in XYAH space (center-x, center-y, aspect
  ratio, height).
- **When to use**: Fast, lightweight, good baseline. Best when appearance features
  are unreliable or unnecessary.

### StrongSort (`strongsort/`)

Appearance + motion tracker combining ReID embeddings with the classic SORT
framework.

- **Appearance model**: `ReidAutoBackend` extracts per-detection appearance
  embeddings. Matching uses cosine distance via `NearestNeighborDistanceMetric`.
- **Motion model**: Camera Motion Correction (CMC) via ECC method, plus a Kalman
  filter for state prediction.
- **Nested SORT module** (`sort/`): Contains `Detection`, `Track`, `Tracker`,
  `linear_assignment`, and `iou_matching` — the full SORT pipeline.
- **When to use**: When objects look distinct and you want ID consistency across
  occlusions and re-appearances. Requires ReID weights (e.g., `lmbn_n_cuhk03_d.pt`).

### DeepOCSoRT (`deepocsort/`)

Extended OC-SORT with deep appearance features.

- **Appearance model**: ReID via `ReidAutoBackend`, combined with motion cues.
- **Motion model**: Kalman filter in XYSR space (center-x, center-y, scale, aspect
  ratio). Maintains observation history and estimates speed/direction.
- **Camera motion**: Compensated via CMC.
- **When to use**: Good balance of appearance and motion. Handles camera motion and
  appearance changes well.

### BoTSORT (`botsort/`)

Combines appearance-based and motion-based tracking with camera motion compensation.

- **Appearance model**: ReID via `ReidAutoBackend` with configurable appearance
  threshold.
- **Motion model**: Kalman filter in XYWH space. Camera motion correction via
  Simple Optical Flow (SOF) or other CMC methods.
- **Association**: Two-stage (similar to ByteTrack) but fuses appearance embedding
  distance with IoU distance. Configurable `proximity_thresh` and
  `appearance_thresh`.
- **When to use**: When you need robust tracking with both appearance and motion
  cues, especially with camera movement.

### OCSoRT (`ocsort/`)

Observation-Centric SORT — a motion-only tracker (no ReID).

- **Motion model**: Kalman filter in XYSR space with observation history. Estimates
  speed and direction from previous observations.
- **Key idea**: Uses observation-centric momentum to recover from occlusions, rather
  than relying on Kalman predictions that degrade over time.
- **When to use**: When you want a motion-only tracker that handles occlusions
  better than vanilla SORT, without the overhead of an appearance model.

## Weights

ReID model weights used by appearance-based trackers (StrongSort, DeepOCSoRT,
BoTSORT) are stored in `boxmot/weights/`. These are specified via the
`reid_weights` parameter when constructing a tracker. The weights are derived from the [ReID model zoo](https://kaiyangzhou.github.io/deep-person-reid/MODEL_ZOO) available online.

## Usage

```python
from boxmot.trackers.bytetrack.bytetrack import ByteTrack
from boxmot.trackers.strongsort.strongsort import StrongSort
from boxmot.trackers.deepocsort.deepocsort import DeepOcSort
from boxmot.trackers.botsort.botsort import BotSort

# Motion-only tracker (no ReID weights needed)
tracker = ByteTrack(track_thresh=0.1, match_thresh=0.8, track_buffer=30, frame_rate=30)

# Appearance-based tracker (requires ReID weights)
tracker = StrongSort(
    reid_weights=Path("boxmot/weights/lmbn_n_cuhk03_d.pt"),
    device="cuda:0",
    half=False,
)

# Per-frame update: pass detections as Nx6 array [x1, y1, x2, y2, conf, cls]
tracks = tracker.update(detections, image)
# Returns Nx8 array [x1, y1, x2, y2, track_id, conf, cls, det_ind]
```
