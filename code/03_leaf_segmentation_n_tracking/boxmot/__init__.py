# Mikel Broström 🔥 Yolo Tracking 🧾 AGPL-3.0 license

__version__ = '11.0.6'

from boxmot.postprocessing.gsi import gsi
from boxmot.tracker_zoo import create_tracker, get_tracker_config
from boxmot.trackers.botsort.botsort import BotSort
from boxmot.trackers.bytetrack.bytetrack import ByteTrack
from boxmot.trackers.deepocsort.deepocsort import DeepOcSort
from boxmot.trackers.strongsort.strongsort import StrongSort


TRACKERS = ['bytetrack', 'botsort', 'strongsort', 'deepocsort']

__all__ = ("__version__",
           "StrongSort", "ByteTrack", "BotSort", "DeepOcSort",
           "create_tracker", "get_tracker_config", "gsi")
