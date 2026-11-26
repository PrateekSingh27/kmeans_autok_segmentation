"""kmseg - K-Means image segmentation with automatic K selection."""
from .kmeans import KMeans
from .selection import ALL_METHODS, SelectionResult, select_k
from .segmentation import AutoKSegmenter, SegmentationResult, build_features, load_image
from .synthetic import make_synthetic_image

__all__ = ["KMeans", "select_k", "SelectionResult", "ALL_METHODS", "AutoKSegmenter",
           "SegmentationResult", "build_features", "load_image", "make_synthetic_image"]
__version__ = "1.0.0"
