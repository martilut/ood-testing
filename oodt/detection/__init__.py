"""
oodt.detection — OOD detection methods for tabular data.

Available detectors
-------------------
KNNOODDetector        k-nearest-neighbour distance
ClusteringOODDetector k-means centroid distance
EnergyOODDetector     energy-based (GMM density or classifier logits)

Comparison pipeline
-------------------
OODDetectionPipeline  fit, calibrate, and compare multiple detectors
DetectorResult        result container for a single detector
"""

from oodt.detection.base import BaseOODDetector
from oodt.detection.knn import KNNOODDetector
from oodt.detection.clustering import ClusteringOODDetector
from oodt.detection.energy import EnergyOODDetector
from oodt.detection.pipeline import OODDetectionPipeline, DetectorResult

__all__ = [
    "BaseOODDetector",
    "KNNOODDetector",
    "ClusteringOODDetector",
    "EnergyOODDetector",
    "OODDetectionPipeline",
    "DetectorResult",
]
