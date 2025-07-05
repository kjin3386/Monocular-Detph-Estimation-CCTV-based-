"""Position Estimator Package for Meta-Sejong AI Robotics Challenge"""

__version__ = "1.0.0"
__author__ = "Your Name"

# Import main modules for easier access
try:
    from .ray_plane_estimator import RayPlaneEstimator
    from .convnext_model import ConvNeXtPositionEstimator, ModelInference
    from .position_estimator_node import PositionEstimatorNode
except ImportError:
    # Handle case where dependencies are not available
    pass
