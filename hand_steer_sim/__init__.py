# hand_steer_sim/__init__.py

__version__ = "0.1.0"

# Factory for choosing between Static vs Steering recognisers
from .dispatcher import RecognizerFactory

# Expose the two concrete recognisers at top level
from .model.static_mode.gesture_recognition import GestureRecognition
from .model.steering_mode.steering_recognition import SteeringRecognition

__all__ = [
    "RecognizerFactory",
    "GestureRecognition",
    "SteeringRecognition",
]
