# hand_steer_sim/__init__.py

"""
hand_steer_sim
~~~~~~~~~~~~~~

Thin wrapper around MediaPipe-based gesture recognition and ROS glue code.
"""

__version__ = "0.1.0"

# What symbols will be imported with “from hand_steer_sim import *”
__all__ = [
    "GestureRecognition",
]

# Expose the core class at the package level
from .gesture_recognition import GestureRecognition
