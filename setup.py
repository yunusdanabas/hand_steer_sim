#!/usr/bin/env python3
from setuptools import setup, find_packages

setup(
    name="hand_steer_sim",
    version="0.1.0",
    packages=find_packages(include=["hand_steer_sim", "hand_steer_sim.*"]),
    scripts=[
        "scripts/camera_publisher_node.py",
        "scripts/hand_sign_recognition_node.py",
        "scripts/gesture_to_twist_node.py",
    ],
    install_requires=[
        "numpy",
        "opencv-python",
        "mediapipe",
    ],
)
