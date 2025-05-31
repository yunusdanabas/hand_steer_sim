#!/usr/bin/env python3
# setup.py  – package + ROS nodes for Hand-Steer-Sim
#
# Run  « pip install -e . »  at the workspace root to get
#  importable Python modules
#  console-scripts that ROS launch files can call directly
# ---------------------------------------------------------

from pathlib import Path
from setuptools import setup, find_packages

PKG_NAME = "hand_steer_sim"

# ─── helper: read long-description from README ──────────────────────────
this_dir = Path(__file__).resolve().parent
readme   = (this_dir / "README.md").read_text(encoding="utf-8") \
           if (this_dir / "README.md").exists() else ""

# ─── package metadata ──────────────────────────────────────────────────
setup(
    name=PKG_NAME,
    version="0.2.0",                         
    description="ROS-integrated hand-gesture control for mobile robots",
    long_description=readme,
    long_description_content_type="text/markdown",
    author="Yunus E. Danabas",
    python_requires=">=3.8",
    packages=find_packages(include=[f"{PKG_NAME}", f"{PKG_NAME}.*"]),

    # ─── console entry points ──────────────────────────────────────────
    entry_points={
        "console_scripts": [
            # Camera and data tools
            "hsim_camera_pub   = hand_steer_sim.scripts.camera_publisher_node:main",
            "hsim_record_data  = hand_steer_sim.scripts.hand_steer_data_recorder:main",
            "hsim_test_gest    = hand_steer_sim.scripts.test_gestures:main",

            # Static mode ROS nodes
            "hsim_hand_sign    = hand_steer_sim.scripts.hand_sign_recognition_node:main",
            "hsim_gest2twist   = hand_steer_sim.scripts.gesture_to_twist_node:main",

            # Steering mode ROS nodes
            "hsim_steer_sign   = hand_steer_sim.scripts.steering_sign_recognition_node:main",
            "hsim_wheel2twist  = hand_steer_sim.scripts.wheel_to_twist_node:main",
        ]
    },

    # ─── dependencies ──────────────────────────────────────────────────
    install_requires=[
        # Core dependencies
        "numpy>=1.24.0",           # Latest stable version
        "opencv-python>=4.8.0",    # Latest stable version
        "mediapipe>=0.10.0",       # Hand tracking and gesture recognition
        "tensorflow~=2.15.0 ; platform_machine!='aarch64'",  # ML framework
    ],
    
    extras_require={
        "realsense": ["pyrealsense2>=2.54.0"],  # Intel RealSense camera support
        "dev": [
            "black>=23.0.0",       # Code formatting
            "flake8>=6.0.0",       # Code linting
            "pytest>=7.0.0",       # Testing framework
        ],
    },

    # ─── package data ──────────────────────────────────────────────────
    package_data={
        "hand_steer_sim": [
            # ML models and labels
            "model/**/**/*.tflite",
            "model/**/**/*.csv",
            # Robot configuration
            "urdf/*.xacro",
            "urdf/*.rviz",
            "config/*.yaml",
        ]
    },

    # ─── metadata ──────────────────────────────────────────────────────
    classifiers=[
        "Development Status :: 4 - Beta",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "License :: OSI Approved :: MIT License",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: System :: Robotics",
        "Operating System :: POSIX :: Linux",
        "Intended Audience :: Science/Research",
    ],
    
    include_package_data=True,
    zip_safe=False,
)
