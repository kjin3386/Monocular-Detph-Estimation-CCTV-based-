# Monocular Depth Estimation ROS Package

**A comprehensive position estimation system for the 2025 IEEE-METACOM Meta-Sejong AI Robotics Challenge**

## Overview
This package provides a complete monocular depth and position estimation solution designed for the 2025 IEEE-METACOM Meta-Sejong AI Robotics Challenge. The system combines geometric ray-plane calculations with deep learning-based ConvNeXt models to accurately estimate 3D positions of trash objects from CCTV camera feeds for autonomous robotic collection.


For the complete simulation environment including Isaac Sim setup, please refer to the main competition repository: https://github.com/junsuk123/metasejong-airobotics. This repository contains the position estimation module specifically designed for Stage 1 of the competition.

<img width="1798" height="1005" alt="Image" src="https://github.com/user-attachments/assets/c95855d7-2830-4e78-a1c0-d210d6ee57cb" />

### Key Features

- **Hybrid Approach**: Combines physics-based geometric calculations with AI-powered position refinement
- **Ray-Plane Geometric Calculation**: Converts YOLO detection results to 3D world coordinates using camera intrinsics
- **ConvNeXt Deep Learning Model**: Multi-modal learning for trash classification and position refinement
- **Real-time ROS2 Integration**: Seamless integration with robotic systems through ROS2 topics
- **Multiple Camera Support**: Pre-configured for various competition cameras
- **Adaptive Position Refinement**: Intelligent blending of geometric and learned estimates

## Competition Context

This software was developed for **Stage 1** of the Meta-Sejong AI Robotics Challenge, where participants must:

1. **Object Detection**: Identify trash objects in CCTV camera feeds
2. **3D Localization**: Convert 2D detections to precise 3D world coordinates
3. **Classification**: Determine trash types for appropriate collection strategies
4. **Path Planning**: Provide accurate positions for optimal robot navigation

The challenge runs in Isaac Sim environment with multiple CCTV cameras providing RGB images and camera parameters via ROS2 messages.

## System Architecture
<img width="856" height="268" alt="Image" src="https://github.com/user-attachments/assets/b376f776-ceee-4226-8a2c-ff57b7c6c6f1" />


#### overall system architecture

<img width="593" height="613" alt="Image" src="https://github.com/user-attachments/assets/d539e44f-fe5e-4633-bf5f-c08f78f79b8e" />


#### visual-geometric network flowchart


## Technical Approach

### 1. Geometric Position Estimation
- **Input**: 2D bounding box center, camera intrinsics, camera pose
- **Method**: Ray-plane intersection with ground plane
- **Output**: Initial 3D world coordinates

### 2. Deep Learning Refinement
- **Architecture**: ConvNeXt-based multi-modal network
- **Inputs**: Cropped object image + initial geometric position
- **Training**: Position offset regression with Huber loss
- **Output**: Refined 3D position coordinates

### 3. Adaptive Fusion
- **Strategy**: Confidence-based blending of geometric and learned estimates
- **Fallback**: Robust error handling with geometric baseline
- **Constraints**: Physical boundary enforcement for realistic positions

## Installation

### Prerequisites

- ROS2 Humble (recommended)
- Python 3.8+
- CUDA-capable GPU (optional, for model inference)

### Dependencies

```bash
# System dependencies
sudo apt update
sudo apt install python3-pip python3-opencv

# Python packages
pip install torch torchvision
pip install albumentations
pip install timm
pip install numpy pandas matplotlib seaborn
pip install tabulate
```

### Model Training Setup

The training pipeline is available as a Kaggle notebook:

**🔗 [Training Notebook](https://www.kaggle.com/code/kjin3386/position-estimation-via-multi-modal-convnext)**

This notebook contains:
- Complete dataset preparation
- ConvNeXt model training
- Evaluation and visualization tools
- Pre-trained model downloads

### Package Installation

```bash
# Clone repository
cd ~/ros2_ws/src
git clone https://github.com/kjin3386/Monocular_Detph_Estimation_ROS_PACKAGE position_estimator

# Install ROS dependencies
cd ~/ros2_ws
rosdep install --from-paths src --ignore-src -r -y

# Build package
colcon build --packages-select position_estimator
source install/setup.bash
```

## Usage

### Basic Usage (Geometric Only)

```bash
ros2 run position_estimator position_estimator_node \
  --ros-args -p camera_name:=demo_1
```

### With Deep Learning Model

```bash
ros2 run position_estimator position_estimator_node \
  --ros-args -p camera_name:=demo_1 \
  -p model_path:=/path/to/trained_model.pth \
  -p device:=cuda
```

### Multi-Camera Setup

For competition scenarios with multiple cameras:

```bash
# Terminal 1
ros2 run position_estimator position_estimator_node \
  --ros-args -p camera_name:=demo_1 -p model_path:=/path/to/model.pth

# Terminal 2  
ros2 run position_estimator position_estimator_node \
  --ros-args -p camera_name:=demo_2 -p model_path:=/path/to/model.pth

# Terminal 3
ros2 run position_estimator position_estimator_node \
  --ros-args -p camera_name:=dongcheon_1 -p model_path:=/path/to/model.pth
```

## ROS2 Interface

### Subscribed Topics

- `/metasejong2025/cameras/{camera_name}/image_raw` ([sensor_msgs/Image])
- `/metasejong2025/cameras/{camera_name}/camera_info` ([sensor_msgs/CameraInfo])
- `/yolo/{camera_name}/detections` ([yolo_msgs/DetectionArray])

### Published Topics

- `/position_estimator/{camera_name}/trash_positions` ([geometry_msgs/PointStamped])
- `/position_estimator/{camera_name}/results` ([std_msgs/String] - JSON format)

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `camera_name` | string | `demo_1` | Camera identifier |
| `model_path` | string | `""` | Path to trained ConvNeXt model |
| `device` | string | `cpu` | Inference device (cpu/cuda) |
| `confidence_threshold` | double | `0.5` | Detection confidence threshold |

## Supported Cameras

Pre-configured camera settings for competition environments:

- **Demo Environment**: `demo_1`, `demo_2`
- **Dongcheon**: `dongcheon_1`, `dongcheon_2`  
- **Gwanggaeto**: `gwanggaeto_1`, `gwanggaeto_2`
- **Jiphyeon**: `jiphyeon_1`, `jiphyeon_2`

## Package Structure

```
position_estimator/
├── package.xml                     # ROS2 package configuration
├── setup.py                        # Python package setup
├── README.md                       # This file
├── position_estimator/
│   ├── __init__.py
│   ├── position_estimator_node.py  # Main ROS2 node
│   ├── ray_plane_estimator.py      # Geometric calculations
│   ├── convnext_model.py           # Deep learning model
│   └── config/
│       └── camera_configs.yaml     # Camera parameters
└── launch/
    ├── single_camera.launch.py     # Single camera launch
    └── multi_camera.launch.py      # Multi-camera launch
```

## Performance

### Accuracy Metrics (Test Results)

- **Average Initial Error**: 2.847m
- **Average Model Error**: 2.103m  
- **Mean Improvement**: 0.744m (26.1% reduction)
- **Success Rate**: 75.0% of cases improved

### Real-time Performance

- **Geometric Calculation**: <10ms per detection
- **Model Inference**: ~50ms per detection (GPU)
- **Total Pipeline**: <100ms end-to-end

## Research & Development

This work was developed as part of academic research for the IEEE-METACOM 2025 conference. Key contributions include:

1. **Hybrid Positioning**: Novel combination of geometric and learning-based approaches
2. **Multi-modal Learning**: Integration of visual and geometric features
3. **Real-time Performance**: Optimized pipeline for robotic applications
4. **Robust Fallbacks**: Graceful degradation when models fail



**Note**: This package was specifically developed for the 2025 IEEE-METACOM Meta-Sejong AI Robotics Challenge and is optimized for the Isaac Sim competition environment.
