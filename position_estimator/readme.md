# Position Estimator Package

Position estimation package for Meta-Sejong AI Robotics Challenge Stage 1.

## Features

- **Ray-Plane Geometric Calculation**: Convert YOLO detection results to 3D world coordinates
- **ConvNeXt Model Inference**: Trash classification and position refinement
- **ROS2 Integration**: Real-time topic processing

## Installation

```bash
# Load Model
https://www.kaggle.com/code/kjin3386/position-estimation-via-multi-modal-convnext
The above link is part of the META-SEJONG CHALLENGE (2025 IEEE Metacom), 
which is the Kaggle notebook used for training.
You can train, test, and extract models using the code and dataset inside.

# Copy package to workspace
cd ~/your_ws/src
git clone <this_repo> position_estimator
# Note: Setup for receiving yolo_msg should be configured

# Install dependencies
cd ~/your_ws
rosdep install --from-paths src --ignore-src -r -y

# Build
colcon build --packages-select position_estimator
source install/setup.bash
```

## Usage

### 1. Basic Execution (without model)
```bash
ros2 run position_estimator position_estimator_node \
  --ros-args -p camera_name:=demo_1
```

### 2. Execution with Model
```bash
ros2 run position_estimator position_estimator_node \
  --ros-args -p camera_name:=demo_1 \
  -p model_path:=/path/to/your/model.pth \
  -p device:=cuda
```

#### When using, apply desired camera with camera_name:=demo_1, demo_2, doncheon_1, etc.
#### For multiple cameras simultaneously, run multiple instances.

## Topics

### Input Topics
- `/metasejong2025/cameras/{camera_name}/image_raw`: Camera image
- `/metasejong2025/cameras/{camera_name}/camera_info`: Camera parameters
- `/yolo/{camera_name}/detections`: YOLO detection results (DetectionArray)

### Output Topics
- `/position_estimator/{camera_name}/trash_positions`: 3D positions (PointStamped)
- `/position_estimator/{camera_name}/results`: Complete results (JSON String)

## Supported Cameras

The following cameras are pre-configured:
- `demo_1`, `demo_2`
- `dongcheon_1`, `dongcheon_2`
- `gwanggaeto_1`, `gwanggaeto_2`
- `jiphyeon_1`, `jiphyeon_2`

## Package Structure

```
position_estimator/
├── package.xml
├── setup.py
├── position_estimator/
│   ├── __init__.py
│   ├── position_estimator_node.py    # Main ROS2 node
│   ├── ray_plane_estimator.py        # Geometric calculation
│   └── convnext_model.py             # ConvNeXt model
└── README.md
```

## Dependencies

- rclpy
- geometry_msgs
- sensor_msgs
- std_msgs
- yolo_msgs
- cv_bridge
- torch (when using model)
- torchvision (when using model)
- opencv-python
- numpy

## Model File

Prepare a trained ConvNeXt model (.pth file) and specify the path with the `model_path` parameter.
The system can operate with geometric calculations alone without a model.

## Result Format (Simplified Version)

```json
{
  "class_name": "cracker_box",
  "score": 0.96,
  "position": [x, y, z]
}
```

## Parameters

- `camera_name`: Camera name to use (default: demo_1)
- `model_path`: ConvNeXt model path (optional)
- `device`: Inference device (cpu/cuda, default: cpu)
- `confidence_threshold`: Confidence threshold (default: 0.5)
