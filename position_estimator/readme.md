# Position Estimator Package

Meta-Sejong AI Robotics Challenge Stage 1용 위치 추정 패키지입니다.

## 기능

- **Ray-Plane 기하학적 계산**: YOLO detection 결과를 3D 월드 좌표로 변환
- **ConvNeXt 모델 추론**: 쓰레기 분류 및 위치 정제
- **ROS2 통합**: 실시간 토픽 처리

## 설치

```bash
# 워크스페이스에 패키지 복사
cd ~/your_ws/src
git clone <this_repo> position_estimator

# 의존성 설치
cd ~/your_ws
rosdep install --from-paths src --ignore-src -r -y

# 빌드
colcon build --packages-select position_estimator
source install/setup.bash
```

## 사용법

### 1. 기본 실행

```bash
ros2 launch position_estimator position_estimator.launch.py camera_name:=demo_1
```

### 2. 모델 경로 지정

```bash
ros2 launch position_estimator position_estimator.launch.py \
  camera_name:=demo_1 \
  model_path:=/path/to/your/model.pth \
  device:=cuda
```

### 3. 수동 실행

```bash
ros2 run position_estimator position_estimator_node \
  --ros-args -p camera_name:=demo_1 -p model_path:=/path/to/model.pth
```

## 토픽

### 입력 토픽
- `/metasejong2025/cameras/{camera_name}/image_raw`: 카메라 이미지
- `/metasejong2025/cameras/{camera_name}/camera_info`: 카메라 파라미터
- `/yolo/{camera_name}/detections`: YOLO detection 결과 (JSON)

### 출력 토픽
- `/position_estimator/{camera_name}/trash_positions`: 3D 위치 (PointStamped)
- `/position_estimator/{camera_name}/results`: 전체 결과 (JSON)

## 설정

### 카메라 파라미터
`position_estimator_node.py`에서 다음 값들을 실제 설정에 맞게 수정하세요:

```python
camera_params = {
    'position': [-60.86416, 152.94693, 21.47511],  # 카메라 월드 좌표
    'rotation': [-74.492, -23.79, -168.446],       # 카메라 회전 (도)
    'camera_height': 4.97511,                      # 지면으로부터 카메라 높이
    'ground_height': 16.5                          # 지면 높이
}
```

## 패키지 구조

```
position_estimator/
├── package.xml
├── setup.py
├── position_estimator/
│   ├── __init__.py
│   ├── position_estimator_node.py    # 메인 ROS2 노드
│   ├── ray_plane_estimator.py        # 기하학적 계산
│   └── convnext_model.py             # ConvNeXt 모델
├── launch/
│   └── position_estimator.launch.py
└── README.md
```

## 의존성

- rclpy
- geometry_msgs
- sensor_msgs
- cv_bridge
- torch
- torchvision
- opencv-python
- numpy

## 모델 파일

학습된 ConvNeXt 모델(.pth 파일)을 준비하고 `model_path` 파라미터로 경로를 지정하세요.

## 결과 포맷

```json
{
  "timestamp": "...",
  "bbox": [x1, y1, x2, y2],
  "geometric_position": [x, y, z],