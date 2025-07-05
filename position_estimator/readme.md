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
# 이때, yolo_msg를 받기위한 세팅이 되어있어야 함.

# 의존성 설치
cd ~/your_ws
rosdep install --from-paths src --ignore-src -r -y

# 빌드
colcon build --packages-select position_estimator
source install/setup.bash
```

## 사용법
### 1. 기본 실행 (모델 없이)
```bash
ros2 run position_estimator position_estimator_node \
  --ros-args -p camera_name:=demo_1
```

### 2. 모델과 함께 실행
```bash
ros2 run position_estimator position_estimator_node \
  --ros-args -p camera_name:=demo_1 \
  -p model_path:=/path/to/your/model.pth \
  -p device:=cuda
```
#### 사용할 때 camera_name:=demo_1, demo_2, doncheon_1 ...등 원하는 카메라 적용해서 실행
#### 여러개의 카메라를 한번에 실행시켜야할 경우 여러개 실행.

## 토픽
### 입력 토픽
- `/metasejong2025/cameras/{camera_name}/image_raw`: 카메라 이미지
- `/metasejong2025/cameras/{camera_name}/camera_info`: 카메라 파라미터
- `/yolo/{camera_name}/detections`: YOLO detection 결과 (DetectionArray)

### 출력 토픽
- `/position_estimator/{camera_name}/trash_positions`: 3D 위치 (PointStamped)
- `/position_estimator/{camera_name}/results`: 전체 결과 (JSON String)

## 지원 카메라
다음 카메라들이 미리 설정되어 있습니다:
- `demo_1`, `demo_2`
- `dongcheon_1`, `dongcheon_2`
- `gwanggaeto_1`, `gwanggaeto_2`
- `jiphyeon_1`, `jiphyeon_2`

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
└── README.md
```

## 의존성
- rclpy
- geometry_msgs
- sensor_msgs
- std_msgs
- yolo_msgs
- cv_bridge
- torch (모델 사용시)
- torchvision (모델 사용시)
- opencv-python
- numpy

## 모델 파일
학습된 ConvNeXt 모델(.pth 파일)을 준비하고 `model_path` 파라미터로 경로를 지정하세요.
모델 없이도 기하학적 계산만으로 동작 가능합니다.

## 결과 포맷 (수정된 간단한 버전)
```json
{
  "class_name": "cracker_box",
  "score": 0.96,
  "position": [x, y, z]
}
```

## 파라미터
- `camera_name`: 사용할 카메라명 (기본값: demo_1)
- `model_path`: ConvNeXt 모델 경로 (선택사항)
- `device`: 추론 장치 (cpu/cuda, 기본값: cpu)
- `confidence_threshold`: 신뢰도 임계값 (기본값: 0.5)
