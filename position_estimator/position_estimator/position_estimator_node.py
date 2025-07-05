#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import Point, PointStamped
from std_msgs.msg import Header, String
from cv_bridge import CvBridge
import cv2
import numpy as np
import json
import os
import time

# Import YOLO message types
from yolo_msgs.msg import DetectionArray

from .ray_plane_estimator import RayPlaneEstimator
from .convnext_model import ModelInference

class PositionEstimatorNode(Node):
    def __init__(self):
        super().__init__('position_estimator_node')
        
        # Parameters
        self.declare_parameter('camera_name', 'demo_1')
        self.declare_parameter('model_path', '')  # 빈 문자열로 설정
        self.declare_parameter('device', 'cpu')
        self.declare_parameter('confidence_threshold', 0.5)
        
        camera_name = self.get_parameter('camera_name').value
        model_path = self.get_parameter('model_path').value
        device = self.get_parameter('device').value
        self.confidence_threshold = self.get_parameter('confidence_threshold').value
        
        # Initialize components
        self.cv_bridge = CvBridge()
        self.estimator = None
        self.model_inference = None
        self.current_image = None
        self.camera_info = None
        
        # Load model if path is provided
        if model_path and os.path.exists(model_path):
            self.model_inference = ModelInference(model_path, device)
            self.get_logger().info(f"Model loaded from {model_path}")
        elif model_path:
            self.get_logger().warn(f"Model file not found: {model_path}")
        else:
            self.get_logger().info("No model path provided, running with geometric calculation only")
        
        # Subscribers
        self.image_sub = self.create_subscription(
            Image,
            f'/metasejong2025/cameras/{camera_name}/image_raw',
            self.image_callback,
            10
        )
        
        self.camera_info_sub = self.create_subscription(
            CameraInfo,
            f'/metasejong2025/cameras/{camera_name}/camera_info',
            self.camera_info_callback,
            10
        )
        
        self.detection_sub = self.create_subscription(
            DetectionArray,
            f'/yolo/{camera_name}/detections',
            self.detection_callback,
            10
        )
        
        # Publishers
        self.position_pub = self.create_publisher(
            PointStamped,
            f'/position_estimator/{camera_name}/trash_positions',
            10
        )
        
        self.result_pub = self.create_publisher(
            String,
            f'/position_estimator/{camera_name}/results',
            10
        )
        
        self.get_logger().info(f"Position estimator node initialized for camera: {camera_name}")

    def get_camera_params(self, camera_name, msg):
        """Get camera-specific parameters"""
        # 카메라별 하드코딩된 파라미터
        camera_configs = {
            # Demo scenarios
            'demo_1': {
                'position': [-60.86416, 152.94693, 21.47511],
                'rotation': [-74.492, -23.79, -168.446],
                'camera_height': 4.97511,
                'ground_height': 16.5
            },
            'demo_2': {
                'position': [-31.50976, 136.45802, 21.27105],
                'rotation': [-72.435, 30.821, 167.397],
                'camera_height': 4.47105,  # 21.27105 - 16.8
                'ground_height': 16.8
            },
            
            # Dongcheon scenarios
            'dongcheon_1': {
                'position': [-55.64634, -9.32143, 21.47511],
                'rotation': [72.326, -16.222, -14.042],
                'camera_height': 5.47511,  # 21.47511 - 16.0
                'ground_height': 16.0
            },
            'dongcheon_2': {
                'position': [-55.54975, 21.41175, 19.37973],
                'rotation': [-67.918, -61.773, -164.059],
                'camera_height': 3.57973,  # 19.37973 - 15.8
                'ground_height': 15.8
            },
            
            # Gwanggaeto scenarios
            'gwanggaeto_1': {
                'position': [-51.92482, -106.64152, 29.81015],
                'rotation': [60.312, -43.533, -26.156],
                'camera_height': 8.41015,  # 29.81015 - 21.4
                'ground_height': 21.4
            },
            'gwanggaeto_2': {
                'position': [-53.89861, -55.17218, 27.25073],
                'rotation': [-71.609, -29.782, -159.929],
                'camera_height': 5.85073,  # 27.25073 - 21.4
                'ground_height': 21.4
            },
            
            # Jiphyeon scenarios
            'jiphyeon_1': {
                'position': [42.24149, -236.6242, 32.3144],
                'rotation': [51.617, 29.087, -6.284],
                'camera_height': 10.9144,  # 32.3144 - 21.4
                'ground_height': 21.4
            },
            'jiphyeon_2': {
                'position': [11.85726, -235.90986, 32.32018],
                'rotation': [53.78, -11.493, -1.764],
                'camera_height': 10.92018,  # 32.32018 - 21.4
                'ground_height': 21.4
            }
        }
        
        if camera_name not in camera_configs:
            self.get_logger().error(f"Unknown camera: {camera_name}")
            return None
            
        config = camera_configs[camera_name]
        
        # CameraInfo에서 내부 파라미터 추출
        K = np.array(msg.k).reshape(3, 3)
        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]
        
        return {
            'fx': fx,
            'fy': fy,
            'cx': cx,
            'cy': cy,
            'image_width': msg.width,
            'image_height': msg.height,
            **config  # 카메라별 설정 병합
        }

    def camera_info_callback(self, msg):
        """Process camera info and initialize ray-plane estimator"""
        if self.estimator is None:
            camera_name = self.get_parameter('camera_name').value
            camera_params = self.get_camera_params(camera_name, msg)
            
            if camera_params is None:
                return
                
            self.estimator = RayPlaneEstimator(camera_params)
            self.camera_info = msg
            self.get_logger().info(f"Ray-plane estimator initialized for {camera_name}")

    def image_callback(self, msg):
        """Store current image"""
        try:
            self.current_image = self.cv_bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception as e:
            self.get_logger().error(f"Error converting image: {e}")

    def detection_callback(self, msg):
        """Process YOLO detections and estimate positions"""
        self.get_logger().info(f"Received detection message with {len(msg.detections)} detections")
        
        if self.estimator is None:
            self.get_logger().warn("Ray-plane estimator not ready")
            return
            
        if self.current_image is None:
            self.get_logger().warn("No image available")
            return
        
        try:
            # Process each detection in the DetectionArray
            for i, detection in enumerate(msg.detections):
                self.get_logger().info(f"Processing detection {i}: {detection.class_name} (score: {detection.score:.3f})")
                self.process_detection(detection)
                
        except Exception as e:
            self.get_logger().error(f"Error processing detections: {e}")
            import traceback
            self.get_logger().error(f"Traceback: {traceback.format_exc()}")

    def process_detection(self, detection):
        """Process single detection"""
        try:
            self.get_logger().info(f"Starting to process {detection.class_name}")
            
            # Extract bounding box from YOLO detection
            center_x = detection.bbox.center.position.x
            center_y = detection.bbox.center.position.y
            size_x = detection.bbox.size.x
            size_y = detection.bbox.size.y
            
            self.get_logger().info(f"Bbox center: ({center_x:.1f}, {center_y:.1f}), size: ({size_x:.1f}, {size_y:.1f})")
            
            # Convert center+size to corner coordinates
            x1 = center_x - size_x / 2
            y1 = center_y - size_y / 2
            x2 = center_x + size_x / 2
            y2 = center_y + size_y / 2
            
            # Estimate 3D position using ray-plane calculation
            self.get_logger().info(f"Calling estimator.estimate_position({center_x:.1f}, {center_y:.1f})")
            position_result = self.estimator.estimate_position(center_x, center_y)
            
            if position_result is None:
                self.get_logger().warn(f"Position estimation failed for {detection.class_name}")
                return
            
            position = position_result['position']
            distance = position_result['distance']
            
            self.get_logger().info(f"Position estimated: {position}, distance: {distance:.2f}")
            
            # Use ConvNeXt model for classification and position refinement
            refined_result = None
            if self.model_inference is not None:
                self.get_logger().info("Running model inference...")
                refined_result = self.model_inference.predict_trash(
                    self.current_image, [x1, y1, x2, y2], distance
                )
            
            # Create result message (simplified)
            result = {
                'class_name': detection.class_name,
                'score': detection.score,
                'position': position.tolist()
            }
            
            if refined_result is not None:
                result['predicted_class'] = refined_result['class_name']
                result['class_confidence'] = float(refined_result['confidence'])
                
                # Apply position refinement
                refined_position = position + refined_result['position_offset']
                result['position'] = refined_position.tolist()  # 정제된 위치로 업데이트
                
                # Only publish if confidence is above threshold
                if refined_result['confidence'] < self.confidence_threshold:
                    self.get_logger().info(f"Confidence {refined_result['confidence']:.3f} below threshold {self.confidence_threshold}")
                    return
            
            # Publish position
            self.publish_position(position, refined_result)
            
            # Publish full result
            result_msg = String()
            result_msg.data = json.dumps(result)
            self.result_pub.publish(result_msg)
            
            self.get_logger().info(
                f"Detected {detection.class_name} at position: ({position[0]:.2f}, {position[1]:.2f}, {position[2]:.2f})"
            )
            
        except Exception as e:
            self.get_logger().error(f"Error processing detection {detection.class_name}: {e}")
            import traceback
            self.get_logger().error(f"Traceback: {traceback.format_exc()}")

    def publish_position(self, position, refined_result=None):
        """Publish 3D position as PointStamped"""
        point_msg = PointStamped()
        point_msg.header = Header()
        point_msg.header.stamp = self.get_clock().now().to_msg()
        point_msg.header.frame_id = "world"  # or your world frame
        
        point_msg.point.x = float(position[0])
        point_msg.point.y = float(position[1])
        point_msg.point.z = float(position[2])
        
        self.position_pub.publish(point_msg)


def main(args=None):
    rclpy.init(args=args)
    node = PositionEstimatorNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
