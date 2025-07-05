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

from .ray_plane_estimator import RayPlaneEstimator
from .convnext_model import ModelInference

# Assuming YOLO detection message structure
class Detection:
    def __init__(self, x1, y1, x2, y2, confidence, class_id, class_name):
        self.x1 = x1
        self.y1 = y1  
        self.x2 = x2
        self.y2 = y2
        self.confidence = confidence
        self.class_id = class_id
        self.class_name = class_name

class PositionEstimatorNode(Node):
    def __init__(self):
        super().__init__('position_estimator_node')
        
        # Parameters
        self.declare_parameter('camera_name', 'demo_1')
        self.declare_parameter('model_path', '/path/to/your/model.pth')
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
        
        # Load model
        if os.path.exists(model_path):
            self.model_inference = ModelInference(model_path, device)
            self.get_logger().info(f"Model loaded from {model_path}")
        else:
            self.get_logger().warn(f"Model file not found: {model_path}")
        
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
            String,  # Assuming detections come as JSON string
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

    def camera_info_callback(self, msg):
        """Process camera info and initialize ray-plane estimator"""
        if self.estimator is None:
            # Extract camera parameters from CameraInfo
            K = np.array(msg.k).reshape(3, 3)
            fx, fy = K[0, 0], K[1, 1]
            cx, cy = K[0, 2], K[1, 2]
            
            # Camera parameters (these should be configured for your specific setup)
            camera_params = {
                'fx': fx,
                'fy': fy,
                'cx': cx,
                'cy': cy,
                'image_width': msg.width,
                'image_height': msg.height,
                'position': [-60.86416, 152.94693, 21.47511],  # Update with actual values
                'rotation': [-74.492, -23.79, -168.446],       # Update with actual values
                'camera_height': 4.97511,                      # Update with actual values
                'ground_height': 16.5                          # Update with actual values
            }
            
            self.estimator = RayPlaneEstimator(camera_params)
            self.camera_info = msg
            self.get_logger().info("Ray-plane estimator initialized")

    def image_callback(self, msg):
        """Store current image"""
        try:
            self.current_image = self.cv_bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception as e:
            self.get_logger().error(f"Error converting image: {e}")

    def detection_callback(self, msg):
        """Process YOLO detections and estimate positions"""
        if self.estimator is None or self.current_image is None:
            return
        
        try:
            # Parse detection data (adjust based on your YOLO message format)
            detections_data = json.loads(msg.data)
            
            # Process each detection
            for detection_data in detections_data:
                self.process_detection(detection_data)
                
        except Exception as e:
            self.get_logger().error(f"Error processing detections: {e}")

    def process_detection(self, detection_data):
        """Process single detection"""
        try:
            # Extract bounding box (adjust field names based on your data structure)
            bbox = detection_data.get('bbox', [])
            if len(bbox) < 4:
                return
            
            x1, y1, x2, y2 = bbox[:4]
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            
            # Estimate 3D position using ray-plane calculation
            position_result = self.estimator.estimate_position(center_x, center_y)
            if position_result is None:
                return
            
            position = position_result['position']
            distance = position_result['distance']
            
            # Use ConvNeXt model for classification and position refinement
            refined_result = None
            if self.model_inference is not None:
                refined_result = self.model_inference.predict_trash(
                    self.current_image, [x1, y1, x2, y2], distance
                )
            
            # Create result message
            result = {
                'timestamp': self.get_clock().now().to_msg(),
                'bbox': [x1, y1, x2, y2],
                'geometric_position': position.tolist(),
                'distance': float(distance),
                'raw_detection': detection_data
            }
            
            if refined_result is not None:
                result.update({
                    'predicted_class': refined_result['class_name'],
                    'class_confidence': float(refined_result['confidence']),
                    'position_offset': refined_result['position_offset'].tolist()
                })
                
                # Apply position refinement
                refined_position = position + refined_result['position_offset']
                result['refined_position'] = refined_position.tolist()
                
                # Only publish if confidence is above threshold
                if refined_result['confidence'] < self.confidence_threshold:
                    return
            
            # Publish position
            self.publish_position(position, refined_result)
            
            # Publish full result
            result_msg = String()
            result_msg.data = json.dumps(result)
            self.result_pub.publish(result_msg)
            
            self.get_logger().info(
                f"Detected trash at position: ({position[0]:.2f}, {position[1]:.2f}, {position[2]:.2f})"
            )
            
        except Exception as e:
            self.get_logger().error(f"Error processing detection: {e}")

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