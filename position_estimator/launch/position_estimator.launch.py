#!/usr/bin/env python3
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare

def generate_launch_description():
    return LaunchDescription([
        # Launch arguments
        DeclareLaunchArgument(
            'camera_name',
            default_value='demo_1',
            description='Camera name (demo_1, demo_2, etc.)'
        ),
        
        DeclareLaunchArgument(
            'model_path',
            default_value=PathJoinSubstitution([
                FindPackageShare('position_estimator'),
                'models',
                'best_position_model.pth'
            ]),
            description='Path to ConvNeXt model file'
        ),
        
        DeclareLaunchArgument(
            'device',
            default_value='cpu',
            description='Device for model inference (cpu/cuda)'
        ),
        
        DeclareLaunchArgument(
            'confidence_threshold',
            default_value='0.5',
            description='Confidence threshold for predictions'
        ),
        
        # Position estimator node
        Node(
            package='position_estimator',
            executable='position_estimator_node',
            name=['position_estimator_', LaunchConfiguration('camera_name')],
            parameters=[{
                'camera_name': LaunchConfiguration('camera_name'),
                'model_path': LaunchConfiguration('model_path'),
                'device': LaunchConfiguration('device'),
                'confidence_threshold': LaunchConfiguration('confidence_threshold'),
            }],
            output='screen'
        ),
    ])
