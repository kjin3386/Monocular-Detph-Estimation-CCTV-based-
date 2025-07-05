#!/usr/bin/env python3
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare

def generate_launch_description():
    # 사용할 카메라 리스트
    cameras = ['demo_1', 'demo_2']  # 필요에 따라 추가
    
    # Launch arguments
    launch_args = [
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
    ]
    
    # 각 카메라마다 노드 생성
    nodes = []
    for camera in cameras:
        node = Node(
            package='position_estimator',
            executable='position_estimator_node',
            name=f'position_estimator_{camera}',
            parameters=[{
                'camera_name': camera,
                'model_path': LaunchConfiguration('model_path'),
                'device': LaunchConfiguration('device'),
                'confidence_threshold': LaunchConfiguration('confidence_threshold'),
            }],
            output='screen',
            remappings=[
                # 각 카메라별로 고유한 토픽 네임스페이스
                ('~/trash_positions', f'/position_estimator/{camera}/trash_positions'),
                ('~/results', f'/position_estimator/{camera}/results'),
            ]
        )
        nodes.append(node)
    
    return LaunchDescription(launch_args + nodes)