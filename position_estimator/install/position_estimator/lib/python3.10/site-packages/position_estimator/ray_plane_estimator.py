#!/usr/bin/env python3
import numpy as np
import math

class RayPlaneEstimator:
    """Geometric depth estimation via Ray-Plane calculation"""
    
    def __init__(self, camera_params):
        self.fx = camera_params['fx']
        self.fy = camera_params['fy']
        self.cx = camera_params.get('cx', camera_params['image_width'] / 2)
        self.cy = camera_params.get('cy', camera_params['image_height'] / 2)
        self.image_width = camera_params['image_width']
        self.image_height = camera_params['image_height']
        self.camera_position = np.array(camera_params['position'])
        self.camera_rotation = np.array(camera_params['rotation'])
        self.camera_height = camera_params['camera_height']
        self.ground_height = camera_params['ground_height']
        
        # Calculate rotation matrix
        self.R = self._rotation_matrix_xyz(*self.camera_rotation)

    def _rotation_matrix_xyz(self, rx_deg, ry_deg, rz_deg):
        """Calculate Camera TF matrix from Euler angles"""
        rx = math.radians(rx_deg)
        ry = math.radians(ry_deg)
        rz = math.radians(rz_deg)
        
        Rx = np.array([
            [1, 0, 0],
            [0, math.cos(rx), -math.sin(rx)],
            [0, math.sin(rx), math.cos(rx)]
        ])
        
        Ry = np.array([
            [math.cos(ry), 0, math.sin(ry)],
            [0, 1, 0],
            [-math.sin(ry), 0, math.cos(ry)]
        ])
        
        Rz = np.array([
            [math.cos(rz), -math.sin(rz), 0],
            [math.sin(rz), math.cos(rz), 0],
            [0, 0, 1]
        ])
        
        return Rz @ Ry @ Rx

    def estimate_position(self, center_x, center_y):
        """Estimate 3D position from pixel coordinates"""
        
        # 1. Transform pixel to normalized coordinates
        x_norm = (center_x - self.cx) / self.fx
        y_norm = (center_y - self.cy) / self.fy
        
        # 2. Create camera ray in camera coordinate system
        camera_ray = np.array([x_norm, y_norm, 1.0])
        camera_ray = camera_ray / np.linalg.norm(camera_ray)
        
        # 3. Transform ray to world coordinate system
        world_ray = self.R @ camera_ray
        world_ray = world_ray / np.linalg.norm(world_ray)
        
        # 4. Find intersection with ground plane
        if abs(world_ray[2]) < 1e-10:
            return None  # Ray is parallel to ground
        
        t = (self.ground_height - self.camera_position[2]) / world_ray[2]
        if t < 0:
            return None  # Intersection is behind camera
            
        intersection_point = self.camera_position + t * world_ray
        distance_3d = np.linalg.norm(intersection_point - self.camera_position)
        
        return {
            'position': intersection_point,
            'distance': distance_3d,
            'pixel': [center_x, center_y]
        }