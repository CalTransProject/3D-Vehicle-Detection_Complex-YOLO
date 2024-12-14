"""
# -*- coding: utf-8 -*-
-----------------------------------------------------------------------------------
# Refer: https://github.com/ghimiredhikura/Complex-YOLOv3
# Source : https://github.com/jeasinema/VoxelNet-tensorflow/blob/master/utils/utils.py
"""
import sys
import math

import numpy as np
import torch

sys.path.append('../')

from config import kitti_config as cnf


def angle_in_limit(angle):
    # To limit the angle in -pi/2 - pi/2
    limit_degree = 5
    while angle >= np.pi / 2:
        angle -= np.pi
    while angle < -np.pi / 2:
        angle += np.pi
    if abs(angle + np.pi / 2) < limit_degree / 180 * np.pi:
        angle = np.pi / 2
    return angle


def camera_to_lidar(x, y, z, V2C=None, R0=None, P2=None):
    p = np.array([x, y, z, 1])
    if V2C is None or R0 is None:
        p = np.matmul(cnf.R0_inv, p)
        p = np.matmul(cnf.Tr_velo_to_cam_inv, p)
    else:
        R0_i = np.zeros((4, 4))
        R0_i[:3, :3] = R0
        R0_i[3, 3] = 1
        p = np.matmul(np.linalg.inv(R0_i), p)
        p = np.matmul(inverse_rigid_trans(V2C), p)
    p = p[0:3]
    return tuple(p)


def lidar_to_camera(x, y, z, V2C=None, R0=None, P2=None):
    p = np.array([x, y, z, 1])
    if V2C is None or R0 is None:
        p = np.matmul(cnf.Tr_velo_to_cam, p)
        p = np.matmul(cnf.R0, p)
    else:
        p = np.matmul(V2C, p)
        p = np.matmul(R0, p)
    p = p[0:3]
    return tuple(p)


def camera_to_lidar_point(points):
    # (N, 3) -> (N, 3)
    N = points.shape[0]
    points = np.hstack([points, np.ones((N, 1))]).T  # (N,4) -> (4,N)

    points = np.matmul(cnf.R0_inv, points)
    points = np.matmul(cnf.Tr_velo_to_cam_inv, points).T  # (4, N) -> (N, 4)
    points = points[:, 0:3]
    return points.reshape(-1, 3)


def lidar_to_camera_point(points, V2C=None, R0=None):
    # (N, 3) -> (N, 3)
    N = points.shape[0]
    points = np.hstack([points, np.ones((N, 1))]).T

    if V2C is None or R0 is None:
        points = np.matmul(cnf.Tr_velo_to_cam, points)
        points = np.matmul(cnf.R0, points).T
    else:
        points = np.matmul(V2C, points)
        points = np.matmul(R0, points).T
    points = points[:, 0:3]
    return points.reshape(-1, 3)


def camera_to_lidar_box(boxes, V2C=None, R0=None, P2=None):
    # (N, 7) -> (N, 7) x,y,z,h,w,l,r
    ret = []
    for box in boxes:
        x, y, z, h, w, l, ry = box
        (x, y, z), h, w, l, rz = camera_to_lidar(
            x, y, z, V2C=V2C, R0=R0, P2=P2), h, w, l, -ry - np.pi / 2
        # rz = angle_in_limit(rz)
        ret.append([x, y, z, h, w, l, rz])
    return np.array(ret).reshape(-1, 7)


def lidar_to_camera_box(boxes, V2C=None, R0=None, P2=None):
    # (N, 7) -> (N, 7) x,y,z,h,w,l,r
    ret = []
    for box in boxes:
        x, y, z, h, w, l, rz = box
        (x, y, z), h, w, l, ry = lidar_to_camera(
            x, y, z, V2C=V2C, R0=R0, P2=P2), h, w, l, -rz - np.pi / 2
        # ry = angle_in_limit(ry)
        ret.append([x, y, z, h, w, l, ry])
    return np.array(ret).reshape(-1, 7)


def center_to_corner_box2d(boxes_center, coordinate='lidar'):
    # (N, 5) -> (N, 4, 2)
    N = boxes_center.shape[0]
    boxes3d_center = np.zeros((N, 7))
    boxes3d_center[:, [0, 1, 4, 5, 6]] = boxes_center
    boxes3d_corner = center_to_corner_box3d(
        boxes3d_center, coordinate=coordinate)

    return boxes3d_corner[:, 0:4, 0:2]


def center_to_corner_box3d(boxes_center, coordinate='lidar'):
    # (N, 7) -> (N, 8, 3)
    N = boxes_center.shape[0]
    ret = np.zeros((N, 8, 3), dtype=np.float32)

    if coordinate == 'camera':
        boxes_center = camera_to_lidar_box(boxes_center)

    for i in range(N):
        box = boxes_center[i]
        translation = box[0:3]
        size = box[3:6]
        rotation = [0, 0, box[-1]]

        h, w, l = size[0], size[1], size[2]
        trackletBox = np.array([  # in velodyne coordinates around zero point and without orientation yet
            [-l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2], \
            [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2], \
            [0, 0, 0, 0, h, h, h, h]])

        # re-create 3D bounding box in velodyne coordinate system
        yaw = rotation[2]
        rotMat = np.array([
            [np.cos(yaw), -np.sin(yaw), 0.0],
            [np.sin(yaw), np.cos(yaw), 0.0],
            [0.0, 0.0, 1.0]])
        cornerPosInVelo = np.dot(rotMat, trackletBox) + \
                          np.tile(translation, (8, 1)).T
        box3d = cornerPosInVelo.transpose()
        ret[i] = box3d

    if coordinate == 'camera':
        for idx in range(len(ret)):
            ret[idx] = lidar_to_camera_point(ret[idx])

    return ret


CORNER2CENTER_AVG = True


def corner_to_center_box3d(boxes_corner, coordinate='camera'):
    # (N, 8, 3) -> (N, 7) x,y,z,h,w,l,ry/z
    if coordinate == 'lidar':
        for idx in range(len(boxes_corner)):
            boxes_corner[idx] = lidar_to_camera_point(boxes_corner[idx])

    ret = []
    for roi in boxes_corner:
        if CORNER2CENTER_AVG:  # average version
            roi = np.array(roi)
            h = abs(np.sum(roi[:4, 1] - roi[4:, 1]) / 4)
            w = np.sum(
                np.sqrt(np.sum((roi[0, [0, 2]] - roi[3, [0, 2]]) ** 2)) +
                np.sqrt(np.sum((roi[1, [0, 2]] - roi[2, [0, 2]]) ** 2)) +
                np.sqrt(np.sum((roi[4, [0, 2]] - roi[7, [0, 2]]) ** 2)) +
                np.sqrt(np.sum((roi[5, [0, 2]] - roi[6, [0, 2]]) ** 2))
            ) / 4
            l = np.sum(
                np.sqrt(np.sum((roi[0, [0, 2]] - roi[1, [0, 2]]) ** 2)) +
                np.sqrt(np.sum((roi[2, [0, 2]] - roi[3, [0, 2]]) ** 2)) +
                np.sqrt(np.sum((roi[4, [0, 2]] - roi[5, [0, 2]]) ** 2)) +
                np.sqrt(np.sum((roi[6, [0, 2]] - roi[7, [0, 2]]) ** 2))
            ) / 4
            x = np.sum(roi[:, 0], axis=0) / 8
            y = np.sum(roi[0:4, 1], axis=0) / 4
            z = np.sum(roi[:, 2], axis=0) / 8
            ry = np.sum(
                math.atan2(roi[2, 0] - roi[1, 0], roi[2, 2] - roi[1, 2]) +
                math.atan2(roi[6, 0] - roi[5, 0], roi[6, 2] - roi[5, 2]) +
                math.atan2(roi[3, 0] - roi[0, 0], roi[3, 2] - roi[0, 2]) +
                math.atan2(roi[7, 0] - roi[4, 0], roi[7, 2] - roi[4, 2]) +
                math.atan2(roi[0, 2] - roi[1, 2], roi[1, 0] - roi[0, 0]) +
                math.atan2(roi[4, 2] - roi[5, 2], roi[5, 0] - roi[4, 0]) +
                math.atan2(roi[3, 2] - roi[2, 2], roi[2, 0] - roi[3, 0]) +
                math.atan2(roi[7, 2] - roi[6, 2], roi[6, 0] - roi[7, 0])
            ) / 8
            if w > l:
                w, l = l, w
                ry = ry - np.pi / 2
            elif l > w:
                l, w = w, l
                ry = ry - np.pi / 2
            ret.append([x, y, z, h, w, l, ry])

        else:  # max version
            h = max(abs(roi[:4, 1] - roi[4:, 1]))
            w = np.max(
                np.sqrt(np.sum((roi[0, [0, 2]] - roi[3, [0, 2]]) ** 2)) +
                np.sqrt(np.sum((roi[1, [0, 2]] - roi[2, [0, 2]]) ** 2)) +
                np.sqrt(np.sum((roi[4, [0, 2]] - roi[7, [0, 2]]) ** 2)) +
                np.sqrt(np.sum((roi[5, [0, 2]] - roi[6, [0, 2]]) ** 2))
            )
            l = np.max(
                np.sqrt(np.sum((roi[0, [0, 2]] - roi[1, [0, 2]]) ** 2)) +
                np.sqrt(np.sum((roi[2, [0, 2]] - roi[3, [0, 2]]) ** 2)) +
                np.sqrt(np.sum((roi[4, [0, 2]] - roi[5, [0, 2]]) ** 2)) +
                np.sqrt(np.sum((roi[6, [0, 2]] - roi[7, [0, 2]]) ** 2))
            )
            x = np.sum(roi[:, 0], axis=0) / 8
            y = np.sum(roi[0:4, 1], axis=0) / 4
            z = np.sum(roi[:, 2], axis=0) / 8
            ry = np.sum(
                math.atan2(roi[2, 0] - roi[1, 0], roi[2, 2] - roi[1, 2]) +
                math.atan2(roi[6, 0] - roi[5, 0], roi[6, 2] - roi[5, 2]) +
                math.atan2(roi[3, 0] - roi[0, 0], roi[3, 2] - roi[0, 2]) +
                math.atan2(roi[7, 0] - roi[4, 0], roi[7, 2] - roi[4, 2]) +
                math.atan2(roi[0, 2] - roi[1, 2], roi[1, 0] - roi[0, 0]) +
                math.atan2(roi[4, 2] - roi[5, 2], roi[5, 0] - roi[4, 0]) +
                math.atan2(roi[3, 2] - roi[2, 2], roi[2, 0] - roi[3, 0]) +
                math.atan2(roi[7, 2] - roi[6, 2], roi[6, 0] - roi[7, 0])
            ) / 8
            if w > l:
                w, l = l, w
                ry = angle_in_limit(ry + np.pi / 2)
            ret.append([x, y, z, h, w, l, ry])

    if coordinate == 'lidar':
        ret = camera_to_lidar_box(np.array(ret))

    return np.array(ret)


def point_transform(points, tx, ty, tz, rx=0, ry=0, rz=0):
    # Input:
    #   points: (N, 3)
    #   rx/y/z: in radians
    # Output:
    #   points: (N, 3)
    N = points.shape[0]
    points = np.hstack([points, np.ones((N, 1))])

    mat1 = np.eye(4)
    mat1[3, 0:3] = tx, ty, tz
    points = np.matmul(points, mat1)

    if rx != 0:
        mat = np.zeros((4, 4))
        mat[0, 0] = 1
        mat[3, 3] = 1
        mat[1, 1] = np.cos(rx)
        mat[1, 2] = -np.sin(rx)
        mat[2, 1] = np.sin(rx)
        mat[2, 2] = np.cos(rx)
        points = np.matmul(points, mat)

    if ry != 0:
        mat = np.zeros((4, 4))
        mat[1, 1] = 1
        mat[3, 3] = 1
        mat[0, 0] = np.cos(ry)
        mat[0, 2] = np.sin(ry)
        mat[2, 0] = -np.sin(ry)
        mat[2, 2] = np.cos(ry)
        points = np.matmul(points, mat)

    if rz != 0:
        mat = np.zeros((4, 4))
        mat[2, 2] = 1
        mat[3, 3] = 1
        mat[0, 0] = np.cos(rz)
        mat[0, 1] = -np.sin(rz)
        mat[1, 0] = np.sin(rz)
        mat[1, 1] = np.cos(rz)
        points = np.matmul(points, mat)

    return points[:, 0:3]


def box_transform(boxes, tx, ty, tz, r=0, coordinate='lidar'):
    # Input:
    #   boxes: (N, 7) x y z h w l rz/y
    # Output:
    #   boxes: (N, 7) x y z h w l rz/y
    boxes_corner = center_to_corner_box3d(
        boxes, coordinate=coordinate)  # (N, 8, 3)
    for idx in range(len(boxes_corner)):
        if coordinate == 'lidar':
            boxes_corner[idx] = point_transform(
                boxes_corner[idx], tx, ty, tz, rz=r)
        else:
            boxes_corner[idx] = point_transform(
                boxes_corner[idx], tx, ty, tz, ry=r)

    return corner_to_center_box3d(boxes_corner, coordinate=coordinate)


def inverse_rigid_trans(Tr):
    ''' Inverse a rigid body transform matrix (3x4 as [R|t])
        [R'|-R't; 0|1]
    '''
    inv_Tr = np.zeros_like(Tr)  # 3x4
    inv_Tr[0:3, 0:3] = np.transpose(Tr[0:3, 0:3])
    inv_Tr[0:3, 3] = np.dot(-np.transpose(Tr[0:3, 0:3]), Tr[0:3, 3])
    return inv_Tr


class Compose(object):
    def __init__(self, transforms, p=1.0):
        self.transforms = transforms
        self.p = p

    def __call__(self, lidar, labels):
        if np.random.random() <= self.p:
            for t in self.transforms:
                lidar, labels = t(lidar, labels)
        return lidar, labels


class OneOf(object):
    def __init__(self, transforms, p=1.0):
        self.transforms = transforms
        self.p = p

    def __call__(self, lidar, labels):
        if np.random.random() <= self.p:
            choice = np.random.randint(low=0, high=len(self.transforms))
            lidar, labels = self.transforms[choice](lidar, labels)

        return lidar, labels


class Random_Rotation(object):
    def __init__(self, limit_angle=20., p=0.5):
        self.limit_angle = limit_angle / 180. * np.pi
        self.p = p

    def __call__(self, lidar, labels):
        """
        :param labels: # (N', 7) x, y, z, h, w, l, r
        :return:
        """
        if np.random.random() <= self.p:
            angle = np.random.uniform(-self.limit_angle, self.limit_angle)
            lidar[:, 0:3] = point_transform(lidar[:, 0:3], 0, 0, 0, rz=angle)
            labels = box_transform(labels, 0, 0, 0, r=angle, coordinate='lidar')

        return lidar, labels


class Random_Scaling(object):
    def __init__(self, scaling_range=(0.95, 1.05), p=0.5):
        self.scaling_range = scaling_range
        self.p = p

    def __call__(self, lidar, labels):
        """
        :param labels: # (N', 7) x, y, z, h, w, l, r
        :return:
        """
        if np.random.random() <= self.p:
            factor = np.random.uniform(self.scaling_range[0], self.scaling_range[0])
            lidar[:, 0:3] = lidar[:, 0:3] * factor
            labels[:, 0:6] = labels[:, 0:6] * factor

        return lidar, labels


class Horizontal_Flip(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img, targets):
        if np.random.random() <= self.p:
            img = torch.flip(img, [-1])
            targets[:, 2] = 1 - targets[:, 2]  # horizontal flip
            targets[:, 6] = - targets[:, 6]  # yaw angle flip

        return img, targets


class Cutout(object):
    """Randomly mask out one or more patches from an image.
    Args:
        n_holes (int): Number of patches to cut out of each image.
        length (int): The length (in pixels) of each square patch.
        Refer from: https://github.com/uoguelph-mlrg/Cutout/blob/master/util/cutout.py
    """

    def __init__(self, n_holes, ratio, fill_value=0., p=1.0):
        self.n_holes = n_holes
        self.ratio = ratio
        assert 0. <= fill_value <= 1., "the fill value is in a range of 0 to 1"
        self.fill_value = fill_value
        self.p = p

    def __call__(self, img, targets):
        """
        Args:
            img (Tensor): Tensor image of size (C, H, W).
        Returns:
            Tensor: Image with n_holes of dimension length x length cut out of it.
        """
        if np.random.random() <= self.p:
            h = img.size(1)
            w = img.size(2)

            h_cutout = int(self.ratio * h)
            w_cutout = int(self.ratio * w)

            for n in range(self.n_holes):
                y = np.random.randint(h)
                x = np.random.randint(w)

                y1 = np.clip(y - h_cutout // 2, 0, h)
                y2 = np.clip(y + h_cutout // 2, 0, h)
                x1 = np.clip(x - w_cutout // 2, 0, w)
                x2 = np.clip(x + w_cutout // 2, 0, w)

                img[:, y1: y2, x1: x2] = self.fill_value  # Zero out the selected area
                # Remove targets that are in the selected area
                keep_target = []
                for target_idx, target in enumerate(targets):
                    _, _, target_x, target_y, target_w, target_l, _, _ = target
                    if (x1 <= target_x * w <= x2) and (y1 <= target_y * h <= y2):
                        continue
                    keep_target.append(target_idx)
                targets = targets[keep_target]

        return img, targets


class VehicleAugmentation:
    """Enhanced augmentation specifically for vehicle detection"""
    def __init__(self, config):
        self.min_points = config.min_vehicle_points
        self.car_scales = {
            'sedan': (4.2, 1.8),   # Length, width ranges
            'suv': (4.8, 1.9),
            'truck': (6.0, 2.2)
        }
    
    def augment_vehicles(self, points, boxes, labels):
        """Vehicle-specific augmentation pipeline"""
        augmented_data = []
        for points, box, label in zip(points, boxes, labels):
            if label == 0:  # Vehicle class
                # Apply vehicle-specific augmentation
                aug_points = self._augment_vehicle_instance(points, box)
                
                # Add realistic occlusions
                aug_points = self._add_realistic_occlusions(aug_points, box)
                
                # Enhance point density for distant vehicles
                if self._is_distant_vehicle(box):
                    aug_points = self._enhance_distant_vehicle(aug_points)
                    
                augmented_data.append(aug_points)
        
        return augmented_data
    
    def _augment_vehicle_instance(self, points, box):
        """Apply vehicle-specific geometric augmentations"""
        # Random rotation within reasonable range
        angle = np.random.uniform(-np.pi/6, np.pi/6)
        points = self._rotate_points(points, angle)
        
        # Scale augmentation based on vehicle type
        scale = np.random.uniform(0.95, 1.05)
        points = self._scale_points(points, scale)
        
        return points
    
    def _add_realistic_occlusions(self, points, box):
        """Add realistic occlusion patterns"""
        # Simulate common occlusion patterns
        occlusion_patterns = [
            'front', 'rear', 'side',
            'partial_top', 'partial_bottom'
        ]
        pattern = np.random.choice(occlusion_patterns)
        return self._apply_occlusion_pattern(points, box, pattern)
    
    def _enhance_distant_vehicle(self, points):
        """Enhance point cloud density for distant vehicles"""
        if len(points) < self.min_points:
            # Interpolate additional points
            new_points = self._interpolate_vehicle_points(points)
            points = np.concatenate([points, new_points], axis=0)
        return points
    
    def _rotate_points(self, points, angle):
        # Rotate points around z-axis
        rotation_matrix = np.array([
            [np.cos(angle), -np.sin(angle), 0],
            [np.sin(angle), np.cos(angle), 0],
            [0, 0, 1]
        ])
        return np.dot(points, rotation_matrix)
    
    def _scale_points(self, points, scale):
        # Scale points uniformly
        return points * scale
    
    def _is_distant_vehicle(self, box):
        # Check if vehicle is distant based on depth
        return box[2] > 20
    
    def _apply_occlusion_pattern(self, points, box, pattern):
        # Apply occlusion pattern to points
        if pattern == 'front':
            # Occlude front part of vehicle
            points = points[points[:, 0] > box[0] - box[3]]
        elif pattern == 'rear':
            # Occlude rear part of vehicle
            points = points[points[:, 0] < box[0] + box[3]]
        elif pattern == 'side':
            # Occlude side part of vehicle
            points = points[points[:, 1] > box[1] - box[4]]
        elif pattern == 'partial_top':
            # Occlude partial top part of vehicle
            points = points[points[:, 2] > box[2] - box[5]]
        elif pattern == 'partial_bottom':
            # Occlude partial bottom part of vehicle
            points = points[points[:, 2] < box[2] + box[5]]
        return points
    
    def _interpolate_vehicle_points(self, points):
        # Interpolate additional points for distant vehicles
        new_points = []
        for i in range(len(points)):
            for j in range(i + 1, len(points)):
                # Interpolate points along line segment
                interpolated_points = self._interpolate_line_segment(points[i], points[j])
                new_points.extend(interpolated_points)
        return np.array(new_points)
    
    def _interpolate_line_segment(self, point1, point2):
        # Interpolate points along line segment
        interpolated_points = []
        for t in np.linspace(0, 1, 10):
            interpolated_point = point1 * (1 - t) + point2 * t
            interpolated_points.append(interpolated_point)
        return interpolated_points
