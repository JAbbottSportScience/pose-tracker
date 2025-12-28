import cv2
import numpy as np
import json
from typing import Tuple, Optional
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Points3D:
    """Container for 3D keypoints"""
    xyz: np.ndarray  # (17, 3) in meters
    valid: np.ndarray  # (17,) bool
    
    def get_valid_points(self) -> np.ndarray:
        """Return only valid 3D points"""
        return self.xyz[self.valid]
    
    def get_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return min and max bounds of valid points"""
        valid_pts = self.get_valid_points()
        if len(valid_pts) == 0:
            return np.zeros(3), np.zeros(3)
        return valid_pts.min(axis=0), valid_pts.max(axis=0)


class StereoTriangulator:
    """Stereo camera triangulation for 3D pose reconstruction"""
    
    def __init__(self, calibration_file: str):
        """Load stereo calibration from JSON file"""
        with open(calibration_file, 'r') as f:
            calib = json.load(f)
        
        self.mtx1 = np.array(calib['mtx1'])
        self.mtx2 = np.array(calib['mtx2'])
        self.dist1 = np.array(calib['dist1'])
        self.dist2 = np.array(calib['dist2'])
        self.P1 = np.array(calib['P1'])
        self.P2 = np.array(calib['P2'])
        self.R1 = np.array(calib['R1'])
        self.R2 = np.array(calib['R2'])
        self.img_size = tuple(calib['img_size'])
        
        # Store original calibration data
        self.R = np.array(calib.get('R', np.eye(3)))
        self.T = np.array(calib.get('T', np.zeros((3, 1))))
        
        # Compute undistortion maps
        self.map1x, self.map1y = cv2.initUndistortRectifyMap(
            self.mtx1, self.dist1, self.R1, self.P1, self.img_size, cv2.CV_32FC1
        )
        self.map2x, self.map2y = cv2.initUndistortRectifyMap(
            self.mtx2, self.dist2, self.R2, self.P2, self.img_size, cv2.CV_32FC1
        )
    
    def undistort(self, frame1: np.ndarray, frame2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Undistort and rectify both frames"""
        rect1 = cv2.remap(frame1, self.map1x, self.map1y, cv2.INTER_LINEAR)
        rect2 = cv2.remap(frame2, self.map2x, self.map2y, cv2.INTER_LINEAR)
        return rect1, rect2
    
    def triangulate(self, pts1: np.ndarray, pts2: np.ndarray, 
                    conf1: np.ndarray, conf2: np.ndarray,
                    conf_thresh: float = 0.5) -> Points3D:
        """
        Triangulate 3D points from matched 2D keypoints.
        
        Args:
            pts1: (17, 2) keypoints from camera 1
            pts2: (17, 2) keypoints from camera 2
            conf1: (17,) confidences from camera 1
            conf2: (17,) confidences from camera 2
            conf_thresh: Minimum confidence threshold
            
        Returns:
            Points3D with xyz coordinates in meters
        """
        num_points = len(pts1)
        points_3d = np.zeros((num_points, 3))
        valid = np.zeros(num_points, dtype=bool)
        
        for i in range(num_points):
            if conf1[i] > conf_thresh and conf2[i] > conf_thresh:
                pt1 = pts1[i].reshape(2, 1).astype(np.float64)
                pt2 = pts2[i].reshape(2, 1).astype(np.float64)
                
                point_4d = cv2.triangulatePoints(self.P1, self.P2, pt1, pt2)
                point_3d = (point_4d[:3] / point_4d[3]).flatten()
                
                points_3d[i] = point_3d
                valid[i] = True
        
        return Points3D(xyz=points_3d, valid=valid)
    
    def compute_reprojection_error(self, pts1: np.ndarray, pts2: np.ndarray,
                                    points_3d: Points3D) -> Tuple[float, float]:
        """Compute reprojection error for validation"""
        errors1, errors2 = [], []
        
        for i in range(len(pts1)):
            if not points_3d.valid[i]:
                continue
            
            pt_3d = points_3d.xyz[i].reshape(3, 1)
            
            # Reproject to camera 1
            pt1_reproj = self.P1 @ np.vstack([pt_3d, [[1]]])
            pt1_reproj = pt1_reproj[:2] / pt1_reproj[2]
            errors1.append(np.linalg.norm(pts1[i] - pt1_reproj.flatten()))
            
            # Reproject to camera 2
            pt2_reproj = self.P2 @ np.vstack([pt_3d, [[1]]])
            pt2_reproj = pt2_reproj[:2] / pt2_reproj[2]
            errors2.append(np.linalg.norm(pts2[i] - pt2_reproj.flatten()))
        
        if errors1 and errors2:
            return np.mean(errors1), np.mean(errors2)
        return float('inf'), float('inf')
    
    @staticmethod
    def create_calibration(images_cam1: list, images_cam2: list,
                           checkerboard: Tuple[int, int] = (9, 6),
                           square_size: float = 0.025,
                           output_path: str = 'stereo_calibration.json') -> dict:
        """
        Create stereo calibration from checkerboard images.
        
        Args:
            images_cam1: List of image paths from camera 1
            images_cam2: List of image paths from camera 2
            checkerboard: Inner corners (columns, rows)
            square_size: Size of squares in meters
            output_path: Where to save calibration JSON
            
        Returns:
            Calibration dictionary
        """
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        
        # Prepare object points
        objp = np.zeros((checkerboard[0] * checkerboard[1], 3), np.float32)
        objp[:, :2] = np.mgrid[0:checkerboard[0], 0:checkerboard[1]].T.reshape(-1, 2)
        objp *= square_size
        
        obj_points = []
        img_points_1 = []
        img_points_2 = []
        img_size = None
        
        for img1_path, img2_path in zip(images_cam1, images_cam2):
            img1 = cv2.imread(str(img1_path))
            img2 = cv2.imread(str(img2_path))
            
            if img_size is None:
                img_size = (img1.shape[1], img1.shape[0])
            
            gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
            gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
            
            found1, corners1 = cv2.findChessboardCorners(gray1, checkerboard, None)
            found2, corners2 = cv2.findChessboardCorners(gray2, checkerboard, None)
            
            if found1 and found2:
                corners1 = cv2.cornerSubPix(gray1, corners1, (11, 11), (-1, -1), criteria)
                corners2 = cv2.cornerSubPix(gray2, corners2, (11, 11), (-1, -1), criteria)
                
                obj_points.append(objp)
                img_points_1.append(corners1)
                img_points_2.append(corners2)
        
        if len(obj_points) < 10:
            raise ValueError(f"Only {len(obj_points)} valid image pairs found. Need at least 10.")
        
        # Individual camera calibration
        ret1, mtx1, dist1, _, _ = cv2.calibrateCamera(obj_points, img_points_1, img_size, None, None)
        ret2, mtx2, dist2, _, _ = cv2.calibrateCamera(obj_points, img_points_2, img_size, None, None)
        
        # Stereo calibration
        flags = cv2.CALIB_FIX_INTRINSIC
        ret_stereo, mtx1, dist1, mtx2, dist2, R, T, E, F = cv2.stereoCalibrate(
            obj_points, img_points_1, img_points_2,
            mtx1, dist1, mtx2, dist2,
            img_size, criteria=criteria, flags=flags
        )
        
        # Rectification
        R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
            mtx1, dist1, mtx2, dist2, img_size, R, T, alpha=0
        )
        
        calibration = {
            'img_size': list(img_size),
            'mtx1': mtx1.tolist(),
            'dist1': dist1.tolist(),
            'mtx2': mtx2.tolist(),
            'dist2': dist2.tolist(),
            'R': R.tolist(),
            'T': T.tolist(),
            'R1': R1.tolist(),
            'R2': R2.tolist(),
            'P1': P1.tolist(),
            'P2': P2.tolist(),
            'Q': Q.tolist(),
            'stereo_rms': ret_stereo
        }
        
        with open(output_path, 'w') as f:
            json.dump(calibration, f, indent=2)
        
        return calibration


class Metrics3DCalculator:
    """Calculate 3D biomechanical metrics from triangulated points"""
    
    # Keypoint indices
    L_HIP, R_HIP = 11, 12
    L_KNEE, R_KNEE = 13, 14
    L_ANKLE, R_ANKLE = 15, 16
    L_SHOULDER, R_SHOULDER = 5, 6
    
    def __init__(self):
        # Vertical in world coordinates (Y-down typically)
        self.vertical = np.array([0, 1, 0])
    
    def calculate(self, points_3d: Points3D) -> dict:
        """Calculate 3D metrics from triangulated keypoints"""
        metrics = {}
        xyz = points_3d.xyz
        valid = points_3d.valid
        
        # 3D Thigh angles from vertical
        if valid[self.L_HIP] and valid[self.L_KNEE]:
            thigh_vec = xyz[self.L_KNEE] - xyz[self.L_HIP]
            thigh_vec_norm = thigh_vec / (np.linalg.norm(thigh_vec) + 1e-6)
            cos_angle = np.dot(thigh_vec_norm, self.vertical)
            metrics['l_thigh_angle_3d'] = np.degrees(np.arccos(np.clip(cos_angle, -1, 1)))
        
        if valid[self.R_HIP] and valid[self.R_KNEE]:
            thigh_vec = xyz[self.R_KNEE] - xyz[self.R_HIP]
            thigh_vec_norm = thigh_vec / (np.linalg.norm(thigh_vec) + 1e-6)
            cos_angle = np.dot(thigh_vec_norm, self.vertical)
            metrics['r_thigh_angle_3d'] = np.degrees(np.arccos(np.clip(cos_angle, -1, 1)))
        
        # Hip-ankle distances in meters
        if valid[self.L_HIP] and valid[self.L_ANKLE]:
            metrics['l_hip_ankle_m'] = np.linalg.norm(xyz[self.L_HIP] - xyz[self.L_ANKLE])
        
        if valid[self.R_HIP] and valid[self.R_ANKLE]:
            metrics['r_hip_ankle_m'] = np.linalg.norm(xyz[self.R_HIP] - xyz[self.R_ANKLE])
        
        # Knee valgus/varus (frontal plane angle)
        if valid[self.L_HIP] and valid[self.L_KNEE] and valid[self.L_ANKLE]:
            metrics['l_knee_valgus'] = self._compute_valgus(
                xyz[self.L_HIP], xyz[self.L_KNEE], xyz[self.L_ANKLE]
            )
        
        if valid[self.R_HIP] and valid[self.R_KNEE] and valid[self.R_ANKLE]:
            metrics['r_knee_valgus'] = -self._compute_valgus(
                xyz[self.R_HIP], xyz[self.R_KNEE], xyz[self.R_ANKLE]
            )
        
        # Shoulder and hip widths
        if valid[self.L_SHOULDER] and valid[self.R_SHOULDER]:
            metrics['shoulder_width_m'] = np.linalg.norm(xyz[self.L_SHOULDER] - xyz[self.R_SHOULDER])
        
        if valid[self.L_HIP] and valid[self.R_HIP]:
            metrics['hip_width_m'] = np.linalg.norm(xyz[self.L_HIP] - xyz[self.R_HIP])
        
        return metrics
    
    def _compute_valgus(self, hip: np.ndarray, knee: np.ndarray, ankle: np.ndarray) -> float:
        """Compute knee valgus/varus angle in frontal plane"""
        hip_knee = knee - hip
        knee_ankle = ankle - knee
        
        # Project onto frontal plane (XY)
        hip_knee_frontal = hip_knee[[0, 1]]
        knee_ankle_frontal = knee_ankle[[0, 1]]
        
        cross = np.cross(
            np.append(hip_knee_frontal, 0),
            np.append(knee_ankle_frontal, 0)
        )[2]
        
        dot = np.dot(hip_knee_frontal, knee_ankle_frontal)
        angle = np.degrees(np.arctan2(abs(cross), dot))
        
        return angle if cross > 0 else -angle
