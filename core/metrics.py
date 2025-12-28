import numpy as np
from dataclasses import dataclass, field, asdict
from typing import Dict, Optional, List
from .pose import Keypoints


@dataclass
class BiomechanicsMetrics:
    """Container for all biomechanical metrics"""
    timestamp: float = 0.0
    frame_number: int = 0
    person_id: str = ""
    
    # Thigh angles from vertical (degrees)
    l_thigh_angle: Optional[float] = None
    r_thigh_angle: Optional[float] = None
    
    # Shank angles from vertical (degrees)
    l_shank_angle: Optional[float] = None
    r_shank_angle: Optional[float] = None
    
    # Hip-ankle distances (pixels or meters if calibrated)
    l_hip_ankle: Optional[float] = None
    r_hip_ankle: Optional[float] = None
    
    # Joint angles (degrees)
    l_knee_angle: Optional[float] = None
    r_knee_angle: Optional[float] = None
    l_hip_angle: Optional[float] = None
    r_hip_angle: Optional[float] = None
    l_ankle_angle: Optional[float] = None
    r_ankle_angle: Optional[float] = None
    l_elbow_angle: Optional[float] = None
    r_elbow_angle: Optional[float] = None
    
    # Knee valgus/varus (requires 3D, degrees)
    l_knee_valgus: Optional[float] = None
    r_knee_valgus: Optional[float] = None
    
    # Trunk metrics
    trunk_lean: Optional[float] = None
    trunk_rotation: Optional[float] = None
    
    # Segment lengths (for scaling validation)
    shoulder_width: Optional[float] = None
    hip_width: Optional[float] = None
    torso_length: Optional[float] = None
    l_thigh_length: Optional[float] = None
    r_thigh_length: Optional[float] = None
    l_shank_length: Optional[float] = None
    r_shank_length: Optional[float] = None
    
    # Center of mass estimate (x, y)
    com_x: Optional[float] = None
    com_y: Optional[float] = None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary, excluding None values"""
        return {k: v for k, v in asdict(self).items() if v is not None}
    
    def to_flat_dict(self) -> Dict:
        """Convert to flat dictionary with all fields"""
        return asdict(self)


class MetricsCalculator:
    """Calculate biomechanical metrics from pose keypoints"""
    
    def __init__(self, confidence_threshold: float = 0.5, px_to_m: Optional[float] = None):
        self.conf_thresh = confidence_threshold
        self.px_to_m = px_to_m  # Pixels to meters conversion (if calibrated)
    
    def _angle_from_vertical(self, p1: np.ndarray, p2: np.ndarray) -> float:
        """
        Calculate angle of segment from vertical (degrees).
        0° = straight down, positive = forward, negative = backward
        """
        vec = p2 - p1
        vertical = np.array([0, 1])  # Y-down in image coordinates
        
        cos_angle = np.dot(vec, vertical) / (np.linalg.norm(vec) * np.linalg.norm(vertical) + 1e-6)
        angle = np.degrees(np.arccos(np.clip(cos_angle, -1, 1)))
        
        # Sign: positive if p2 is in front of p1 (negative x direction in image)
        if vec[0] < 0:
            angle = -angle
        
        return angle
    
    def _joint_angle(self, p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> float:
        """Calculate angle at p2 formed by p1-p2-p3"""
        v1 = p1 - p2
        v2 = p3 - p2
        
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
        return np.degrees(np.arccos(np.clip(cos_angle, -1, 1)))
    
    def _distance(self, p1: np.ndarray, p2: np.ndarray) -> float:
        """Calculate distance between two points"""
        dist = np.linalg.norm(p1 - p2)
        if self.px_to_m:
            dist *= self.px_to_m
        return dist
    
    def _estimate_com(self, kpts: Keypoints) -> Optional[tuple]:
        """
        Estimate center of mass using segment method.
        Simplified version using major body segments.
        """
        # Segment weights (percentage of body mass)
        weights = {
            'head': 0.08,
            'trunk': 0.50,
            'upper_arm': 0.03,
            'forearm': 0.02,
            'thigh': 0.10,
            'shank': 0.05,
        }
        
        total_weight = 0
        com = np.array([0.0, 0.0])
        
        # Head (use nose as approximation)
        if kpts.is_valid(Keypoints.NOSE, self.conf_thresh):
            com += weights['head'] * kpts.xy[Keypoints.NOSE]
            total_weight += weights['head']
        
        # Trunk (midpoint of shoulders to midpoint of hips)
        if all(kpts.is_valid(i, self.conf_thresh) for i in 
               [Keypoints.L_SHOULDER, Keypoints.R_SHOULDER, Keypoints.L_HIP, Keypoints.R_HIP]):
            mid_shoulder = (kpts.xy[Keypoints.L_SHOULDER] + kpts.xy[Keypoints.R_SHOULDER]) / 2
            mid_hip = (kpts.xy[Keypoints.L_HIP] + kpts.xy[Keypoints.R_HIP]) / 2
            trunk_com = (mid_shoulder + mid_hip) / 2
            com += weights['trunk'] * trunk_com
            total_weight += weights['trunk']
        
        # Thighs
        for hip, knee, w in [(Keypoints.L_HIP, Keypoints.L_KNEE, 'thigh'),
                              (Keypoints.R_HIP, Keypoints.R_KNEE, 'thigh')]:
            if kpts.is_valid(hip, self.conf_thresh) and kpts.is_valid(knee, self.conf_thresh):
                segment_com = (kpts.xy[hip] + kpts.xy[knee]) / 2
                com += weights[w] * segment_com
                total_weight += weights[w]
        
        # Shanks
        for knee, ankle, w in [(Keypoints.L_KNEE, Keypoints.L_ANKLE, 'shank'),
                                (Keypoints.R_KNEE, Keypoints.R_ANKLE, 'shank')]:
            if kpts.is_valid(knee, self.conf_thresh) and kpts.is_valid(ankle, self.conf_thresh):
                segment_com = (kpts.xy[knee] + kpts.xy[ankle]) / 2
                com += weights[w] * segment_com
                total_weight += weights[w]
        
        if total_weight > 0:
            return tuple(com / total_weight)
        return None
    
    def calculate(self, kpts: Keypoints, timestamp: float = 0.0, 
                  frame_number: int = 0, person_id: str = "") -> BiomechanicsMetrics:
        """Calculate all metrics from keypoints"""
        metrics = BiomechanicsMetrics(
            timestamp=timestamp,
            frame_number=frame_number,
            person_id=person_id
        )
        
        # --- Thigh angles ---
        if kpts.is_valid(Keypoints.L_HIP) and kpts.is_valid(Keypoints.L_KNEE):
            metrics.l_thigh_angle = self._angle_from_vertical(
                kpts.xy[Keypoints.L_HIP], kpts.xy[Keypoints.L_KNEE]
            )
        
        if kpts.is_valid(Keypoints.R_HIP) and kpts.is_valid(Keypoints.R_KNEE):
            metrics.r_thigh_angle = self._angle_from_vertical(
                kpts.xy[Keypoints.R_HIP], kpts.xy[Keypoints.R_KNEE]
            )
        
        # --- Shank angles ---
        if kpts.is_valid(Keypoints.L_KNEE) and kpts.is_valid(Keypoints.L_ANKLE):
            metrics.l_shank_angle = self._angle_from_vertical(
                kpts.xy[Keypoints.L_KNEE], kpts.xy[Keypoints.L_ANKLE]
            )
        
        if kpts.is_valid(Keypoints.R_KNEE) and kpts.is_valid(Keypoints.R_ANKLE):
            metrics.r_shank_angle = self._angle_from_vertical(
                kpts.xy[Keypoints.R_KNEE], kpts.xy[Keypoints.R_ANKLE]
            )
        
        # --- Hip-ankle distances ---
        if kpts.is_valid(Keypoints.L_HIP) and kpts.is_valid(Keypoints.L_ANKLE):
            metrics.l_hip_ankle = self._distance(
                kpts.xy[Keypoints.L_HIP], kpts.xy[Keypoints.L_ANKLE]
            )
        
        if kpts.is_valid(Keypoints.R_HIP) and kpts.is_valid(Keypoints.R_ANKLE):
            metrics.r_hip_ankle = self._distance(
                kpts.xy[Keypoints.R_HIP], kpts.xy[Keypoints.R_ANKLE]
            )
        
        # --- Knee angles ---
        if all(kpts.is_valid(i) for i in [Keypoints.L_HIP, Keypoints.L_KNEE, Keypoints.L_ANKLE]):
            metrics.l_knee_angle = self._joint_angle(
                kpts.xy[Keypoints.L_HIP],
                kpts.xy[Keypoints.L_KNEE],
                kpts.xy[Keypoints.L_ANKLE]
            )
        
        if all(kpts.is_valid(i) for i in [Keypoints.R_HIP, Keypoints.R_KNEE, Keypoints.R_ANKLE]):
            metrics.r_knee_angle = self._joint_angle(
                kpts.xy[Keypoints.R_HIP],
                kpts.xy[Keypoints.R_KNEE],
                kpts.xy[Keypoints.R_ANKLE]
            )
        
        # --- Hip angles ---
        if all(kpts.is_valid(i) for i in [Keypoints.L_SHOULDER, Keypoints.L_HIP, Keypoints.L_KNEE]):
            metrics.l_hip_angle = self._joint_angle(
                kpts.xy[Keypoints.L_SHOULDER],
                kpts.xy[Keypoints.L_HIP],
                kpts.xy[Keypoints.L_KNEE]
            )
        
        if all(kpts.is_valid(i) for i in [Keypoints.R_SHOULDER, Keypoints.R_HIP, Keypoints.R_KNEE]):
            metrics.r_hip_angle = self._joint_angle(
                kpts.xy[Keypoints.R_SHOULDER],
                kpts.xy[Keypoints.R_HIP],
                kpts.xy[Keypoints.R_KNEE]
            )
        
        # --- Elbow angles ---
        if all(kpts.is_valid(i) for i in [Keypoints.L_SHOULDER, Keypoints.L_ELBOW, Keypoints.L_WRIST]):
            metrics.l_elbow_angle = self._joint_angle(
                kpts.xy[Keypoints.L_SHOULDER],
                kpts.xy[Keypoints.L_ELBOW],
                kpts.xy[Keypoints.L_WRIST]
            )
        
        if all(kpts.is_valid(i) for i in [Keypoints.R_SHOULDER, Keypoints.R_ELBOW, Keypoints.R_WRIST]):
            metrics.r_elbow_angle = self._joint_angle(
                kpts.xy[Keypoints.R_SHOULDER],
                kpts.xy[Keypoints.R_ELBOW],
                kpts.xy[Keypoints.R_WRIST]
            )
        
        # --- Trunk lean ---
        if all(kpts.is_valid(i) for i in [Keypoints.L_SHOULDER, Keypoints.R_SHOULDER, 
                                           Keypoints.L_HIP, Keypoints.R_HIP]):
            mid_shoulder = (kpts.xy[Keypoints.L_SHOULDER] + kpts.xy[Keypoints.R_SHOULDER]) / 2
            mid_hip = (kpts.xy[Keypoints.L_HIP] + kpts.xy[Keypoints.R_HIP]) / 2
            metrics.trunk_lean = self._angle_from_vertical(mid_hip, mid_shoulder)
        
        # --- Segment lengths ---
        if kpts.is_valid(Keypoints.L_SHOULDER) and kpts.is_valid(Keypoints.R_SHOULDER):
            metrics.shoulder_width = self._distance(
                kpts.xy[Keypoints.L_SHOULDER], kpts.xy[Keypoints.R_SHOULDER]
            )
        
        if kpts.is_valid(Keypoints.L_HIP) and kpts.is_valid(Keypoints.R_HIP):
            metrics.hip_width = self._distance(
                kpts.xy[Keypoints.L_HIP], kpts.xy[Keypoints.R_HIP]
            )
        
        if kpts.is_valid(Keypoints.L_HIP) and kpts.is_valid(Keypoints.L_KNEE):
            metrics.l_thigh_length = self._distance(
                kpts.xy[Keypoints.L_HIP], kpts.xy[Keypoints.L_KNEE]
            )
        
        if kpts.is_valid(Keypoints.R_HIP) and kpts.is_valid(Keypoints.R_KNEE):
            metrics.r_thigh_length = self._distance(
                kpts.xy[Keypoints.R_HIP], kpts.xy[Keypoints.R_KNEE]
            )
        
        if kpts.is_valid(Keypoints.L_KNEE) and kpts.is_valid(Keypoints.L_ANKLE):
            metrics.l_shank_length = self._distance(
                kpts.xy[Keypoints.L_KNEE], kpts.xy[Keypoints.L_ANKLE]
            )
        
        if kpts.is_valid(Keypoints.R_KNEE) and kpts.is_valid(Keypoints.R_ANKLE):
            metrics.r_shank_length = self._distance(
                kpts.xy[Keypoints.R_KNEE], kpts.xy[Keypoints.R_ANKLE]
            )
        
        # --- Center of mass ---
        com = self._estimate_com(kpts)
        if com:
            metrics.com_x, metrics.com_y = com
        
        return metrics
    
    def calculate_batch(self, keypoints_list: List[Keypoints], 
                        timestamp: float = 0.0, frame_number: int = 0) -> List[BiomechanicsMetrics]:
        """Calculate metrics for multiple people"""
        return [
            self.calculate(kpts, timestamp, frame_number, f"person_{i}")
            for i, kpts in enumerate(keypoints_list)
        ]
