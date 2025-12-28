import numpy as np
from ultralytics import YOLO
from dataclasses import dataclass, field
from typing import Optional, List, Tuple
import cv2


@dataclass
class Keypoints:
    """Container for pose keypoints with utilities"""
    xy: np.ndarray  # (17, 2)
    confidence: np.ndarray  # (17,)
    
    NAMES = [
        'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
        'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
        'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
        'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
    ]
    
    SKELETON = [
        (0, 1), (0, 2), (1, 3), (2, 4),           # Head
        (5, 6),                                    # Shoulders
        (5, 7), (7, 9),                           # Left arm
        (6, 8), (8, 10),                          # Right arm
        (5, 11), (6, 12), (11, 12),               # Torso
        (11, 13), (13, 15),                       # Left leg
        (12, 14), (14, 16)                        # Right leg
    ]
    
    # Keypoint indices as class attributes
    NOSE = 0
    L_EYE, R_EYE = 1, 2
    L_EAR, R_EAR = 3, 4
    L_SHOULDER, R_SHOULDER = 5, 6
    L_ELBOW, R_ELBOW = 7, 8
    L_WRIST, R_WRIST = 9, 10
    L_HIP, R_HIP = 11, 12
    L_KNEE, R_KNEE = 13, 14
    L_ANKLE, R_ANKLE = 15, 16
    
    def get(self, name: str) -> Tuple[np.ndarray, float]:
        """Get keypoint by name"""
        idx = self.NAMES.index(name)
        return self.xy[idx], self.confidence[idx]
    
    def is_valid(self, idx: int, threshold: float = 0.5) -> bool:
        """Check if keypoint confidence is above threshold"""
        return self.confidence[idx] > threshold
    
    def get_center(self) -> np.ndarray:
        """Get center point of valid keypoints"""
        valid_mask = self.confidence > 0.5
        if valid_mask.any():
            return self.xy[valid_mask].mean(axis=0)
        return np.array([0, 0])
    
    def get_bbox(self, padding: float = 0.1) -> Tuple[int, int, int, int]:
        """Get bounding box of valid keypoints with padding"""
        valid_mask = self.confidence > 0.5
        if not valid_mask.any():
            return (0, 0, 0, 0)
        
        valid_pts = self.xy[valid_mask]
        x_min, y_min = valid_pts.min(axis=0)
        x_max, y_max = valid_pts.max(axis=0)
        
        w, h = x_max - x_min, y_max - y_min
        pad_x, pad_y = w * padding, h * padding
        
        return (
            int(x_min - pad_x),
            int(y_min - pad_y),
            int(x_max + pad_x),
            int(y_max + pad_y)
        )


@dataclass
class PoseResult:
    """Container for pose estimation results"""
    keypoints: List[Keypoints]
    boxes: np.ndarray  # (N, 4) bounding boxes
    inference_time: float
    
    @property
    def num_people(self) -> int:
        return len(self.keypoints)


class PoseEstimator:
    """YOLO-based pose estimation"""
    
    def __init__(self, model_path: str = 'yolo11n-pose.pt', 
                 device: str = 'mps', confidence: float = 0.5):
        self.model = YOLO(model_path)
        self.device = device
        self.confidence = confidence
        
        # Move to device
        if device != 'cpu':
            try:
                self.model.to(device)
            except Exception as e:
                print(f"Warning: Could not move model to {device}: {e}")
                print("Falling back to CPU")
                self.device = 'cpu'
    
    def process(self, frame: np.ndarray, imgsz: int = 640) -> PoseResult:
        """Run pose estimation on frame"""
        results = self.model(frame, imgsz=imgsz, verbose=False, conf=self.confidence)
        
        keypoints_list = []
        boxes = []
        inference_time = results[0].speed['inference'] if results else 0
        
        for result in results:
            if result.keypoints is not None:
                kpts_xy = result.keypoints.xy.cpu().numpy()
                kpts_conf = result.keypoints.conf.cpu().numpy()
                
                for i in range(len(kpts_xy)):
                    keypoints_list.append(Keypoints(
                        xy=kpts_xy[i],
                        confidence=kpts_conf[i]
                    ))
                
                if result.boxes is not None:
                    boxes = result.boxes.xyxy.cpu().numpy()
        
        return PoseResult(
            keypoints=keypoints_list,
            boxes=np.array(boxes) if len(boxes) > 0 else np.array([]),
            inference_time=inference_time
        )
    
    def draw(self, frame: np.ndarray, result: PoseResult, 
             show_skeleton: bool = True, show_keypoints: bool = True,
             keypoint_radius: int = 5, line_thickness: int = 2,
             person_ids: Optional[List[str]] = None) -> np.ndarray:
        """Draw pose on frame with optional person labels"""
        output = frame.copy()
        
        # Color scheme for left/right distinction
        colors = {
            'left': (0, 255, 0),      # Green
            'right': (0, 0, 255),     # Red
            'center': (255, 255, 0),  # Cyan
            'skeleton': (255, 128, 0) # Orange
        }
        
        for idx, kpts in enumerate(result.keypoints):
            # Draw keypoints
            if show_keypoints:
                for i, (pt, conf) in enumerate(zip(kpts.xy, kpts.confidence)):
                    if conf > self.confidence:
                        name = kpts.NAMES[i]
                        if 'left' in name:
                            color = colors['left']
                        elif 'right' in name:
                            color = colors['right']
                        else:
                            color = colors['center']
                        
                        cv2.circle(output, (int(pt[0]), int(pt[1])), 
                                   keypoint_radius, color, -1)
            
            # Draw skeleton
            if show_skeleton:
                for start, end in kpts.SKELETON:
                    if kpts.confidence[start] > self.confidence and \
                       kpts.confidence[end] > self.confidence:
                        pt1 = tuple(map(int, kpts.xy[start]))
                        pt2 = tuple(map(int, kpts.xy[end]))
                        cv2.line(output, pt1, pt2, colors['skeleton'], line_thickness)
            
            # Draw person ID if provided
            if person_ids and idx < len(person_ids):
                center = kpts.get_center()
                bbox = kpts.get_bbox()
                label_pos = (bbox[0], bbox[1] - 10)
                cv2.putText(output, person_ids[idx], label_pos,
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return output
    
    def draw_with_metrics(self, frame: np.ndarray, result: PoseResult,
                          metrics_list: List[dict]) -> np.ndarray:
        """Draw pose with inline metrics display"""
        output = self.draw(frame, result)
        
        for kpts, metrics in zip(result.keypoints, metrics_list):
            # Draw thigh angle visualization
            if 'l_thigh_angle' in metrics and kpts.is_valid(Keypoints.L_HIP):
                hip = kpts.xy[Keypoints.L_HIP]
                angle = metrics['l_thigh_angle']
                color = (0, 255, 0) if abs(angle) < 15 else (0, 255, 255) if abs(angle) < 30 else (0, 0, 255)
                cv2.putText(output, f"L:{angle:+.0f}°", 
                           (int(hip[0]) - 60, int(hip[1])),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            if 'r_thigh_angle' in metrics and kpts.is_valid(Keypoints.R_HIP):
                hip = kpts.xy[Keypoints.R_HIP]
                angle = metrics['r_thigh_angle']
                color = (0, 255, 0) if abs(angle) < 15 else (0, 255, 255) if abs(angle) < 30 else (0, 0, 255)
                cv2.putText(output, f"R:{angle:+.0f}°", 
                           (int(hip[0]) + 10, int(hip[1])),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        return output
