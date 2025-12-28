from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
                              QSplitter, QSizePolicy)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QImage, QPixmap
import numpy as np
from typing import Optional, Tuple

from .video_widget import VideoWidget


class DualViewWidget(QWidget):
    """Widget for displaying two camera views side by side"""
    
    # Signals
    cam1_clicked = pyqtSignal(int, int)  # x, y
    cam2_clicked = pyqtSignal(int, int)  # x, y
    person_clicked = pyqtSignal(int, int)  # camera_idx, person_idx
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # Splitter for resizable views
        self.splitter = QSplitter(Qt.Orientation.Horizontal)
        layout.addWidget(self.splitter)
        
        # Camera 1
        self.cam1_container = QWidget()
        cam1_layout = QVBoxLayout(self.cam1_container)
        cam1_layout.setContentsMargins(0, 0, 0, 0)
        
        self.cam1_label = QLabel("Camera 1")
        self.cam1_label.setStyleSheet("color: #0af; font-weight: bold; padding: 2px;")
        cam1_layout.addWidget(self.cam1_label)
        
        self.cam1_view = VideoWidget()
        self.cam1_view.frame_clicked.connect(lambda x, y: self.cam1_clicked.emit(x, y))
        self.cam1_view.person_clicked.connect(lambda idx: self.person_clicked.emit(0, idx))
        cam1_layout.addWidget(self.cam1_view)
        
        self.splitter.addWidget(self.cam1_container)
        
        # Camera 2
        self.cam2_container = QWidget()
        cam2_layout = QVBoxLayout(self.cam2_container)
        cam2_layout.setContentsMargins(0, 0, 0, 0)
        
        self.cam2_label = QLabel("Camera 2")
        self.cam2_label.setStyleSheet("color: #0af; font-weight: bold; padding: 2px;")
        cam2_layout.addWidget(self.cam2_label)
        
        self.cam2_view = VideoWidget()
        self.cam2_view.frame_clicked.connect(lambda x, y: self.cam2_clicked.emit(x, y))
        self.cam2_view.person_clicked.connect(lambda idx: self.person_clicked.emit(1, idx))
        cam2_layout.addWidget(self.cam2_view)
        
        self.splitter.addWidget(self.cam2_container)
        
        # Sync indicator
        self.sync_label = QLabel("Sync: --")
        self.sync_label.setStyleSheet("color: #888; padding: 2px;")
        self.sync_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.sync_label)
    
    def update_frames(self, frame1: np.ndarray, frame2: np.ndarray,
                      bboxes1: Optional[list] = None, bboxes2: Optional[list] = None):
        """Update both camera views"""
        self.cam1_view.update_frame(frame1, bboxes1)
        self.cam2_view.update_frame(frame2, bboxes2)
    
    def update_cam1(self, frame: np.ndarray, bboxes: Optional[list] = None):
        """Update camera 1 only"""
        self.cam1_view.update_frame(frame, bboxes)
    
    def update_cam2(self, frame: np.ndarray, bboxes: Optional[list] = None):
        """Update camera 2 only"""
        self.cam2_view.update_frame(frame, bboxes)
    
    def set_labels(self, label1: str, label2: str):
        """Set camera labels"""
        self.cam1_label.setText(label1)
        self.cam2_label.setText(label2)
    
    def set_sync_status(self, delta_ms: float):
        """Update sync status indicator"""
        if delta_ms < 33:
            color = "#0f0"  # Green
            status = "Good"
        elif delta_ms < 66:
            color = "#ff0"  # Yellow
            status = "OK"
        else:
            color = "#f00"  # Red
            status = "Poor"
        
        self.sync_label.setText(f"Sync: {delta_ms:.0f}ms ({status})")
        self.sync_label.setStyleSheet(f"color: {color}; padding: 2px;")
    
    def clear(self):
        """Clear both views"""
        self.cam1_view.clear()
        self.cam2_view.clear()
        self.sync_label.setText("Sync: --")
        self.sync_label.setStyleSheet("color: #888; padding: 2px;")
    
    def get_frame(self, camera: int) -> Optional[np.ndarray]:
        """Get current frame from camera (0 or 1)"""
        if camera == 0:
            return self.cam1_view.get_frame()
        elif camera == 1:
            return self.cam2_view.get_frame()
        return None


class TripleViewWidget(QWidget):
    """Widget for two cameras plus 3D view"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        from .view_3d_widget import View3DWidget
        
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # Main splitter
        main_splitter = QSplitter(Qt.Orientation.Horizontal)
        layout.addWidget(main_splitter)
        
        # Left: dual camera views
        self.dual_view = DualViewWidget()
        main_splitter.addWidget(self.dual_view)
        
        # Right: 3D view
        self.view_3d = View3DWidget()
        main_splitter.addWidget(self.view_3d)
        
        # Set initial sizes (cameras get 60%, 3D gets 40%)
        main_splitter.setSizes([600, 400])
    
    def update_frames(self, frame1: np.ndarray, frame2: np.ndarray,
                      bboxes1: Optional[list] = None, bboxes2: Optional[list] = None):
        """Update camera views"""
        self.dual_view.update_frames(frame1, frame2, bboxes1, bboxes2)
    
    def update_3d(self, points_3d, metrics: Optional[dict] = None):
        """Update 3D view"""
        self.view_3d.update_skeleton(points_3d, metrics)
    
    def set_sync_status(self, delta_ms: float):
        """Update sync status"""
        self.dual_view.set_sync_status(delta_ms)
    
    def clear(self):
        """Clear all views"""
        self.dual_view.clear()
        self.view_3d.clear()
