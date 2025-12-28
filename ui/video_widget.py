from PyQt6.QtWidgets import QWidget, QLabel, QVBoxLayout, QSizePolicy
from PyQt6.QtCore import Qt, pyqtSignal, QPoint
from PyQt6.QtGui import QImage, QPixmap, QMouseEvent
import numpy as np
from typing import Optional, Tuple


class VideoWidget(QWidget):
    """Widget for displaying video frames with click detection"""
    
    # Signals
    frame_clicked = pyqtSignal(int, int)  # x, y in frame coordinates
    person_clicked = pyqtSignal(int)  # person index
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        self.label = QLabel()
        self.label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.label.setStyleSheet("background-color: #1e1e1e; border: 1px solid #333;")
        self.label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.label.setMinimumSize(640, 480)
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.label)
        
        self._current_frame: Optional[np.ndarray] = None
        self._scale_factor = 1.0
        self._offset = QPoint(0, 0)
        
        # Bounding boxes for click detection
        self._bboxes: list = []
    
    def update_frame(self, frame: np.ndarray, bboxes: Optional[list] = None):
        """Update displayed frame with optional bounding boxes for click detection"""
        self._current_frame = frame
        self._bboxes = bboxes or []
        
        # Convert BGR to RGB
        rgb = frame[..., ::-1].copy()
        
        h, w, ch = rgb.shape
        bytes_per_line = ch * w
        
        qimg = QImage(rgb.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg)
        
        # Scale to fit widget while maintaining aspect ratio
        scaled = pixmap.scaled(
            self.label.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        )
        
        self._scale_factor = scaled.width() / w if w > 0 else 1.0
        
        # Calculate offset for centering
        x_offset = (self.label.width() - scaled.width()) // 2
        y_offset = (self.label.height() - scaled.height()) // 2
        self._offset = QPoint(x_offset, y_offset)
        
        self.label.setPixmap(scaled)
    
    def clear(self):
        """Clear the display"""
        self.label.clear()
        self._current_frame = None
        self._bboxes = []
    
    def _widget_to_frame_coords(self, widget_pos: QPoint) -> Tuple[int, int]:
        """Convert widget coordinates to frame coordinates"""
        # Get position relative to label
        label_pos = self.label.mapFrom(self, widget_pos)
        
        # Account for centering offset
        frame_x = int((label_pos.x() - self._offset.x()) / self._scale_factor)
        frame_y = int((label_pos.y() - self._offset.y()) / self._scale_factor)
        
        return frame_x, frame_y
    
    def _find_clicked_person(self, frame_x: int, frame_y: int) -> int:
        """Find which person bbox contains the click point. Returns -1 if none."""
        for i, bbox in enumerate(self._bboxes):
            x1, y1, x2, y2 = bbox[:4]
            if x1 <= frame_x <= x2 and y1 <= frame_y <= y2:
                return i
        return -1
    
    def mousePressEvent(self, event: QMouseEvent):
        """Handle mouse clicks on video"""
        if self._current_frame is None:
            return
        
        if event.button() == Qt.MouseButton.LeftButton:
            frame_x, frame_y = self._widget_to_frame_coords(event.pos())
            
            # Emit frame coordinates
            self.frame_clicked.emit(frame_x, frame_y)
            
            # Check if clicked on a person
            person_idx = self._find_clicked_person(frame_x, frame_y)
            if person_idx >= 0:
                self.person_clicked.emit(person_idx)
    
    @property
    def frame_size(self) -> Tuple[int, int]:
        """Get current frame size"""
        if self._current_frame is not None:
            return self._current_frame.shape[1], self._current_frame.shape[0]
        return 0, 0
    
    def get_frame(self) -> Optional[np.ndarray]:
        """Get current frame"""
        return self._current_frame.copy() if self._current_frame is not None else None
