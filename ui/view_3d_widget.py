from PyQt6.QtWidgets import QWidget, QVBoxLayout
from PyQt6.QtCore import Qt
import numpy as np
from typing import Optional, Dict

import matplotlib
matplotlib.use('QtAgg')
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D

from core.triangulation import Points3D


class View3DWidget(QWidget):
    """Widget for 3D skeleton visualization using matplotlib"""
    
    SKELETON = [
        (0, 1), (0, 2), (1, 3), (2, 4),           # Head
        (5, 6),                                    # Shoulders
        (5, 7), (7, 9),                           # Left arm
        (6, 8), (8, 10),                          # Right arm
        (5, 11), (6, 12), (11, 12),               # Torso
        (11, 13), (13, 15),                       # Left leg
        (12, 14), (14, 16)                        # Right leg
    ]
    
    KEYPOINT_NAMES = [
        'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
        'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
        'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
        'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
    ]
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        self.setMinimumSize(400, 400)
        
        # Create matplotlib figure
        self.figure = Figure(figsize=(5, 5), facecolor='#1e1e1e')
        self.canvas = FigureCanvas(self.figure)
        
        # Create 3D axes
        self.ax = self.figure.add_subplot(111, projection='3d')
        self._setup_axes()
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.canvas)
        
        # Store last valid points for smooth updates
        self._last_points: Optional[Points3D] = None
    
    def _setup_axes(self):
        """Setup axis labels and appearance"""
        self.ax.set_facecolor('#1e1e1e')
        self.figure.patch.set_facecolor('#1e1e1e')
        
        self.ax.set_xlabel('X (m)', color='white', fontsize=9)
        self.ax.set_ylabel('Z (m)', color='white', fontsize=9)
        self.ax.set_zlabel('Y (m)', color='white', fontsize=9)
        
        # Set tick colors
        self.ax.tick_params(colors='white', labelsize=7)
        self.ax.xaxis.pane.fill = False
        self.ax.yaxis.pane.fill = False
        self.ax.zaxis.pane.fill = False
        
        # Grid
        self.ax.xaxis._axinfo['grid']['color'] = (0.3, 0.3, 0.3, 0.3)
        self.ax.yaxis._axinfo['grid']['color'] = (0.3, 0.3, 0.3, 0.3)
        self.ax.zaxis._axinfo['grid']['color'] = (0.3, 0.3, 0.3, 0.3)
        
        self.ax.view_init(elev=15, azim=-60)
    
    def _get_segment_color(self, start_idx: int, end_idx: int) -> str:
        """Get color for skeleton segment"""
        # Head
        if start_idx in [0, 1, 2, 3, 4] or end_idx in [0, 1, 2, 3, 4]:
            return '#00ffff'  # Cyan
        # Left arm
        if start_idx in [5, 7, 9] and end_idx in [5, 7, 9]:
            return '#00ff00'  # Green
        # Right arm
        if start_idx in [6, 8, 10] and end_idx in [6, 8, 10]:
            return '#ff0000'  # Red
        # Torso
        if start_idx in [5, 6, 11, 12] and end_idx in [5, 6, 11, 12]:
            return '#0088ff'  # Blue
        # Left leg
        if start_idx in [11, 13, 15] and end_idx in [11, 13, 15]:
            return '#00ff00'  # Green
        # Right leg
        if start_idx in [12, 14, 16] and end_idx in [12, 14, 16]:
            return '#ff0000'  # Red
        return '#ffffff'
    
    def _get_keypoint_color(self, idx: int) -> str:
        """Get color for keypoint"""
        name = self.KEYPOINT_NAMES[idx]
        if 'left' in name:
            return '#00ff00'  # Green
        elif 'right' in name:
            return '#ff0000'  # Red
        return '#00ffff'  # Cyan
    
    def update_skeleton(self, points_3d: Points3D, metrics: Optional[Dict] = None):
        """Update 3D skeleton visualization"""
        self._last_points = points_3d
        
        self.ax.cla()
        self._setup_axes()
        
        xyz = points_3d.xyz
        valid = points_3d.valid
        
        # Auto-scale based on valid points
        valid_points = xyz[valid]
        if len(valid_points) > 0:
            padding = 0.3
            x_min, x_max = valid_points[:, 0].min() - padding, valid_points[:, 0].max() + padding
            y_min, y_max = valid_points[:, 1].min() - padding, valid_points[:, 1].max() + padding
            z_min, z_max = valid_points[:, 2].min() - padding, valid_points[:, 2].max() + padding
            
            self.ax.set_xlim(x_min, x_max)
            self.ax.set_ylim(z_min, z_max)  # Z is depth
            self.ax.set_zlim(-y_max, -y_min)  # Y inverted for height
        
        # Plot keypoints
        for i, (pt, v) in enumerate(zip(xyz, valid)):
            if v:
                color = self._get_keypoint_color(i)
                self.ax.scatter(pt[0], pt[2], -pt[1], 
                               c=color, s=50, edgecolors='white', linewidth=0.5)
        
        # Plot skeleton segments
        for start, end in self.SKELETON:
            if valid[start] and valid[end]:
                color = self._get_segment_color(start, end)
                self.ax.plot(
                    [xyz[start, 0], xyz[end, 0]],
                    [xyz[start, 2], xyz[end, 2]],
                    [-xyz[start, 1], -xyz[end, 1]],
                    color=color, linewidth=2.5, alpha=0.9
                )
        
        # Draw ground plane reference
        if len(valid_points) > 0:
            ground_y = -valid_points[:, 1].max()
            xx, zz = np.meshgrid(
                np.linspace(x_min, x_max, 3),
                np.linspace(z_min, z_max, 3)
            )
            yy = np.ones_like(xx) * ground_y
            self.ax.plot_surface(xx, zz, yy, alpha=0.1, color='gray')
        
        # Add metrics text
        if metrics:
            text_lines = []
            if 'l_thigh_angle_3d' in metrics:
                text_lines.append(f"L Thigh: {metrics['l_thigh_angle_3d']:.1f}°")
            if 'r_thigh_angle_3d' in metrics:
                text_lines.append(f"R Thigh: {metrics['r_thigh_angle_3d']:.1f}°")
            if 'l_knee_valgus' in metrics:
                val = metrics['l_knee_valgus']
                label = 'valgus' if val > 0 else 'varus'
                text_lines.append(f"L Knee: {abs(val):.1f}° {label}")
            if 'r_knee_valgus' in metrics:
                val = metrics['r_knee_valgus']
                label = 'valgus' if val > 0 else 'varus'
                text_lines.append(f"R Knee: {abs(val):.1f}° {label}")
            
            if text_lines:
                self.ax.text2D(0.02, 0.98, '\n'.join(text_lines),
                              transform=self.ax.transAxes, fontsize=8,
                              verticalalignment='top', fontfamily='monospace',
                              bbox=dict(boxstyle='round', facecolor='black', alpha=0.7),
                              color='white')
        
        self.canvas.draw_idle()
    
    def clear(self):
        """Clear the visualization"""
        self.ax.cla()
        self._setup_axes()
        self.canvas.draw_idle()
        self._last_points = None
    
    def set_view_angle(self, elev: float, azim: float):
        """Set the viewing angle"""
        self.ax.view_init(elev=elev, azim=azim)
        self.canvas.draw_idle()
    
    def reset_view(self):
        """Reset to default view angle"""
        self.ax.view_init(elev=15, azim=-60)
        self.canvas.draw_idle()
