from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
                              QGroupBox, QGridLayout, QFrame, QScrollArea)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont
from typing import Optional
from core.metrics import BiomechanicsMetrics


class MetricLabel(QWidget):
    """Single metric display with label and value"""
    
    def __init__(self, name: str, unit: str = "°", parent=None):
        super().__init__(parent)
        
        self.name = name
        self.unit = unit
        
        layout = QHBoxLayout(self)
        layout.setContentsMargins(5, 2, 5, 2)
        
        self.name_label = QLabel(f"{name}:")
        self.name_label.setStyleSheet("color: #888;")
        
        self.value_label = QLabel("--")
        self.value_label.setStyleSheet("color: #0f0; font-weight: bold;")
        self.value_label.setAlignment(Qt.AlignmentFlag.AlignRight)
        self.value_label.setMinimumWidth(80)
        
        layout.addWidget(self.name_label)
        layout.addStretch()
        layout.addWidget(self.value_label)
    
    def set_value(self, value: Optional[float], 
                  threshold_warn: Optional[float] = None, 
                  threshold_bad: Optional[float] = None):
        """Update value with optional color thresholds"""
        if value is None:
            self.value_label.setText("--")
            self.value_label.setStyleSheet("color: #666; font-weight: bold;")
            return
        
        text = f"{value:.1f}{self.unit}"
        self.value_label.setText(text)
        
        # Color based on thresholds
        if threshold_bad and abs(value) > threshold_bad:
            color = "#f00"  # Red
        elif threshold_warn and abs(value) > threshold_warn:
            color = "#ff0"  # Yellow
        else:
            color = "#0f0"  # Green
        
        self.value_label.setStyleSheet(f"color: {color}; font-weight: bold;")
    
    def clear(self):
        """Clear the value"""
        self.value_label.setText("--")
        self.value_label.setStyleSheet("color: #666; font-weight: bold;")


class MetricsPanel(QWidget):
    """Panel showing all biomechanical metrics"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        self.setMinimumWidth(280)
        self.setMaximumWidth(350)
        self.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                border: 1px solid #444;
                border-radius: 5px;
                margin-top: 10px;
                padding-top: 10px;
                background-color: #2a2a2a;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
                color: #0af;
            }
        """)
        
        # Scroll area for many metrics
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setStyleSheet("QScrollArea { border: none; background: transparent; }")
        
        scroll_widget = QWidget()
        layout = QVBoxLayout(scroll_widget)
        layout.setSpacing(10)
        
        # --- Thigh angles group ---
        thigh_group = QGroupBox("Thigh Angles")
        thigh_layout = QVBoxLayout(thigh_group)
        self.l_thigh = MetricLabel("Left Thigh")
        self.r_thigh = MetricLabel("Right Thigh")
        thigh_layout.addWidget(self.l_thigh)
        thigh_layout.addWidget(self.r_thigh)
        layout.addWidget(thigh_group)
        
        # --- Shank angles group ---
        shank_group = QGroupBox("Shank Angles")
        shank_layout = QVBoxLayout(shank_group)
        self.l_shank = MetricLabel("Left Shank")
        self.r_shank = MetricLabel("Right Shank")
        shank_layout.addWidget(self.l_shank)
        shank_layout.addWidget(self.r_shank)
        layout.addWidget(shank_group)
        
        # --- Knee angles group ---
        knee_group = QGroupBox("Knee Angles")
        knee_layout = QVBoxLayout(knee_group)
        self.l_knee = MetricLabel("Left Knee")
        self.r_knee = MetricLabel("Right Knee")
        knee_layout.addWidget(self.l_knee)
        knee_layout.addWidget(self.r_knee)
        layout.addWidget(knee_group)
        
        # --- Hip angles group ---
        hip_group = QGroupBox("Hip Angles")
        hip_layout = QVBoxLayout(hip_group)
        self.l_hip = MetricLabel("Left Hip")
        self.r_hip = MetricLabel("Right Hip")
        hip_layout.addWidget(self.l_hip)
        hip_layout.addWidget(self.r_hip)
        layout.addWidget(hip_group)
        
        # --- Elbow angles group ---
        elbow_group = QGroupBox("Elbow Angles")
        elbow_layout = QVBoxLayout(elbow_group)
        self.l_elbow = MetricLabel("Left Elbow")
        self.r_elbow = MetricLabel("Right Elbow")
        elbow_layout.addWidget(self.l_elbow)
        elbow_layout.addWidget(self.r_elbow)
        layout.addWidget(elbow_group)
        
        # --- Distances group ---
        dist_group = QGroupBox("Distances")
        dist_layout = QVBoxLayout(dist_group)
        self.l_hip_ankle = MetricLabel("L Hip-Ankle", "px")
        self.r_hip_ankle = MetricLabel("R Hip-Ankle", "px")
        self.shoulder_w = MetricLabel("Shoulder W", "px")
        self.hip_w = MetricLabel("Hip W", "px")
        dist_layout.addWidget(self.l_hip_ankle)
        dist_layout.addWidget(self.r_hip_ankle)
        dist_layout.addWidget(self.shoulder_w)
        dist_layout.addWidget(self.hip_w)
        layout.addWidget(dist_group)
        
        # --- Trunk group ---
        trunk_group = QGroupBox("Trunk")
        trunk_layout = QVBoxLayout(trunk_group)
        self.trunk_lean = MetricLabel("Trunk Lean")
        trunk_layout.addWidget(self.trunk_lean)
        layout.addWidget(trunk_group)
        
        # --- 3D Metrics group (for stereo mode) ---
        self.group_3d = QGroupBox("3D Metrics")
        layout_3d = QVBoxLayout(self.group_3d)
        self.l_thigh_3d = MetricLabel("L Thigh 3D")
        self.r_thigh_3d = MetricLabel("R Thigh 3D")
        self.l_valgus = MetricLabel("L Knee Valgus")
        self.r_valgus = MetricLabel("R Knee Valgus")
        self.shoulder_w_m = MetricLabel("Shoulder W", "cm")
        self.hip_w_m = MetricLabel("Hip W", "cm")
        layout_3d.addWidget(self.l_thigh_3d)
        layout_3d.addWidget(self.r_thigh_3d)
        layout_3d.addWidget(self.l_valgus)
        layout_3d.addWidget(self.r_valgus)
        layout_3d.addWidget(self.shoulder_w_m)
        layout_3d.addWidget(self.hip_w_m)
        layout.addWidget(self.group_3d)
        self.group_3d.hide()  # Hidden by default
        
        layout.addStretch()
        
        scroll.setWidget(scroll_widget)
        
        # Main layout
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        
        # FPS display at top
        self.fps_label = QLabel("FPS: --")
        self.fps_label.setStyleSheet("color: #888; padding: 5px; font-weight: bold;")
        main_layout.addWidget(self.fps_label)
        
        # Recording indicator
        self.recording_label = QLabel("")
        self.recording_label.setStyleSheet("color: #f00; padding: 5px; font-weight: bold;")
        main_layout.addWidget(self.recording_label)
        
        main_layout.addWidget(scroll)
    
    def update_metrics(self, metrics: BiomechanicsMetrics):
        """Update all metric displays from BiomechanicsMetrics"""
        # Thigh angles
        self.l_thigh.set_value(metrics.l_thigh_angle, threshold_warn=15, threshold_bad=30)
        self.r_thigh.set_value(metrics.r_thigh_angle, threshold_warn=15, threshold_bad=30)
        
        # Shank angles
        self.l_shank.set_value(metrics.l_shank_angle)
        self.r_shank.set_value(metrics.r_shank_angle)
        
        # Knee angles
        self.l_knee.set_value(metrics.l_knee_angle)
        self.r_knee.set_value(metrics.r_knee_angle)
        
        # Hip angles
        self.l_hip.set_value(metrics.l_hip_angle)
        self.r_hip.set_value(metrics.r_hip_angle)
        
        # Elbow angles
        self.l_elbow.set_value(metrics.l_elbow_angle)
        self.r_elbow.set_value(metrics.r_elbow_angle)
        
        # Distances
        self.l_hip_ankle.set_value(metrics.l_hip_ankle)
        self.r_hip_ankle.set_value(metrics.r_hip_ankle)
        self.shoulder_w.set_value(metrics.shoulder_width)
        self.hip_w.set_value(metrics.hip_width)
        
        # Trunk
        self.trunk_lean.set_value(metrics.trunk_lean, threshold_warn=10, threshold_bad=20)
    
    def update_3d_metrics(self, metrics: dict):
        """Update 3D metrics from triangulation"""
        self.group_3d.show()
        
        self.l_thigh_3d.set_value(metrics.get('l_thigh_angle_3d'))
        self.r_thigh_3d.set_value(metrics.get('r_thigh_angle_3d'))
        
        l_valgus = metrics.get('l_knee_valgus')
        r_valgus = metrics.get('r_knee_valgus')
        self.l_valgus.set_value(l_valgus, threshold_warn=10, threshold_bad=15)
        self.r_valgus.set_value(r_valgus, threshold_warn=10, threshold_bad=15)
        
        # Convert to cm
        if 'shoulder_width_m' in metrics:
            self.shoulder_w_m.set_value(metrics['shoulder_width_m'] * 100)
        if 'hip_width_m' in metrics:
            self.hip_w_m.set_value(metrics['hip_width_m'] * 100)
    
    def hide_3d_metrics(self):
        """Hide 3D metrics group"""
        self.group_3d.hide()
    
    def set_fps(self, fps: float):
        """Update FPS display"""
        self.fps_label.setText(f"FPS: {fps:.1f}")
    
    def set_recording(self, is_recording: bool, frame_count: int = 0):
        """Update recording indicator"""
        if is_recording:
            self.recording_label.setText(f"● REC ({frame_count} frames)")
            self.recording_label.setStyleSheet("color: #f00; padding: 5px; font-weight: bold;")
        else:
            self.recording_label.setText("")
    
    def clear_all(self):
        """Clear all metric values"""
        for attr in dir(self):
            widget = getattr(self, attr)
            if isinstance(widget, MetricLabel):
                widget.clear()
