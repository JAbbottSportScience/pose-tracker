from PyQt6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QFormLayout,
                              QLineEdit, QComboBox, QCheckBox, QPushButton,
                              QGroupBox, QFileDialog, QSpinBox, QDoubleSpinBox,
                              QTabWidget, QWidget, QLabel)
from PyQt6.QtCore import Qt
import yaml
from pathlib import Path


class SettingsDialog(QDialog):
    """Settings configuration dialog with tabs"""
    
    def __init__(self, config: dict, parent=None):
        super().__init__(parent)
        
        self.config = config.copy()
        self.setWindowTitle("Settings")
        self.setMinimumSize(500, 450)
        
        layout = QVBoxLayout(self)
        
        # Tab widget
        tabs = QTabWidget()
        layout.addWidget(tabs)
        
        # --- Camera Tab ---
        camera_tab = QWidget()
        cam_layout = QVBoxLayout(camera_tab)
        
        cam_group = QGroupBox("Primary Camera")
        cam_form = QFormLayout(cam_group)
        
        self.source_edit = QLineEdit(str(config.get('camera', {}).get('source', '0')))
        self.source_browse = QPushButton("Browse...")
        self.source_browse.clicked.connect(self._browse_source)
        
        source_layout = QHBoxLayout()
        source_layout.addWidget(self.source_edit)
        source_layout.addWidget(self.source_browse)
        cam_form.addRow("Source:", source_layout)
        
        self.width_spin = QSpinBox()
        self.width_spin.setRange(320, 3840)
        self.width_spin.setValue(config.get('camera', {}).get('width', 1280))
        cam_form.addRow("Width:", self.width_spin)
        
        self.height_spin = QSpinBox()
        self.height_spin.setRange(240, 2160)
        self.height_spin.setValue(config.get('camera', {}).get('height', 720))
        cam_form.addRow("Height:", self.height_spin)
        
        self.fps_spin = QSpinBox()
        self.fps_spin.setRange(1, 120)
        self.fps_spin.setValue(config.get('camera', {}).get('fps', 30))
        cam_form.addRow("FPS:", self.fps_spin)
        
        cam_layout.addWidget(cam_group)
        
        # Secondary camera for stereo
        cam2_group = QGroupBox("Secondary Camera (Stereo)")
        cam2_form = QFormLayout(cam2_group)
        
        self.source2_edit = QLineEdit(str(config.get('camera', {}).get('source_2', '') or ''))
        self.source2_browse = QPushButton("Browse...")
        self.source2_browse.clicked.connect(self._browse_source2)
        
        source2_layout = QHBoxLayout()
        source2_layout.addWidget(self.source2_edit)
        source2_layout.addWidget(self.source2_browse)
        cam2_form.addRow("Source:", source2_layout)
        
        cam_layout.addWidget(cam2_group)
        cam_layout.addStretch()
        
        tabs.addTab(camera_tab, "Camera")
        
        # --- Pose Tab ---
        pose_tab = QWidget()
        pose_layout = QVBoxLayout(pose_tab)
        
        pose_group = QGroupBox("Pose Estimation")
        pose_form = QFormLayout(pose_group)
        
        self.model_combo = QComboBox()
        self.model_combo.addItems([
        'yolo11n-pose.pt', 'yolo11s-pose.pt', 'yolo11m-pose.pt',
        'yolo11l-pose.pt', 'yolo11x-pose.pt'
        ])
        current_model = config.get('pose', {}).get('model', 'yolo11x-pose.pt')
        self.model_combo.setCurrentText(current_model)
        pose_form.addRow("Model:", self.model_combo)
        
        self.device_combo = QComboBox()
        self.device_combo.addItems(['mps', 'cuda', 'cpu'])
        self.device_combo.setCurrentText(config.get('pose', {}).get('device', 'mps'))
        pose_form.addRow("Device:", self.device_combo)
        
        self.conf_spin = QDoubleSpinBox()
        self.conf_spin.setRange(0.1, 0.9)
        self.conf_spin.setSingleStep(0.05)
        self.conf_spin.setValue(config.get('pose', {}).get('confidence', 0.5))
        pose_form.addRow("Confidence:", self.conf_spin)
        
        pose_layout.addWidget(pose_group)
        pose_layout.addStretch()
        
        tabs.addTab(pose_tab, "Pose")
        
        # --- Display Tab ---
        display_tab = QWidget()
        display_layout = QVBoxLayout(display_tab)
        
        display_group = QGroupBox("Display Options")
        display_form = QFormLayout(display_group)
        
        self.skeleton_check = QCheckBox()
        self.skeleton_check.setChecked(config.get('ui', {}).get('show_skeleton', True))
        display_form.addRow("Show Skeleton:", self.skeleton_check)
        
        self.keypoints_check = QCheckBox()
        self.keypoints_check.setChecked(config.get('ui', {}).get('show_keypoints', True))
        display_form.addRow("Show Keypoints:", self.keypoints_check)
        
        self.show_3d_check = QCheckBox()
        self.show_3d_check.setChecked(config.get('ui', {}).get('show_3d_view', False))
        display_form.addRow("Show 3D View:", self.show_3d_check)
        
        display_layout.addWidget(display_group)
        display_layout.addStretch()
        
        tabs.addTab(display_tab, "Display")
        
        # --- Recording Tab ---
        record_tab = QWidget()
        record_layout = QVBoxLayout(record_tab)
        
        record_group = QGroupBox("Recording")
        record_form = QFormLayout(record_group)
        
        self.record_path_edit = QLineEdit(
            config.get('recording', {}).get('output_path', 'data/recordings/')
        )
        self.record_path_browse = QPushButton("Browse...")
        self.record_path_browse.clicked.connect(self._browse_record_path)
        
        record_path_layout = QHBoxLayout()
        record_path_layout.addWidget(self.record_path_edit)
        record_path_layout.addWidget(self.record_path_browse)
        record_form.addRow("Output Path:", record_path_layout)
        
        self.include_overlay_check = QCheckBox()
        self.include_overlay_check.setChecked(
            config.get('recording', {}).get('include_overlay', True)
        )
        record_form.addRow("Include Overlay:", self.include_overlay_check)
        
        self.log_csv_check = QCheckBox()
        self.log_csv_check.setChecked(
            config.get('metrics', {}).get('log_to_csv', False)
        )
        record_form.addRow("Log Metrics to CSV:", self.log_csv_check)
        
        record_layout.addWidget(record_group)
        record_layout.addStretch()
        
        tabs.addTab(record_tab, "Recording")
        
        # --- Stereo Tab ---
        stereo_tab = QWidget()
        stereo_layout = QVBoxLayout(stereo_tab)
        
        stereo_group = QGroupBox("Stereo Calibration")
        stereo_form = QFormLayout(stereo_group)
        
        self.calib_path_edit = QLineEdit(
            config.get('stereo', {}).get('calibration_file', '') or ''
        )
        self.calib_path_browse = QPushButton("Browse...")
        self.calib_path_browse.clicked.connect(self._browse_calib_path)
        
        calib_path_layout = QHBoxLayout()
        calib_path_layout.addWidget(self.calib_path_edit)
        calib_path_layout.addWidget(self.calib_path_browse)
        stereo_form.addRow("Calibration File:", calib_path_layout)
        
        self.stereo_enabled_check = QCheckBox()
        self.stereo_enabled_check.setChecked(
            config.get('stereo', {}).get('enabled', False)
        )
        stereo_form.addRow("Enable Stereo:", self.stereo_enabled_check)
        
        stereo_layout.addWidget(stereo_group)
        
        # Calibration info
        info_label = QLabel(
            "To create a stereo calibration:\n"
            "1. Print a 9x6 checkerboard (25mm squares)\n"
            "2. Run the calibration script\n"
            "3. Load the resulting JSON file here"
        )
        info_label.setStyleSheet("color: #888; padding: 10px;")
        stereo_layout.addWidget(info_label)
        stereo_layout.addStretch()
        
        tabs.addTab(stereo_tab, "Stereo")
        
        # --- Buttons ---
        button_layout = QHBoxLayout()
        
        save_btn = QPushButton("Save")
        save_btn.clicked.connect(self.accept)
        
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        
        export_btn = QPushButton("Export Config...")
        export_btn.clicked.connect(self._export_config)
        
        button_layout.addWidget(export_btn)
        button_layout.addStretch()
        button_layout.addWidget(cancel_btn)
        button_layout.addWidget(save_btn)
        
        layout.addLayout(button_layout)
    
    def _browse_source(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Video File", "",
            "Video Files (*.mp4 *.mov *.avi *.mkv);;All Files (*)"
        )
        if file_path:
            self.source_edit.setText(file_path)
    
    def _browse_source2(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Video File", "",
            "Video Files (*.mp4 *.mov *.avi *.mkv);;All Files (*)"
        )
        if file_path:
            self.source2_edit.setText(file_path)
    
    def _browse_record_path(self):
        dir_path = QFileDialog.getExistingDirectory(self, "Select Output Directory")
        if dir_path:
            self.record_path_edit.setText(dir_path)
    
    def _browse_calib_path(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Calibration File", "",
            "JSON Files (*.json);;All Files (*)"
        )
        if file_path:
            self.calib_path_edit.setText(file_path)
    
    def _export_config(self):
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Export Configuration", "config.yaml",
            "YAML Files (*.yaml *.yml);;All Files (*)"
        )
        if file_path:
            config = self.get_config()
            with open(file_path, 'w') as f:
                yaml.dump(config, f, default_flow_style=False)
    
    def get_config(self) -> dict:
        """Return updated config"""
        source2 = self.source2_edit.text().strip()
        calib_file = self.calib_path_edit.text().strip()
        
        return {
            'camera': {
                'source': self.source_edit.text(),
                'source_2': source2 if source2 else None,
                'width': self.width_spin.value(),
                'height': self.height_spin.value(),
                'fps': self.fps_spin.value()
            },
            'pose': {
                'model': self.model_combo.currentText(),
                'device': self.device_combo.currentText(),
                'confidence': self.conf_spin.value()
            },
            'ui': {
                'show_skeleton': self.skeleton_check.isChecked(),
                'show_keypoints': self.keypoints_check.isChecked(),
                'show_3d_view': self.show_3d_check.isChecked(),
                'theme': 'dark'
            },
            'recording': {
                'output_path': self.record_path_edit.text(),
                'include_overlay': self.include_overlay_check.isChecked(),
                'format': 'mp4'
            },
            'metrics': {
                'log_to_csv': self.log_csv_check.isChecked(),
                'show_angles': True,
                'show_distances': True
            },
            'stereo': {
                'enabled': self.stereo_enabled_check.isChecked(),
                'calibration_file': calib_file if calib_file else None
            }
        }
