from PyQt6.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                              QPushButton, QFileDialog, QStatusBar, QToolBar,
                              QSlider, QLabel, QMessageBox, QInputDialog,
                              QStackedWidget, QSplitter, QMenu)
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QAction, QIcon, QKeySequence
import time
import yaml
from pathlib import Path
from typing import Optional

from .video_widget import VideoWidget
from .metrics_panel import MetricsPanel
from .settings_dialog import SettingsDialog
from .player_dialog import PlayerDialog, QuickTagDialog
from .view_3d_widget import View3DWidget
from .dual_view_widget import DualViewWidget, TripleViewWidget

from core import (VideoSource, DualCameraSource, PoseEstimator, MetricsCalculator,
                  StereoTriangulator, Metrics3DCalculator, SessionRecorder, 
                  PlayerTagger, LiveTagger)


class MainWindow(QMainWindow):
    """Main application window"""
    
    def __init__(self, config: dict):
        super().__init__()
        
        self.config = config
        
        # Core components
        self.video_source: Optional[VideoSource] = None
        self.dual_source: Optional[DualCameraSource] = None
        self.pose_estimator: Optional[PoseEstimator] = None
        self.metrics_calc = MetricsCalculator()
        self.triangulator: Optional[StereoTriangulator] = None
        self.metrics_3d_calc = Metrics3DCalculator()
        self.session_recorder = SessionRecorder(config.get('recording', {}).get('output_path', 'data/recordings'))
        self.player_tagger = PlayerTagger(config.get('players', {}).get('database_path', 'data/headshots'))
        self.live_tagger = LiveTagger(self.player_tagger)
        
        # State
        self._is_playing = False
        self._is_stereo_mode = False
        self._fps_counter = 0
        self._fps_time = time.time()
        self._current_fps = 0
        
        self._setup_ui()
        self._setup_timer()
        self._setup_shortcuts()
        self._init_pose_model()
        self._load_stereo_calibration()
    
    def _setup_ui(self):
        """Setup the user interface"""
        self.setWindowTitle("Pose Tracker")
        self.setMinimumSize(1400, 900)
        self._apply_dark_theme()
        
        # Central widget with stacked views
        central = QWidget()
        self.setCentralWidget(central)
        
        main_layout = QHBoxLayout(central)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(10)
        
        # Left side - video views (stacked for single/dual/triple modes)
        self.view_stack = QStackedWidget()
        
        # Single camera view
        self.single_view_container = QWidget()
        single_layout = QVBoxLayout(self.single_view_container)
        single_layout.setContentsMargins(0, 0, 0, 0)
        self.video_widget = VideoWidget()
        self.video_widget.person_clicked.connect(self._on_person_clicked)
        single_layout.addWidget(self.video_widget)
        self.view_stack.addWidget(self.single_view_container)
        
        # Dual camera view
        self.dual_view = DualViewWidget()
        self.view_stack.addWidget(self.dual_view)
        
        # Triple view (dual + 3D)
        self.triple_view = TripleViewWidget()
        self.view_stack.addWidget(self.triple_view)
        
        # Left layout with views and controls
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.addWidget(self.view_stack, stretch=1)
        
        # Playback controls
        controls_layout = QHBoxLayout()
        
        self.play_btn = QPushButton("▶ Play")
        self.play_btn.clicked.connect(self._toggle_playback)
        self.play_btn.setEnabled(False)
        controls_layout.addWidget(self.play_btn)
        
        self.stop_btn = QPushButton("⏹ Stop")
        self.stop_btn.clicked.connect(self._stop_playback)
        self.stop_btn.setEnabled(False)
        controls_layout.addWidget(self.stop_btn)
        
        self.record_btn = QPushButton("⏺ Record")
        self.record_btn.clicked.connect(self._toggle_recording)
        self.record_btn.setEnabled(False)
        self.record_btn.setStyleSheet("QPushButton:checked { background-color: #a00; }")
        controls_layout.addWidget(self.record_btn)
        
        controls_layout.addSpacing(20)
        
        self.time_label = QLabel("00:00 / 00:00")
        controls_layout.addWidget(self.time_label)
        
        self.seek_slider = QSlider(Qt.Orientation.Horizontal)
        self.seek_slider.setEnabled(False)
        self.seek_slider.sliderMoved.connect(self._seek)
        controls_layout.addWidget(self.seek_slider, stretch=1)
        
        left_layout.addLayout(controls_layout)
        
        main_layout.addWidget(left_widget, stretch=1)
        
        # Right side - metrics panel
        self.metrics_panel = MetricsPanel()
        main_layout.addWidget(self.metrics_panel)
        
        # Toolbar
        self._create_toolbar()
        
        # Status bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready - Open a video or connect to camera")
    
    def _create_toolbar(self):
        """Create main toolbar"""
        toolbar = QToolBar()
        toolbar.setMovable(False)
        self.addToolBar(toolbar)
        
        # File actions
        open_action = QAction("📂 Open Video", self)
        open_action.setShortcut(QKeySequence.StandardKey.Open)
        open_action.triggered.connect(self._open_video)
        toolbar.addAction(open_action)
        
        webcam_action = QAction("📷 Webcam", self)
        webcam_action.triggered.connect(self._open_webcam)
        toolbar.addAction(webcam_action)
        
        rtsp_action = QAction("📡 RTSP", self)
        rtsp_action.triggered.connect(self._open_rtsp)
        toolbar.addAction(rtsp_action)
        
        toolbar.addSeparator()
        
        # Stereo mode
        stereo_action = QAction("👁️ Dual Camera", self)
        stereo_action.triggered.connect(self._open_dual_cameras)
        toolbar.addAction(stereo_action)
        
        toolbar.addSeparator()
        
        # View toggle
        self.view_3d_action = QAction("🎲 3D View", self)
        self.view_3d_action.setCheckable(True)
        self.view_3d_action.triggered.connect(self._toggle_3d_view)
        toolbar.addAction(self.view_3d_action)
        
        toolbar.addSeparator()
        
        # Player management
        players_action = QAction("👤 Players", self)
        players_action.triggered.connect(self._open_player_manager)
        toolbar.addAction(players_action)
        
        toolbar.addSeparator()
        
        # Settings
        settings_action = QAction("⚙️ Settings", self)
        settings_action.triggered.connect(self._open_settings)
        toolbar.addAction(settings_action)
    
    def _setup_timer(self):
        """Setup frame processing timer"""
        self.timer = QTimer()
        self.timer.timeout.connect(self._process_frame)
    
    def _setup_shortcuts(self):
        """Setup keyboard shortcuts"""
        pass  # Shortcuts defined in toolbar actions
    
    def _apply_dark_theme(self):
        """Apply dark theme styling"""
        self.setStyleSheet("""
            QMainWindow { background-color: #1e1e1e; }
            QToolBar { 
                background-color: #2d2d2d; 
                border: none; 
                spacing: 5px; 
                padding: 5px; 
            }
            QToolBar QToolButton {
                background-color: transparent;
                color: white;
                padding: 5px 10px;
                border-radius: 3px;
            }
            QToolBar QToolButton:hover {
                background-color: #3d3d3d;
            }
            QToolBar QToolButton:pressed {
                background-color: #4d4d4d;
            }
            QPushButton { 
                background-color: #3d3d3d; 
                color: white; 
                border: 1px solid #555;
                padding: 8px 16px;
                border-radius: 4px;
            }
            QPushButton:hover { background-color: #4d4d4d; }
            QPushButton:pressed { background-color: #2d2d2d; }
            QPushButton:disabled { background-color: #2d2d2d; color: #666; }
            QSlider::groove:horizontal {
                border: 1px solid #444;
                height: 8px;
                background: #2d2d2d;
                border-radius: 4px;
            }
            QSlider::handle:horizontal {
                background: #0af;
                width: 16px;
                margin: -4px 0;
                border-radius: 8px;
            }
            QLabel { color: #ccc; }
            QStatusBar { background-color: #2d2d2d; color: #888; }
        """)
    
    def _init_pose_model(self):
        """Initialize pose estimation model"""
        try:
            pose_config = self.config.get('pose', {})
            self.pose_estimator = PoseEstimator(
                model_path=pose_config.get('model', 'yolo11x-pose.pt'),
                device=pose_config.get('device', 'mps'),
                confidence=pose_config.get('confidence', 0.5)
            )
            self.status_bar.showMessage("Model loaded successfully")
        except Exception as e:
            QMessageBox.warning(self, "Model Error", f"Failed to load pose model: {e}")
    
    def _load_stereo_calibration(self):
        """Load stereo calibration if configured"""
        stereo_config = self.config.get('stereo', {})
        calib_file = stereo_config.get('calibration_file')
        
        if calib_file and Path(calib_file).exists():
            try:
                self.triangulator = StereoTriangulator(calib_file)
                self.status_bar.showMessage(f"Stereo calibration loaded: {calib_file}")
            except Exception as e:
                print(f"Failed to load stereo calibration: {e}")
    
    # --- Source Opening ---
    
    def _open_video(self):
        """Open video file"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Open Video", "",
            "Video Files (*.mp4 *.mov *.avi *.mkv *.webm);;All Files (*)"
        )
        if file_path:
            self._is_stereo_mode = False
            self.view_stack.setCurrentIndex(0)
            self._start_source(file_path)
    
    def _open_webcam(self):
        """Open webcam"""
        self._is_stereo_mode = False
        self.view_stack.setCurrentIndex(0)
        self._start_source(0)
    
    def _open_rtsp(self):
        """Open RTSP stream"""
        url, ok = QInputDialog.getText(
            self, "RTSP Stream", "Enter RTSP URL:",
            text="rtsp://admin:pass@192.168.0.200:554/stream1"
        )
        if ok and url:
            self._is_stereo_mode = False
            self.view_stack.setCurrentIndex(0)
            self._start_source(url)
    
    def _open_dual_cameras(self):
        """Open dual camera setup"""
        # Get sources from config or prompt
        cam_config = self.config.get('camera', {})
        source1 = cam_config.get('source', '0')
        source2 = cam_config.get('source_2', '')
        
        if not source2:
            source2, ok = QInputDialog.getText(
                self, "Second Camera", "Enter second camera source (RTSP URL or device number):",
                text="rtsp://admin:pass@192.168.0.201:554/stream1"
            )
            if not ok or not source2:
                return
        
        self._stop_playback()
        
        self._is_stereo_mode = True
        
        # Determine view mode (with or without 3D)
        if self.triangulator and self.view_3d_action.isChecked():
            self.view_stack.setCurrentIndex(2)  # Triple view
        else:
            self.view_stack.setCurrentIndex(1)  # Dual view
        
        # Create dual source
        self.dual_source = DualCameraSource(
            source1, source2,
            width=cam_config.get('width', 1280),
            height=cam_config.get('height', 720),
            fps=cam_config.get('fps', 30)
        )
        
        if self.dual_source.start():
            self.play_btn.setEnabled(True)
            self.stop_btn.setEnabled(True)
            self.record_btn.setEnabled(True)
            self.seek_slider.setEnabled(False)
            
            self._is_playing = True
            self.play_btn.setText("⏸ Pause")
            self.timer.start(33)
            
            self.status_bar.showMessage(f"Dual cameras: {source1} + {source2}")
        else:
            QMessageBox.warning(self, "Error", "Failed to open dual cameras")
    
    def _start_source(self, source):
        """Start single video source"""
        self._stop_playback()
        
        cam_config = self.config.get('camera', {})
        self.video_source = VideoSource(
            source,
            width=cam_config.get('width', 1280),
            height=cam_config.get('height', 720),
            fps=cam_config.get('fps', 30)
        )
        
        if self.video_source.start():
            self.play_btn.setEnabled(True)
            self.stop_btn.setEnabled(True)
            self.record_btn.setEnabled(True)
            
            if self.video_source.is_file:
                self.seek_slider.setEnabled(True)
                self.seek_slider.setMaximum(self.video_source.total_frames)
            else:
                self.seek_slider.setEnabled(False)
            
            self._is_playing = True
            self.play_btn.setText("⏸ Pause")
            self.timer.start(33)
            
            self.status_bar.showMessage(f"Playing: {source}")
        else:
            QMessageBox.warning(self, "Error", f"Failed to open: {source}")
    
    # --- Playback Control ---
    
    def _toggle_playback(self):
        """Toggle play/pause"""
        if self._is_playing:
            self._is_playing = False
            self.play_btn.setText("▶ Play")
            self.timer.stop()
        else:
            self._is_playing = True
            self.play_btn.setText("⏸ Pause")
            self.timer.start(33)
    
    def _stop_playback(self):
        """Stop playback and cleanup"""
        self._is_playing = False
        self.timer.stop()
        
        # Stop recording if active
        if self.session_recorder.is_recording:
            self._toggle_recording()
        
        if self.video_source:
            self.video_source.stop()
            self.video_source = None
        
        if self.dual_source:
            self.dual_source.stop()
            self.dual_source = None
        
        self.play_btn.setText("▶ Play")
        self.play_btn.setEnabled(False)
        self.stop_btn.setEnabled(False)
        self.record_btn.setEnabled(False)
        self.seek_slider.setEnabled(False)
        
        self.video_widget.clear()
        self.dual_view.clear()
        self.triple_view.clear()
        
        self.status_bar.showMessage("Stopped")
    
    def _seek(self, position):
        """Seek to position"""
        if self.video_source and self.video_source.is_file:
            self.video_source.seek(position)
    
    # --- Recording ---
    
    def _toggle_recording(self):
        """Toggle recording on/off"""
        if self.session_recorder.is_recording:
            summary = self.session_recorder.stop()
            self.record_btn.setText("⏺ Record")
            self.record_btn.setStyleSheet("")
            self.metrics_panel.set_recording(False)
            self.status_bar.showMessage(f"Recording saved: {summary.get('video_frames', 0)} frames")
        else:
            # Get frame size from current source
            if self._is_stereo_mode and self.dual_source:
                frame_size = self.dual_source.cam1.frame_size
            elif self.video_source:
                frame_size = self.video_source.frame_size
            else:
                frame_size = (1280, 720)
            
            paths = self.session_recorder.start(
                include_video=True,
                include_csv=self.config.get('metrics', {}).get('log_to_csv', True),
                frame_size=frame_size
            )
            
            self.record_btn.setText("⏹ Stop Rec")
            self.record_btn.setStyleSheet("background-color: #a00;")
            self.status_bar.showMessage(f"Recording to: {paths.get('session_dir', '')}")
    
    # --- View Toggles ---
    
    def _toggle_3d_view(self, checked: bool):
        """Toggle 3D view"""
        if self._is_stereo_mode:
            if checked and self.triangulator:
                self.view_stack.setCurrentIndex(2)  # Triple view
            else:
                self.view_stack.setCurrentIndex(1)  # Dual view
    
    # --- Frame Processing ---
    
    def _process_frame(self):
        """Process a single frame"""
        if self._is_stereo_mode:
            self._process_stereo_frame()
        else:
            self._process_single_frame()
    
    def _process_single_frame(self):
        """Process single camera frame"""
        if not self.video_source:
            return
        
        frame_data = self.video_source.read()
        if frame_data is None:
            return
        
        frame = frame_data.frame
        bboxes = []
        
        # Run pose estimation
        if self.pose_estimator:
            result = self.pose_estimator.process(frame)
            
            # Get person labels for tagging
            labels = self.live_tagger.get_labels(result.num_people)
            
            # Draw pose
            ui_config = self.config.get('ui', {})
            frame = self.pose_estimator.draw(
                frame, result,
                show_skeleton=ui_config.get('show_skeleton', True),
                show_keypoints=ui_config.get('show_keypoints', True),
                person_ids=labels
            )
            
            # Extract bboxes for click detection
            bboxes = result.boxes.tolist() if len(result.boxes) > 0 else []
            
            # Calculate and display metrics for first person
            if result.keypoints:
                metrics = self.metrics_calc.calculate(
                    result.keypoints[0],
                    timestamp=frame_data.timestamp,
                    frame_number=frame_data.frame_number
                )
                self.metrics_panel.update_metrics(metrics)
                
                # Record if active
                if self.session_recorder.is_recording:
                    self.session_recorder.record_frame(frame, metrics)
                    self.metrics_panel.set_recording(True, self.session_recorder.video_recorder.frame_count if self.session_recorder.video_recorder else 0)
        
        # Update display
        self.video_widget.update_frame(frame, bboxes)
        
        # Update timeline for video files
        if self.video_source.is_file:
            self.seek_slider.blockSignals(True)
            self.seek_slider.setValue(self.video_source.frame_position)
            self.seek_slider.blockSignals(False)
            
            current = self.video_source.frame_position / self.video_source.actual_fps
            total = self.video_source.duration
            self.time_label.setText(
                f"{int(current//60):02d}:{int(current%60):02d} / "
                f"{int(total//60):02d}:{int(total%60):02d}"
            )
        
        self._update_fps()
    
    def _process_stereo_frame(self):
        """Process stereo camera frames"""
        if not self.dual_source:
            return
        
        frame1_data, frame2_data = self.dual_source.read()
        if frame1_data is None or frame2_data is None:
            return
        
        frame1 = frame1_data.frame
        frame2 = frame2_data.frame
        
        # Undistort if calibrated
        if self.triangulator:
            frame1, frame2 = self.triangulator.undistort(frame1, frame2)
        
        # Run pose on both
        result1, result2 = None, None
        if self.pose_estimator:
            result1 = self.pose_estimator.process(frame1)
            result2 = self.pose_estimator.process(frame2)
            
            ui_config = self.config.get('ui', {})
            frame1 = self.pose_estimator.draw(frame1, result1,
                show_skeleton=ui_config.get('show_skeleton', True),
                show_keypoints=ui_config.get('show_keypoints', True))
            frame2 = self.pose_estimator.draw(frame2, result2,
                show_skeleton=ui_config.get('show_skeleton', True),
                show_keypoints=ui_config.get('show_keypoints', True))
            
            # 2D metrics from cam1
            if result1.keypoints:
                metrics = self.metrics_calc.calculate(
                    result1.keypoints[0],
                    timestamp=frame1_data.timestamp,
                    frame_number=frame1_data.frame_number
                )
                self.metrics_panel.update_metrics(metrics)
        
        # Triangulate if possible
        if self.triangulator and result1 and result2 and result1.keypoints and result2.keypoints:
            points_3d = self.triangulator.triangulate(
                result1.keypoints[0].xy,
                result2.keypoints[0].xy,
                result1.keypoints[0].confidence,
                result2.keypoints[0].confidence
            )
            
            metrics_3d = self.metrics_3d_calc.calculate(points_3d)
            self.metrics_panel.update_3d_metrics(metrics_3d)
            
            # Update 3D view if visible
            if self.view_stack.currentIndex() == 2:
                self.triple_view.update_3d(points_3d, metrics_3d)
        
        # Update views
        sync_ms = abs(frame1_data.timestamp - frame2_data.timestamp) * 1000
        
        if self.view_stack.currentIndex() == 2:
            self.triple_view.update_frames(frame1, frame2)
            self.triple_view.set_sync_status(sync_ms)
        else:
            self.dual_view.update_frames(frame1, frame2)
            self.dual_view.set_sync_status(sync_ms)
        
        self._update_fps()
    
    def _update_fps(self):
        """Update FPS counter"""
        self._fps_counter += 1
        if self._fps_counter >= 30:
            elapsed = time.time() - self._fps_time
            self._current_fps = self._fps_counter / elapsed if elapsed > 0 else 0
            self.metrics_panel.set_fps(self._current_fps)
            self._fps_counter = 0
            self._fps_time = time.time()
    
    # --- Player Tagging ---
    
    def _on_person_clicked(self, person_idx: int):
        """Handle click on detected person"""
        dialog = QuickTagDialog(self.player_tagger, self)
        if dialog.exec():
            player_id = dialog.get_selected_player_id()
            if player_id:
                self.live_tagger.set_pending(person_idx)
                self.live_tagger.assign_tag(player_id)
                
                # Save body crop
                frame = self.video_widget.get_frame()
                if frame is not None:
                    # Get bbox for person
                    # This would need the bbox from the last pose result
                    pass
    
    def _open_player_manager(self):
        """Open player management dialog"""
        dialog = PlayerDialog(self.player_tagger, self)
        dialog.exec()
    
    # --- Settings ---
    
    def _open_settings(self):
        """Open settings dialog"""
        dialog = SettingsDialog(self.config, self)
        if dialog.exec():
            self.config = dialog.get_config()
            self._init_pose_model()
            self._load_stereo_calibration()
            
            # Save config
            with open('config.yaml', 'w') as f:
                yaml.dump(self.config, f, default_flow_style=False)
    
    # --- Cleanup ---
    
    def closeEvent(self, event):
        """Handle window close"""
        self._stop_playback()
        event.accept()
