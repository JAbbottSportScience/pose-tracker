"""
Analysis Panel - Post-session review with synced video and metrics graphs

Features:
- Load recorded sessions (CSV + video)
- Interactive time-series graphs of biomechanical metrics
- Video playback synced with graph cursor
- Summary statistics (min, max, avg, std)
- Session comparison mode
"""

from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QSplitter,
                              QPushButton, QLabel, QComboBox, QGroupBox,
                              QFormLayout, QFileDialog, QSlider, QCheckBox,
                              QTableWidget, QTableWidgetItem, QTabWidget,
                              QListWidget, QListWidgetItem, QScrollArea,
                              QFrame, QGridLayout, QSpinBox, QMessageBox)
from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QImage, QPixmap, QColor
import numpy as np
import pandas as pd
import cv2
from pathlib import Path
from typing import Optional, List, Dict, Tuple
from dataclasses import dataclass

import matplotlib
matplotlib.use('QtAgg')
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure


@dataclass
class SessionData:
    """Container for loaded session data"""
    name: str
    csv_path: Path
    video_path: Optional[Path]
    df: pd.DataFrame
    duration: float  # seconds
    fps: float
    
    @property
    def num_frames(self) -> int:
        return len(self.df)


class MetricsGraphWidget(QWidget):
    """Interactive matplotlib widget for metrics time series"""
    
    cursor_moved = pyqtSignal(float)  # Emits timestamp when cursor moves
    
    # Metric groups for organized display
    METRIC_GROUPS = {
        'Thigh Angles': ['l_thigh_angle', 'r_thigh_angle'],
        'Shank Angles': ['l_shank_angle', 'r_shank_angle'],
        'Knee Angles': ['l_knee_angle', 'r_knee_angle'],
        'Hip Angles': ['l_hip_angle', 'r_hip_angle'],
        'Elbow Angles': ['l_elbow_angle', 'r_elbow_angle'],
        'Trunk': ['trunk_lean'],
        'Distances': ['l_hip_ankle', 'r_hip_ankle', 'shoulder_width', 'hip_width'],
    }
    
    COLORS = {
        'l_thigh_angle': '#00ff00', 'r_thigh_angle': '#ff0000',
        'l_shank_angle': '#00cc00', 'r_shank_angle': '#cc0000',
        'l_knee_angle': '#00ff88', 'r_knee_angle': '#ff8800',
        'l_hip_angle': '#00ffcc', 'r_hip_angle': '#ffcc00',
        'l_elbow_angle': '#88ff00', 'r_elbow_angle': '#ff0088',
        'trunk_lean': '#00aaff',
        'l_hip_ankle': '#00ff00', 'r_hip_ankle': '#ff0000',
        'shoulder_width': '#ffff00', 'hip_width': '#ff00ff',
    }
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        self.session: Optional[SessionData] = None
        self.comparison_session: Optional[SessionData] = None
        self._cursor_line = None
        self._current_time = 0
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # Metric selector
        selector_layout = QHBoxLayout()
        
        selector_layout.addWidget(QLabel("Metrics:"))
        self.metric_combo = QComboBox()
        self.metric_combo.addItems(list(self.METRIC_GROUPS.keys()))
        self.metric_combo.currentTextChanged.connect(self._update_plot)
        selector_layout.addWidget(self.metric_combo)
        
        self.show_comparison = QCheckBox("Show Comparison")
        self.show_comparison.setEnabled(False)
        self.show_comparison.toggled.connect(self._update_plot)
        selector_layout.addWidget(self.show_comparison)
        
        selector_layout.addStretch()
        layout.addLayout(selector_layout)
        
        # Matplotlib figure
        self.figure = Figure(figsize=(10, 4), facecolor='#1e1e1e')
        self.canvas = FigureCanvas(self.figure)
        self.canvas.mpl_connect('button_press_event', self._on_click)
        self.canvas.mpl_connect('motion_notify_event', self._on_motion)
        
        # Toolbar for zoom/pan
        self.toolbar = NavigationToolbar(self.canvas, self)
        self.toolbar.setStyleSheet("background-color: #2d2d2d;")
        
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas, stretch=1)
        
        self._setup_axes()
    
    def _setup_axes(self):
        """Setup the plot axes"""
        self.ax = self.figure.add_subplot(111)
        self.ax.set_facecolor('#1e1e1e')
        self.figure.patch.set_facecolor('#1e1e1e')
        
        self.ax.set_xlabel('Time (s)', color='white')
        self.ax.set_ylabel('Angle (°)', color='white')
        self.ax.tick_params(colors='white')
        self.ax.spines['bottom'].set_color('white')
        self.ax.spines['left'].set_color('white')
        self.ax.spines['top'].set_visible(False)
        self.ax.spines['right'].set_visible(False)
        self.ax.grid(True, alpha=0.3, color='white')
    
    def load_session(self, session: SessionData):
        """Load a session for plotting"""
        self.session = session
        self._update_plot()
    
    def load_comparison(self, session: Optional[SessionData]):
        """Load a comparison session"""
        self.comparison_session = session
        self.show_comparison.setEnabled(session is not None)
        if session is None:
            self.show_comparison.setChecked(False)
        self._update_plot()
    
    def _update_plot(self):
        """Update the plot with current metric group"""
        self.ax.clear()
        self._setup_axes()
        
        if self.session is None:
            self.canvas.draw_idle()
            return
        
        group_name = self.metric_combo.currentText()
        metrics = self.METRIC_GROUPS.get(group_name, [])
        
        df = self.session.df
        
        # Create time axis
        if 'timestamp' in df.columns:
            time = df['timestamp'] - df['timestamp'].iloc[0]
        else:
            time = np.arange(len(df)) / self.session.fps
        
        # Plot each metric in group
        for metric in metrics:
            if metric in df.columns:
                color = self.COLORS.get(metric, '#ffffff')
                label = metric.replace('_', ' ').title()
                self.ax.plot(time, df[metric], color=color, label=label, linewidth=1.5)
        
        # Plot comparison session
        if self.show_comparison.isChecked() and self.comparison_session is not None:
            comp_df = self.comparison_session.df
            if 'timestamp' in comp_df.columns:
                comp_time = comp_df['timestamp'] - comp_df['timestamp'].iloc[0]
            else:
                comp_time = np.arange(len(comp_df)) / self.comparison_session.fps
            
            for metric in metrics:
                if metric in comp_df.columns:
                    color = self.COLORS.get(metric, '#ffffff')
                    label = f"{metric.replace('_', ' ').title()} (comp)"
                    self.ax.plot(comp_time, comp_df[metric], color=color, 
                                linestyle='--', alpha=0.6, label=label, linewidth=1.5)
        
        # Add cursor line
        if self._current_time is not None:
            self._cursor_line = self.ax.axvline(x=self._current_time, color='#ff0000', 
                                                 linewidth=2, alpha=0.8)
        
        self.ax.legend(loc='upper right', facecolor='#2d2d2d', edgecolor='#444',
                       labelcolor='white', fontsize=8)
        self.ax.set_xlim(0, time.iloc[-1] if len(time) > 0 else 1)
        
        self.canvas.draw_idle()
    
    def set_cursor_time(self, time: float):
        """Set the cursor position"""
        self._current_time = time
        if self._cursor_line:
            self._cursor_line.set_xdata([time, time])
            self.canvas.draw_idle()
        else:
            self._update_plot()
    
    def _on_click(self, event):
        """Handle click to move cursor"""
        if event.inaxes == self.ax and event.xdata is not None:
            self._current_time = event.xdata
            self.cursor_moved.emit(event.xdata)
            self.set_cursor_time(event.xdata)
    
    def _on_motion(self, event):
        """Handle drag to scrub"""
        if event.button == 1 and event.inaxes == self.ax and event.xdata is not None:
            self._current_time = event.xdata
            self.cursor_moved.emit(event.xdata)
            self.set_cursor_time(event.xdata)


class VideoPlayerWidget(QWidget):
    """Video player with frame-accurate seeking"""
    
    frame_changed = pyqtSignal(int, float)  # frame_number, timestamp
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        self._cap: Optional[cv2.VideoCapture] = None
        self._timer = QTimer()
        self._timer.timeout.connect(self._next_frame)
        self._is_playing = False
        self._current_frame = 0
        self._total_frames = 0
        self._fps = 30.0
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # Video display
        self.video_label = QLabel()
        self.video_label.setMinimumSize(640, 360)
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_label.setStyleSheet("background-color: #000; border: 1px solid #333;")
        layout.addWidget(self.video_label, stretch=1)
        
        # Controls
        controls = QHBoxLayout()
        
        self.play_btn = QPushButton("▶")
        self.play_btn.setFixedWidth(40)
        self.play_btn.clicked.connect(self._toggle_play)
        controls.addWidget(self.play_btn)
        
        self.time_label = QLabel("00:00.0 / 00:00.0")
        self.time_label.setStyleSheet("color: #888;")
        controls.addWidget(self.time_label)
        
        self.seek_slider = QSlider(Qt.Orientation.Horizontal)
        self.seek_slider.sliderMoved.connect(self._seek)
        self.seek_slider.sliderPressed.connect(self._pause)
        controls.addWidget(self.seek_slider, stretch=1)
        
        # Speed control
        controls.addWidget(QLabel("Speed:"))
        self.speed_combo = QComboBox()
        self.speed_combo.addItems(['0.25x', '0.5x', '1x', '2x'])
        self.speed_combo.setCurrentText('1x')
        self.speed_combo.currentTextChanged.connect(self._update_speed)
        controls.addWidget(self.speed_combo)
        
        layout.addLayout(controls)
    
    def load_video(self, path: str) -> bool:
        """Load a video file"""
        self.close()
        
        self._cap = cv2.VideoCapture(path)
        if not self._cap.isOpened():
            return False
        
        self._total_frames = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self._fps = self._cap.get(cv2.CAP_PROP_FPS) or 30.0
        self._current_frame = 0
        
        self.seek_slider.setRange(0, self._total_frames - 1)
        self.seek_slider.setValue(0)
        
        self._show_frame(0)
        return True
    
    def _show_frame(self, frame_num: int):
        """Display a specific frame"""
        if self._cap is None:
            return
        
        self._cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = self._cap.read()
        
        if ret:
            self._current_frame = frame_num
            
            # Convert to QPixmap
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb.shape
            qimg = QImage(rgb.data, w, h, ch * w, QImage.Format.Format_RGB888)
            pixmap = QPixmap.fromImage(qimg)
            
            # Scale to fit
            scaled = pixmap.scaled(self.video_label.size(),
                                   Qt.AspectRatioMode.KeepAspectRatio,
                                   Qt.TransformationMode.SmoothTransformation)
            self.video_label.setPixmap(scaled)
            
            # Update time label
            current_time = frame_num / self._fps
            total_time = self._total_frames / self._fps
            self.time_label.setText(
                f"{int(current_time//60):02d}:{current_time%60:04.1f} / "
                f"{int(total_time//60):02d}:{total_time%60:04.1f}"
            )
            
            # Update slider
            self.seek_slider.blockSignals(True)
            self.seek_slider.setValue(frame_num)
            self.seek_slider.blockSignals(False)
            
            self.frame_changed.emit(frame_num, current_time)
    
    def _next_frame(self):
        """Advance to next frame"""
        if self._current_frame < self._total_frames - 1:
            self._show_frame(self._current_frame + 1)
        else:
            self._pause()
    
    def _toggle_play(self):
        """Toggle play/pause"""
        if self._is_playing:
            self._pause()
        else:
            self._play()
    
    def _play(self):
        """Start playback"""
        if self._cap is None:
            return
        
        self._is_playing = True
        self.play_btn.setText("⏸")
        
        speed = float(self.speed_combo.currentText().replace('x', ''))
        interval = int(1000 / (self._fps * speed))
        self._timer.start(interval)
    
    def _pause(self):
        """Pause playback"""
        self._is_playing = False
        self.play_btn.setText("▶")
        self._timer.stop()
    
    def _seek(self, frame_num: int):
        """Seek to frame"""
        self._show_frame(frame_num)
    
    def seek_to_time(self, time: float):
        """Seek to time in seconds"""
        frame = int(time * self._fps)
        frame = max(0, min(frame, self._total_frames - 1))
        self._show_frame(frame)
    
    def _update_speed(self):
        """Update playback speed"""
        if self._is_playing:
            self._play()  # Restart timer with new interval
    
    def close(self):
        """Close video"""
        self._timer.stop()
        if self._cap:
            self._cap.release()
            self._cap = None
        self.video_label.clear()
        self._is_playing = False
        self.play_btn.setText("▶")
    
    @property
    def fps(self) -> float:
        return self._fps
    
    @property
    def duration(self) -> float:
        return self._total_frames / self._fps if self._fps > 0 else 0


class StatisticsWidget(QWidget):
    """Display summary statistics for session metrics"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        layout = QVBoxLayout(self)
        
        self.table = QTableWidget()
        self.table.setColumnCount(6)
        self.table.setHorizontalHeaderLabels(['Metric', 'Min', 'Max', 'Mean', 'Std', 'Range'])
        self.table.setStyleSheet("""
            QTableWidget {
                background-color: #2a2a2a;
                color: #ccc;
                gridline-color: #444;
            }
            QHeaderView::section {
                background-color: #333;
                color: white;
                padding: 5px;
                border: 1px solid #444;
            }
        """)
        layout.addWidget(self.table)
    
    def load_session(self, session: SessionData):
        """Calculate and display statistics"""
        df = session.df
        
        # Get numeric columns (excluding timestamp and frame_number)
        numeric_cols = [col for col in df.select_dtypes(include=[np.number]).columns
                       if col not in ['timestamp', 'frame_number']]
        
        self.table.setRowCount(len(numeric_cols))
        
        for i, col in enumerate(numeric_cols):
            values = df[col].dropna()
            if len(values) == 0:
                continue
            
            self.table.setItem(i, 0, QTableWidgetItem(col.replace('_', ' ').title()))
            self.table.setItem(i, 1, QTableWidgetItem(f"{values.min():.1f}"))
            self.table.setItem(i, 2, QTableWidgetItem(f"{values.max():.1f}"))
            self.table.setItem(i, 3, QTableWidgetItem(f"{values.mean():.1f}"))
            self.table.setItem(i, 4, QTableWidgetItem(f"{values.std():.1f}"))
            self.table.setItem(i, 5, QTableWidgetItem(f"{values.max() - values.min():.1f}"))
        
        self.table.resizeColumnsToContents()
    
    def clear(self):
        """Clear the table"""
        self.table.setRowCount(0)


class AnalysisPanel(QWidget):
    """Main analysis panel combining video, graphs, and statistics"""
    
    def __init__(self, recordings_path: str = "data/recordings", parent=None):
        super().__init__(parent)
        
        self.recordings_path = Path(recordings_path)
        self.session: Optional[SessionData] = None
        self.comparison_session: Optional[SessionData] = None
        
        self._setup_ui()
    
    def _setup_ui(self):
        """Setup the user interface"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        
        # Top: Session selection
        session_layout = QHBoxLayout()
        
        session_layout.addWidget(QLabel("Session:"))
        self.session_combo = QComboBox()
        self.session_combo.setMinimumWidth(300)
        self.session_combo.currentTextChanged.connect(self._load_session)
        session_layout.addWidget(self.session_combo)
        
        refresh_btn = QPushButton("🔄 Refresh")
        refresh_btn.clicked.connect(self._refresh_sessions)
        session_layout.addWidget(refresh_btn)
        
        browse_btn = QPushButton("📂 Browse...")
        browse_btn.clicked.connect(self._browse_session)
        session_layout.addWidget(browse_btn)
        
        session_layout.addSpacing(20)
        
        session_layout.addWidget(QLabel("Compare with:"))
        self.comparison_combo = QComboBox()
        self.comparison_combo.addItem("None")
        self.comparison_combo.currentTextChanged.connect(self._load_comparison)
        session_layout.addWidget(self.comparison_combo)
        
        session_layout.addStretch()
        layout.addLayout(session_layout)
        
        # Main content splitter
        main_splitter = QSplitter(Qt.Orientation.Horizontal)
        layout.addWidget(main_splitter, stretch=1)
        
        # Left: Video player
        video_container = QWidget()
        video_layout = QVBoxLayout(video_container)
        video_layout.setContentsMargins(0, 0, 0, 0)
        
        video_label = QLabel("Video Playback")
        video_label.setStyleSheet("color: #0af; font-weight: bold;")
        video_layout.addWidget(video_label)
        
        self.video_player = VideoPlayerWidget()
        self.video_player.frame_changed.connect(self._on_frame_changed)
        video_layout.addWidget(self.video_player)
        
        main_splitter.addWidget(video_container)
        
        # Right: Tabs for graphs and stats
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_layout.setContentsMargins(0, 0, 0, 0)
        
        tabs = QTabWidget()
        
        # Graphs tab
        self.graphs_widget = MetricsGraphWidget()
        self.graphs_widget.cursor_moved.connect(self._on_graph_cursor_moved)
        tabs.addTab(self.graphs_widget, "📈 Time Series")
        
        # Statistics tab
        self.stats_widget = StatisticsWidget()
        tabs.addTab(self.stats_widget, "📊 Statistics")
        
        # Events tab (placeholder for future)
        events_widget = QWidget()
        events_layout = QVBoxLayout(events_widget)
        events_layout.addWidget(QLabel("Event detection coming soon..."))
        events_layout.addWidget(QLabel("• Stride detection\n• Peak angle detection\n• Phase segmentation"))
        events_layout.addStretch()
        tabs.addTab(events_widget, "🎯 Events")
        
        right_layout.addWidget(tabs)
        main_splitter.addWidget(right_widget)
        
        # Set initial splitter sizes (40% video, 60% analysis)
        main_splitter.setSizes([400, 600])
        
        # Initial refresh
        self._refresh_sessions()
    
    def _refresh_sessions(self):
        """Refresh list of available sessions"""
        self.session_combo.clear()
        self.comparison_combo.clear()
        self.comparison_combo.addItem("None")
        
        if not self.recordings_path.exists():
            return
        
        sessions = []
        
        # Look for session directories
        for session_dir in sorted(self.recordings_path.iterdir(), reverse=True):
            if session_dir.is_dir():
                csv_files = list(session_dir.glob("*.csv"))
                if csv_files:
                    sessions.append(session_dir.name)
        
        # Also look for loose CSV files
        for csv_file in sorted(self.recordings_path.glob("*.csv"), reverse=True):
            sessions.append(csv_file.stem)
        
        for session in sessions:
            self.session_combo.addItem(session)
            self.comparison_combo.addItem(session)
    
    def _browse_session(self):
        """Browse for a session CSV file"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Session CSV",
            str(self.recordings_path),
            "CSV Files (*.csv);;All Files (*)"
        )
        
        if file_path:
            self._load_session_from_path(Path(file_path))
    
    def _load_session(self, session_name: str):
        """Load a session by name"""
        if not session_name:
            return
        
        # Try session directory first
        session_dir = self.recordings_path / session_name
        if session_dir.exists():
            csv_files = list(session_dir.glob("*.csv"))
            if csv_files:
                self._load_session_from_path(csv_files[0])
                return
        
        # Try loose CSV
        csv_path = self.recordings_path / f"{session_name}.csv"
        if csv_path.exists():
            self._load_session_from_path(csv_path)
    
    def _load_session_from_path(self, csv_path: Path):
        """Load session from CSV path"""
        try:
            df = pd.read_csv(csv_path)
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Failed to load CSV: {e}")
            return
        
        # Look for video file
        video_path = None
        for ext in ['.mp4', '.avi', '.mov', '.mkv']:
            candidate = csv_path.with_suffix(ext)
            if candidate.exists():
                video_path = candidate
                break
        
        # Determine FPS and duration
        fps = 30.0  # Default
        if 'timestamp' in df.columns and len(df) > 1:
            time_diffs = df['timestamp'].diff().dropna()
            if len(time_diffs) > 0:
                avg_diff = time_diffs.mean()
                if avg_diff > 0:
                    fps = 1.0 / avg_diff
        
        duration = len(df) / fps
        
        self.session = SessionData(
            name=csv_path.stem,
            csv_path=csv_path,
            video_path=video_path,
            df=df,
            duration=duration,
            fps=fps
        )
        
        # Load video if available
        if video_path:
            self.video_player.load_video(str(video_path))
        else:
            self.video_player.close()
        
        # Update graphs and stats
        self.graphs_widget.load_session(self.session)
        self.stats_widget.load_session(self.session)
    
    def _load_comparison(self, session_name: str):
        """Load comparison session"""
        if session_name == "None" or not session_name:
            self.comparison_session = None
            self.graphs_widget.load_comparison(None)
            return
        
        # Try session directory first
        session_dir = self.recordings_path / session_name
        if session_dir.exists():
            csv_files = list(session_dir.glob("*.csv"))
            if csv_files:
                self._load_comparison_from_path(csv_files[0])
                return
        
        # Try loose CSV
        csv_path = self.recordings_path / f"{session_name}.csv"
        if csv_path.exists():
            self._load_comparison_from_path(csv_path)
    
    def _load_comparison_from_path(self, csv_path: Path):
        """Load comparison session from path"""
        try:
            df = pd.read_csv(csv_path)
        except Exception as e:
            return
        
        fps = 30.0
        if 'timestamp' in df.columns and len(df) > 1:
            time_diffs = df['timestamp'].diff().dropna()
            if len(time_diffs) > 0:
                avg_diff = time_diffs.mean()
                if avg_diff > 0:
                    fps = 1.0 / avg_diff
        
        self.comparison_session = SessionData(
            name=csv_path.stem,
            csv_path=csv_path,
            video_path=None,
            df=df,
            duration=len(df) / fps,
            fps=fps
        )
        
        self.graphs_widget.load_comparison(self.comparison_session)
    
    def _on_frame_changed(self, frame_num: int, timestamp: float):
        """Handle video frame change - sync graph cursor"""
        self.graphs_widget.set_cursor_time(timestamp)
    
    def _on_graph_cursor_moved(self, timestamp: float):
        """Handle graph cursor move - sync video"""
        self.video_player.seek_to_time(timestamp)
    
    def set_recordings_path(self, path: str):
        """Update recordings path and refresh"""
        self.recordings_path = Path(path)
        self._refresh_sessions()
