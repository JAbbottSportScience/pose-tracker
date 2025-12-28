import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Dict, Any
from threading import Thread, Lock
from queue import Queue
import json

from .metrics import BiomechanicsMetrics


class VideoRecorder:
    """Record video with optional pose overlay"""
    
    def __init__(self, output_path: str, fps: float = 30.0, 
                 frame_size: tuple = (1280, 720), codec: str = 'mp4v'):
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        
        self.fps = fps
        self.frame_size = frame_size
        self.codec = codec
        
        self._writer: Optional[cv2.VideoWriter] = None
        self._is_recording = False
        self._frame_count = 0
        self._lock = Lock()
        
        # Async writing
        self._queue: Queue = Queue(maxsize=100)
        self._thread: Optional[Thread] = None
    
    def start(self, filename: Optional[str] = None) -> str:
        """Start recording to file"""
        if self._is_recording:
            self.stop()
        
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"recording_{timestamp}.mp4"
        
        output_file = self.output_path / filename
        
        fourcc = cv2.VideoWriter_fourcc(*self.codec)
        self._writer = cv2.VideoWriter(
            str(output_file), fourcc, self.fps, self.frame_size
        )
        
        if not self._writer.isOpened():
            raise RuntimeError(f"Failed to open video writer: {output_file}")
        
        self._is_recording = True
        self._frame_count = 0
        
        # Start async write thread
        self._thread = Thread(target=self._write_loop, daemon=True)
        self._thread.start()
        
        return str(output_file)
    
    def _write_loop(self):
        """Background thread for writing frames"""
        while self._is_recording or not self._queue.empty():
            try:
                frame = self._queue.get(timeout=0.1)
                with self._lock:
                    if self._writer and self._writer.isOpened():
                        self._writer.write(frame)
            except:
                continue
    
    def write_frame(self, frame: np.ndarray):
        """Add frame to recording queue"""
        if not self._is_recording:
            return
        
        # Resize if needed
        if frame.shape[1] != self.frame_size[0] or frame.shape[0] != self.frame_size[1]:
            frame = cv2.resize(frame, self.frame_size)
        
        try:
            self._queue.put_nowait(frame)
            self._frame_count += 1
        except:
            pass  # Drop frame if queue full
    
    def stop(self) -> int:
        """Stop recording and return frame count"""
        self._is_recording = False
        
        if self._thread:
            self._thread.join(timeout=2.0)
        
        with self._lock:
            if self._writer:
                self._writer.release()
                self._writer = None
        
        count = self._frame_count
        self._frame_count = 0
        return count
    
    @property
    def is_recording(self) -> bool:
        return self._is_recording
    
    @property
    def frame_count(self) -> int:
        return self._frame_count


class CSVLogger:
    """Log biomechanical metrics to CSV"""
    
    def __init__(self, output_path: str):
        self.output_path = Path(output_path)
        self.output_path.mkdir(parents=True, exist_ok=True)
        
        self._file_path: Optional[Path] = None
        self._data: List[Dict] = []
        self._is_logging = False
        self._session_info: Dict = {}
    
    def start(self, filename: Optional[str] = None, session_info: Optional[Dict] = None) -> str:
        """Start logging to CSV file"""
        if self._is_logging:
            self.stop()
        
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"metrics_{timestamp}.csv"
        
        self._file_path = self.output_path / filename
        self._data = []
        self._is_logging = True
        self._session_info = session_info or {}
        
        # Save session info as JSON
        if self._session_info:
            info_path = self._file_path.with_suffix('.json')
            with open(info_path, 'w') as f:
                json.dump(self._session_info, f, indent=2)
        
        return str(self._file_path)
    
    def log(self, metrics: BiomechanicsMetrics):
        """Log a single metrics instance"""
        if not self._is_logging:
            return
        
        self._data.append(metrics.to_flat_dict())
    
    def log_batch(self, metrics_list: List[BiomechanicsMetrics]):
        """Log multiple metrics instances"""
        for m in metrics_list:
            self.log(m)
    
    def log_dict(self, data: Dict):
        """Log arbitrary dictionary data"""
        if not self._is_logging:
            return
        self._data.append(data)
    
    def stop(self) -> str:
        """Stop logging and save to file"""
        if not self._is_logging or not self._data:
            self._is_logging = False
            return ""
        
        df = pd.DataFrame(self._data)
        df.to_csv(self._file_path, index=False)
        
        self._is_logging = False
        saved_path = str(self._file_path)
        self._file_path = None
        self._data = []
        
        return saved_path
    
    def flush(self):
        """Write current data to file without stopping"""
        if not self._is_logging or not self._data or not self._file_path:
            return
        
        df = pd.DataFrame(self._data)
        df.to_csv(self._file_path, index=False)
    
    @property
    def is_logging(self) -> bool:
        return self._is_logging
    
    @property
    def row_count(self) -> int:
        return len(self._data)


class SessionRecorder:
    """Combined video and CSV recording for a session"""
    
    def __init__(self, output_dir: str = "data/recordings"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.video_recorder: Optional[VideoRecorder] = None
        self.csv_logger: Optional[CSVLogger] = None
        
        self._session_id: Optional[str] = None
        self._is_recording = False
    
    def start(self, include_video: bool = True, include_csv: bool = True,
              frame_size: tuple = (1280, 720), fps: float = 30.0,
              session_info: Optional[Dict] = None) -> Dict[str, str]:
        """Start a new recording session"""
        if self._is_recording:
            self.stop()
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._session_id = f"session_{timestamp}"
        
        session_dir = self.output_dir / self._session_id
        session_dir.mkdir(exist_ok=True)
        
        paths = {'session_dir': str(session_dir)}
        
        if include_video:
            self.video_recorder = VideoRecorder(session_dir, fps, frame_size)
            paths['video'] = self.video_recorder.start(f"{self._session_id}.mp4")
        
        if include_csv:
            self.csv_logger = CSVLogger(session_dir)
            paths['csv'] = self.csv_logger.start(f"{self._session_id}.csv", session_info)
        
        self._is_recording = True
        return paths
    
    def record_frame(self, frame: np.ndarray, metrics: Optional[BiomechanicsMetrics] = None):
        """Record a frame and optionally its metrics"""
        if not self._is_recording:
            return
        
        if self.video_recorder:
            self.video_recorder.write_frame(frame)
        
        if self.csv_logger and metrics:
            self.csv_logger.log(metrics)
    
    def stop(self) -> Dict[str, Any]:
        """Stop recording and return summary"""
        if not self._is_recording:
            return {}
        
        summary = {'session_id': self._session_id}
        
        if self.video_recorder:
            summary['video_frames'] = self.video_recorder.stop()
            self.video_recorder = None
        
        if self.csv_logger:
            summary['csv_path'] = self.csv_logger.stop()
            self.csv_logger = None
        
        self._is_recording = False
        self._session_id = None
        
        return summary
    
    @property
    def is_recording(self) -> bool:
        return self._is_recording
