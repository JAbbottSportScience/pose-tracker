import cv2
import numpy as np
from threading import Thread, Lock
from typing import Optional, Tuple
from dataclasses import dataclass
import time


@dataclass
class FrameData:
    """Container for frame data with metadata"""
    frame: np.ndarray
    timestamp: float
    frame_number: int


class VideoSource:
    """Handles video files, webcams, and RTSP streams uniformly"""
    
    def __init__(self, source, width: int = 1280, height: int = 720, fps: int = 30):
        self.source = source
        self.width = width
        self.height = height
        self.target_fps = fps
        
        self._cap: Optional[cv2.VideoCapture] = None
        self._frame: Optional[np.ndarray] = None
        self._timestamp: float = 0
        self._frame_number: int = 0
        self._lock = Lock()
        self._running = False
        self._thread: Optional[Thread] = None
        
        # Video file properties
        self.is_file = False
        self.total_frames = 0
        self.duration = 0
        self.actual_fps = fps
    
    def open(self) -> bool:
        """Open the video source"""
        if isinstance(self.source, int):
            # Webcam
            self._cap = cv2.VideoCapture(self.source)
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            self._cap.set(cv2.CAP_PROP_FPS, self.target_fps)
        elif str(self.source).startswith('rtsp://'):
            # RTSP stream
            import os
            os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"
            self._cap = cv2.VideoCapture(self.source, cv2.CAP_FFMPEG)
            self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        else:
            # Video file
            self._cap = cv2.VideoCapture(str(self.source))
            self.is_file = True
            self.total_frames = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.actual_fps = self._cap.get(cv2.CAP_PROP_FPS) or self.target_fps
            self.duration = self.total_frames / self.actual_fps if self.actual_fps > 0 else 0
        
        return self._cap is not None and self._cap.isOpened()
    
    def start(self) -> bool:
        """Start background frame capture thread"""
        if not self._cap or not self._cap.isOpened():
            if not self.open():
                return False
        
        self._running = True
        self._thread = Thread(target=self._capture_loop, daemon=True)
        self._thread.start()
        return True
    
    def _capture_loop(self):
        """Background capture loop"""
        while self._running:
            ret, frame = self._cap.read()
            
            if not ret:
                if self.is_file:
                    # Loop video files
                    self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    self._frame_number = 0
                    continue
                else:
                    time.sleep(0.01)
                    continue
            
            with self._lock:
                self._frame = frame
                self._timestamp = time.time()
                self._frame_number += 1
            
            # Throttle for video files to match original fps
            if self.is_file:
                time.sleep(1.0 / self.actual_fps)
    
    def read(self) -> Optional[FrameData]:
        """Get the latest frame"""
        with self._lock:
            if self._frame is None:
                return None
            return FrameData(
                frame=self._frame.copy(),
                timestamp=self._timestamp,
                frame_number=self._frame_number
            )
    
    def read_sync(self) -> Optional[FrameData]:
        """Synchronous read - blocks until new frame"""
        if not self._cap or not self._cap.isOpened():
            return None
        
        ret, frame = self._cap.read()
        if not ret:
            return None
        
        self._frame_number += 1
        return FrameData(
            frame=frame,
            timestamp=time.time(),
            frame_number=self._frame_number
        )
    
    def seek(self, frame_number: int):
        """Seek to specific frame (video files only)"""
        if self.is_file and self._cap:
            with self._lock:
                self._cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
                self._frame_number = frame_number
    
    def stop(self):
        """Stop capture"""
        self._running = False
        if self._thread:
            self._thread.join(timeout=1.0)
        if self._cap:
            self._cap.release()
            self._cap = None
    
    @property
    def is_opened(self) -> bool:
        return self._cap is not None and self._cap.isOpened()
    
    @property
    def frame_position(self) -> int:
        return self._frame_number
    
    @property
    def frame_size(self) -> Tuple[int, int]:
        """Return (width, height) of frames"""
        if self._cap:
            w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            return (w, h)
        return (self.width, self.height)


class CameraStream(VideoSource):
    """Alias for backwards compatibility"""
    pass


class DualCameraSource:
    """Synchronized dual camera source for stereo capture"""
    
    def __init__(self, source1, source2, width: int = 1280, height: int = 720, fps: int = 30):
        self.cam1 = VideoSource(source1, width, height, fps)
        self.cam2 = VideoSource(source2, width, height, fps)
        self._running = False
    
    def start(self) -> bool:
        """Start both cameras"""
        ok1 = self.cam1.start()
        ok2 = self.cam2.start()
        self._running = ok1 and ok2
        return self._running
    
    def read(self) -> Tuple[Optional[FrameData], Optional[FrameData]]:
        """Read from both cameras"""
        return self.cam1.read(), self.cam2.read()
    
    def stop(self):
        """Stop both cameras"""
        self._running = False
        self.cam1.stop()
        self.cam2.stop()
    
    @property
    def is_opened(self) -> bool:
        return self.cam1.is_opened and self.cam2.is_opened
    
    @property
    def sync_delta_ms(self) -> float:
        """Get time delta between cameras in milliseconds"""
        f1 = self.cam1.read()
        f2 = self.cam2.read()
        if f1 and f2:
            return abs(f1.timestamp - f2.timestamp) * 1000
        return float('inf')
