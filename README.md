# Pose Tracker

Real-time skeletal pose estimation and biomechanical analysis application built with PyQt6 and YOLOv8.

## Features

- **Real-time Pose Estimation**: YOLO11-pose model with Apple Silicon (MPS), CUDA, and CPU support
- **Single & Dual Camera Support**: Work with webcams, video files, or RTSP streams
- **3D Triangulation**: Stereo camera calibration and 3D pose reconstruction
- **Biomechanical Metrics**: Automatic calculation of joint angles, segment lengths, and body positions
- **Video Recording**: Record sessions with pose overlay
- **CSV Logging**: Export metrics data for further analysis
- **Player Tagging**: Tag and identify athletes across sessions

## Installation

### Requirements

- Python 3.9+
- PyQt6
- OpenCV
- Ultralytics (YOLO11)
- NumPy
- Matplotlib
- PyYAML
- Pandas

### Setup

```bash
# Clone or extract the project
cd pose-tracker

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the application
python main.py
```

### Apple Silicon Optimization

For best performance on M1/M2/M3/M4 Macs:

```python
# The app defaults to MPS (Metal Performance Shaders)
# For even better performance, export to CoreML:
from ultralytics import YOLO
model = YOLO('yolo11x-pose.pt')
model.export(format='coreml', nms=True)
```

Then update `config.yaml` to use the `.mlpackage` model.

## Usage

### Basic Usage

```bash
# Launch with default settings
python main.py

# Open specific video file
python main.py --source path/to/video.mp4

# Use webcam
python main.py --source 0

# Use RTSP stream
python main.py --source "rtsp://admin:pass@192.168.0.200:554/stream1"
```

### Keyboard Shortcuts

| Key | Action |
|-----|--------|
| Ctrl+O | Open video file |
| Space | Play/Pause |
| Q | Quit |

### Interface

- **Toolbar**: Quick access to cameras, settings, and features
- **Video Panel**: Live view with pose overlay
- **Metrics Panel**: Real-time biomechanical measurements
- **3D View**: Interactive 3D skeleton visualization (stereo mode)

## Stereo Calibration

For 3D pose reconstruction, you need to calibrate your stereo camera setup:

### Requirements

- Checkerboard pattern: 9x6 inner corners (10 columns × 7 rows of squares)
- Recommended square size: 25mm
- Print on rigid, flat surface

### Calibration Process

```bash
# Live capture mode
python calibrate_stereo.py --cam1 0 --cam2 1

# With RTSP cameras
python calibrate_stereo.py \
    --cam1 "rtsp://admin:pass@192.168.0.200:554/stream1" \
    --cam2 "rtsp://admin:pass@192.168.0.201:554/stream1"

# From pre-captured images
python calibrate_stereo.py \
    --images1 cam1/*.jpg \
    --images2 cam2/*.jpg
```

### Calibration Tips

1. Capture 15-25 image pairs
2. Cover all areas of both camera views
3. Vary the checkerboard angle and distance
4. Keep the board still when capturing
5. Aim for stereo RMS error < 0.5

## Configuration

Edit `config.yaml` to customize settings:

```yaml
camera:
  source: "0"              # Primary camera
  source_2: null           # Secondary camera for stereo
  width: 1280
  height: 720
  fps: 30

pose:
  model: "yolov8n-pose.pt" # n, s, m, l, x
  confidence: 0.5
  device: "mps"            # mps, cuda, cpu

stereo:
  enabled: false
  calibration_file: null   # Path to stereo_calibration.json

recording:
  output_path: "data/recordings/"
  include_overlay: true

metrics:
  log_to_csv: true
```

## Biomechanical Metrics

### 2D Metrics (Single Camera)

| Metric | Description |
|--------|-------------|
| Thigh Angle | Angle of thigh from vertical |
| Shank Angle | Angle of lower leg from vertical |
| Knee Angle | Flexion angle at knee joint |
| Hip Angle | Flexion angle at hip joint |
| Elbow Angle | Flexion angle at elbow joint |
| Trunk Lean | Forward/backward lean of torso |
| Hip-Ankle Distance | Vertical distance from hip to ankle |

### 3D Metrics (Stereo Mode)

| Metric | Description |
|--------|-------------|
| 3D Thigh Angle | True 3D angle from vertical |
| Knee Valgus/Varus | Frontal plane knee deviation |
| Shoulder Width | Actual shoulder width in meters |
| Hip Width | Actual hip width in meters |

## Project Structure

```
pose-tracker/
├── main.py                 # Application entry point
├── config.yaml             # Configuration file
├── requirements.txt        # Python dependencies
├── calibrate_stereo.py     # Stereo calibration tool
├── core/
│   ├── __init__.py
│   ├── camera.py           # Video source handling
│   ├── pose.py             # Pose estimation
│   ├── metrics.py          # Biomechanical calculations
│   ├── triangulation.py    # 3D reconstruction
│   ├── recorder.py         # Video/CSV recording
│   └── player_tagger.py    # Player identification
├── ui/
│   ├── __init__.py
│   ├── main_window.py      # Main application window
│   ├── video_widget.py     # Video display widget
│   ├── metrics_panel.py    # Metrics display panel
│   ├── settings_dialog.py  # Settings configuration
│   ├── player_dialog.py    # Player management
│   ├── view_3d_widget.py   # 3D visualization
│   └── dual_view_widget.py # Dual camera display
└── data/
    ├── calibrations/       # Stereo calibration files
    ├── recordings/         # Recorded sessions
    └── headshots/          # Player database
```

## Extending

### Adding New Metrics

Edit `core/metrics.py` to add custom calculations:

```python
def calculate(self, kpts: Keypoints, ...) -> BiomechanicsMetrics:
    # Add your custom metric
    if kpts.is_valid(Keypoints.L_KNEE) and kpts.is_valid(Keypoints.L_ANKLE):
        metrics.custom_metric = self._your_calculation(...)
```

### Custom Visualization

The 3D view in `ui/view_3d_widget.py` uses matplotlib and can be customized for different visualization needs.

## Troubleshooting

### Camera Not Opening

- Check camera permissions (macOS: System Preferences > Security & Privacy > Camera)
- Verify RTSP URL format and network connectivity
- Try different camera index (0, 1, 2...)

### Low FPS

- Use smaller model: `yolov8n-pose.pt` instead of `yolov8m-pose.pt`
- Reduce resolution in config
- Export to CoreML for Apple Silicon
- Check CPU/GPU utilization

### Stereo Calibration Issues

- Ensure checkerboard is completely visible in both cameras
- Use consistent lighting
- Verify checkerboard dimensions match parameters
- Capture more image pairs from varied angles

## License

MIT License - See LICENSE file for details.

## Acknowledgments

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- [OpenCV](https://opencv.org/)
- [PyQt6](https://www.riverbankcomputing.com/software/pyqt/)
