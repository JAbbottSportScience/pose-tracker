#!/usr/bin/env python3
"""
Pose Tracker - Real-time skeletal pose estimation and biomechanical analysis

Features:
- Single and dual camera support
- 2D and 3D pose estimation
- Real-time biomechanical metrics
- Video recording with overlay
- CSV logging of metrics
- Player tagging and management
- Stereo calibration and triangulation

Usage:
    python main.py [--config CONFIG_FILE]
"""

import sys
import argparse
import yaml
from pathlib import Path
from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QPalette, QColor

from ui import MainWindow


def load_config(config_path: str = 'config.yaml') -> dict:
    """Load configuration from YAML file with defaults"""
    default_config = {
        'camera': {
            'source': '0',
            'source_2': None,
            'width': 1280,
            'height': 720,
            'fps': 30
        },
        'pose': {
            'model': 'yolo11x-pose.pt',
            'confidence': 0.5,
            'device': 'mps'  # mps for Apple Silicon, cuda for NVIDIA, cpu for fallback
        },
        'stereo': {
            'enabled': False,
            'calibration_file': None
        },
        'metrics': {
            'show_angles': True,
            'show_distances': True,
            'log_to_csv': True,
            'csv_path': 'data/recordings/'
        },
        'recording': {
            'output_path': 'data/recordings/',
            'format': 'mp4',
            'include_overlay': True
        },
        'ui': {
            'theme': 'dark',
            'show_skeleton': True,
            'show_keypoints': True,
            'show_3d_view': False
        },
        'players': {
            'database_path': 'data/headshots/'
        }
    }
    
    config_file = Path(config_path)
    if config_file.exists():
        try:
            with open(config_file, 'r') as f:
                loaded = yaml.safe_load(f)
                if loaded:
                    # Deep merge with defaults
                    for section in default_config:
                        if section in loaded and isinstance(loaded[section], dict):
                            default_config[section].update(loaded[section])
                        elif section in loaded:
                            default_config[section] = loaded[section]
        except Exception as e:
            print(f"Warning: Could not load config file: {e}")
    
    return default_config


def setup_dark_palette() -> QPalette:
    """Create dark color palette for application"""
    palette = QPalette()
    
    # Window colors
    palette.setColor(QPalette.ColorRole.Window, QColor(30, 30, 30))
    palette.setColor(QPalette.ColorRole.WindowText, QColor(200, 200, 200))
    
    # Base colors (for text inputs, lists, etc.)
    palette.setColor(QPalette.ColorRole.Base, QColor(45, 45, 45))
    palette.setColor(QPalette.ColorRole.AlternateBase, QColor(60, 60, 60))
    palette.setColor(QPalette.ColorRole.Text, QColor(200, 200, 200))
    
    # Button colors
    palette.setColor(QPalette.ColorRole.Button, QColor(60, 60, 60))
    palette.setColor(QPalette.ColorRole.ButtonText, QColor(200, 200, 200))
    
    # Selection colors
    palette.setColor(QPalette.ColorRole.Highlight, QColor(0, 120, 215))
    palette.setColor(QPalette.ColorRole.HighlightedText, QColor(255, 255, 255))
    
    # Tooltip colors
    palette.setColor(QPalette.ColorRole.ToolTipBase, QColor(45, 45, 45))
    palette.setColor(QPalette.ColorRole.ToolTipText, QColor(200, 200, 200))
    
    # Link colors
    palette.setColor(QPalette.ColorRole.Link, QColor(0, 170, 255))
    palette.setColor(QPalette.ColorRole.LinkVisited, QColor(150, 120, 200))
    
    # Disabled colors
    palette.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.WindowText, QColor(100, 100, 100))
    palette.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.Text, QColor(100, 100, 100))
    palette.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.ButtonText, QColor(100, 100, 100))
    
    return palette


def create_directories(config: dict):
    """Create required directories"""
    dirs = [
        config.get('recording', {}).get('output_path', 'data/recordings'),
        config.get('players', {}).get('database_path', 'data/headshots'),
        'data/calibrations'
    ]
    
    for dir_path in dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='Pose Tracker')
    parser.add_argument('--config', '-c', default='config.yaml',
                        help='Path to configuration file')
    parser.add_argument('--source', '-s', default=None,
                        help='Video source (file path, camera index, or RTSP URL)')
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Override source if provided
    if args.source:
        try:
            config['camera']['source'] = int(args.source)
        except ValueError:
            config['camera']['source'] = args.source
    
    # Create required directories
    create_directories(config)
    
    # Setup Qt application
    QApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
    )
    
    app = QApplication(sys.argv)
    app.setApplicationName("Pose Tracker")
    app.setOrganizationName("SportScience")
    app.setStyle("Fusion")
    
    # Apply dark palette
    app.setPalette(setup_dark_palette())
    
    # Create and show main window
    window = MainWindow(config)
    window.show()
    
    # Auto-start source if provided via command line
    if args.source:
        window._start_source(config['camera']['source'])
    
    sys.exit(app.exec())


if __name__ == '__main__':
    main()
