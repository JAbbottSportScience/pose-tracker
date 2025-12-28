#!/usr/bin/env python3
"""
Stereo Camera Calibration Tool

Captures synchronized checkerboard images from two cameras and computes
stereo calibration parameters for 3D triangulation.

Usage:
    python calibrate_stereo.py --cam1 0 --cam2 1
    python calibrate_stereo.py --cam1 rtsp://... --cam2 rtsp://...
    python calibrate_stereo.py --images1 cam1/*.jpg --images2 cam2/*.jpg
"""

import cv2
import numpy as np
import json
import argparse
from pathlib import Path
from datetime import datetime
from typing import List, Tuple, Optional


class StereoCalibrator:
    """Stereo camera calibration using checkerboard pattern"""
    
    def __init__(self, checkerboard: Tuple[int, int] = (9, 6), 
                 square_size: float = 0.025):
        """
        Initialize calibrator.
        
        Args:
            checkerboard: Inner corners (columns, rows) - e.g., (9, 6) for 10x7 squares
            square_size: Size of each square in meters
        """
        self.checkerboard = checkerboard
        self.square_size = square_size
        
        # Termination criteria for corner refinement
        self.criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        
        # Prepare object points (3D points in real world space)
        self.objp = np.zeros((checkerboard[0] * checkerboard[1], 3), np.float32)
        self.objp[:, :2] = np.mgrid[0:checkerboard[0], 0:checkerboard[1]].T.reshape(-1, 2)
        self.objp *= square_size
        
        # Storage for calibration points
        self.obj_points: List[np.ndarray] = []
        self.img_points_1: List[np.ndarray] = []
        self.img_points_2: List[np.ndarray] = []
        self.img_size: Optional[Tuple[int, int]] = None
    
    def find_corners(self, img1: np.ndarray, img2: np.ndarray) -> Tuple[bool, Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Find checkerboard corners in both images.
        
        Returns:
            (success, corners1, corners2)
        """
        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY) if len(img1.shape) == 3 else img1
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY) if len(img2.shape) == 3 else img2
        
        if self.img_size is None:
            self.img_size = (gray1.shape[1], gray1.shape[0])
        
        found1, corners1 = cv2.findChessboardCorners(gray1, self.checkerboard, None)
        found2, corners2 = cv2.findChessboardCorners(gray2, self.checkerboard, None)
        
        if found1 and found2:
            corners1 = cv2.cornerSubPix(gray1, corners1, (11, 11), (-1, -1), self.criteria)
            corners2 = cv2.cornerSubPix(gray2, corners2, (11, 11), (-1, -1), self.criteria)
            return True, corners1, corners2
        
        return False, None, None
    
    def add_points(self, corners1: np.ndarray, corners2: np.ndarray):
        """Add a valid point pair to calibration data"""
        self.obj_points.append(self.objp)
        self.img_points_1.append(corners1)
        self.img_points_2.append(corners2)
    
    def calibrate(self) -> dict:
        """
        Run stereo calibration.
        
        Returns:
            Calibration dictionary
        """
        if len(self.obj_points) < 10:
            raise ValueError(f"Need at least 10 image pairs, have {len(self.obj_points)}")
        
        print(f"Calibrating with {len(self.obj_points)} image pairs...")
        
        # Individual camera calibration
        print("Calibrating camera 1...")
        ret1, mtx1, dist1, rvecs1, tvecs1 = cv2.calibrateCamera(
            self.obj_points, self.img_points_1, self.img_size, None, None
        )
        print(f"  RMS error: {ret1:.4f}")
        
        print("Calibrating camera 2...")
        ret2, mtx2, dist2, rvecs2, tvecs2 = cv2.calibrateCamera(
            self.obj_points, self.img_points_2, self.img_size, None, None
        )
        print(f"  RMS error: {ret2:.4f}")
        
        # Stereo calibration
        print("Running stereo calibration...")
        flags = cv2.CALIB_FIX_INTRINSIC
        ret_stereo, mtx1, dist1, mtx2, dist2, R, T, E, F = cv2.stereoCalibrate(
            self.obj_points, self.img_points_1, self.img_points_2,
            mtx1, dist1, mtx2, dist2,
            self.img_size, criteria=self.criteria, flags=flags
        )
        print(f"  Stereo RMS error: {ret_stereo:.4f}")
        
        # Stereo rectification
        print("Computing rectification...")
        R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
            mtx1, dist1, mtx2, dist2, self.img_size, R, T, alpha=0
        )
        
        # Calculate baseline
        baseline = np.linalg.norm(T)
        print(f"  Baseline: {baseline*100:.1f} cm")
        
        calibration = {
            'img_size': list(self.img_size),
            'checkerboard': list(self.checkerboard),
            'square_size': self.square_size,
            'num_images': len(self.obj_points),
            'mtx1': mtx1.tolist(),
            'dist1': dist1.tolist(),
            'mtx2': mtx2.tolist(),
            'dist2': dist2.tolist(),
            'R': R.tolist(),
            'T': T.tolist(),
            'E': E.tolist(),
            'F': F.tolist(),
            'R1': R1.tolist(),
            'R2': R2.tolist(),
            'P1': P1.tolist(),
            'P2': P2.tolist(),
            'Q': Q.tolist(),
            'roi1': list(roi1),
            'roi2': list(roi2),
            'baseline_m': float(baseline),
            'cam1_rms': float(ret1),
            'cam2_rms': float(ret2),
            'stereo_rms': float(ret_stereo),
            'calibration_date': datetime.now().isoformat()
        }
        
        return calibration
    
    def save_calibration(self, calibration: dict, output_path: str):
        """Save calibration to JSON file"""
        with open(output_path, 'w') as f:
            json.dump(calibration, f, indent=2)
        print(f"Calibration saved to: {output_path}")
    
    def draw_corners(self, img: np.ndarray, corners: np.ndarray, found: bool) -> np.ndarray:
        """Draw detected corners on image"""
        vis = img.copy()
        cv2.drawChessboardCorners(vis, self.checkerboard, corners, found)
        return vis


def capture_live(cam1_source, cam2_source, calibrator: StereoCalibrator, 
                 output_dir: str = "calibration_images", 
                 target_pairs: int = 20):
    """
    Live capture mode for collecting calibration images.
    
    Controls:
        SPACE - Capture current frame pair
        Q - Quit and calibrate
        ESC - Quit without calibrating
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Open cameras
    cap1 = cv2.VideoCapture(cam1_source)
    cap2 = cv2.VideoCapture(cam2_source)
    
    if not cap1.isOpened() or not cap2.isOpened():
        print("Error: Could not open cameras")
        return None
    
    # Set resolution
    for cap in [cap1, cap2]:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    print("\n=== Stereo Calibration Capture ===")
    print(f"Checkerboard: {calibrator.checkerboard[0]}x{calibrator.checkerboard[1]} inner corners")
    print(f"Square size: {calibrator.square_size*1000:.0f}mm")
    print(f"Target: {target_pairs} image pairs")
    print("\nControls:")
    print("  SPACE - Capture when checkerboard detected in both views")
    print("  Q     - Quit and run calibration")
    print("  ESC   - Quit without calibrating")
    print("\nTips:")
    print("  - Hold checkerboard at various angles and distances")
    print("  - Cover all areas of both camera views")
    print("  - Keep the board still when capturing")
    print()
    
    pair_count = 0
    
    while True:
        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()
        
        if not ret1 or not ret2:
            continue
        
        # Find corners
        found, corners1, corners2 = calibrator.find_corners(frame1, frame2)
        
        # Draw corners
        vis1 = calibrator.draw_corners(frame1, corners1, found) if found else frame1.copy()
        vis2 = calibrator.draw_corners(frame2, corners2, found) if found else frame2.copy()
        
        # Add status text
        status = f"Pairs: {pair_count}/{target_pairs}"
        if found:
            status += " - READY (press SPACE)"
            color = (0, 255, 0)
        else:
            status += " - Searching..."
            color = (0, 0, 255)
        
        cv2.putText(vis1, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv2.putText(vis2, "Camera 2", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        
        # Stack views
        combined = np.hstack([vis1, vis2])
        combined = cv2.resize(combined, (1600, 450))
        
        cv2.imshow("Stereo Calibration", combined)
        
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord(' ') and found:
            # Capture
            calibrator.add_points(corners1, corners2)
            pair_count += 1
            
            # Save images
            cv2.imwrite(str(output_path / f"cam1_{pair_count:03d}.jpg"), frame1)
            cv2.imwrite(str(output_path / f"cam2_{pair_count:03d}.jpg"), frame2)
            
            print(f"Captured pair {pair_count}/{target_pairs}")
            
            if pair_count >= target_pairs:
                print("\nTarget reached! Press Q to calibrate or continue capturing.")
        
        elif key == ord('q'):
            if pair_count >= 10:
                break
            else:
                print(f"Need at least 10 pairs (have {pair_count})")
        
        elif key == 27:  # ESC
            print("Cancelled")
            cap1.release()
            cap2.release()
            cv2.destroyAllWindows()
            return None
    
    cap1.release()
    cap2.release()
    cv2.destroyAllWindows()
    
    # Run calibration
    return calibrator.calibrate()


def calibrate_from_images(images1: List[str], images2: List[str], 
                          calibrator: StereoCalibrator) -> dict:
    """Calibrate from pre-captured image files"""
    
    if len(images1) != len(images2):
        raise ValueError("Number of images must match")
    
    print(f"Processing {len(images1)} image pairs...")
    
    for i, (img1_path, img2_path) in enumerate(zip(images1, images2)):
        img1 = cv2.imread(img1_path)
        img2 = cv2.imread(img2_path)
        
        if img1 is None or img2 is None:
            print(f"  Skipping {img1_path} / {img2_path} - could not read")
            continue
        
        found, corners1, corners2 = calibrator.find_corners(img1, img2)
        
        if found:
            calibrator.add_points(corners1, corners2)
            print(f"  {i+1}: Found corners")
        else:
            print(f"  {i+1}: No corners detected")
    
    return calibrator.calibrate()


def main():
    parser = argparse.ArgumentParser(description='Stereo Camera Calibration')
    parser.add_argument('--cam1', default=0, help='Camera 1 source (device index or RTSP URL)')
    parser.add_argument('--cam2', default=1, help='Camera 2 source (device index or RTSP URL)')
    parser.add_argument('--images1', nargs='*', help='Camera 1 images (for offline calibration)')
    parser.add_argument('--images2', nargs='*', help='Camera 2 images (for offline calibration)')
    parser.add_argument('--checkerboard', default='9,6', help='Inner corners: cols,rows')
    parser.add_argument('--square-size', type=float, default=0.025, help='Square size in meters')
    parser.add_argument('--output', '-o', default='stereo_calibration.json', help='Output file')
    parser.add_argument('--target', type=int, default=20, help='Target number of image pairs')
    
    args = parser.parse_args()
    
    # Parse checkerboard size
    cols, rows = map(int, args.checkerboard.split(','))
    
    calibrator = StereoCalibrator(
        checkerboard=(cols, rows),
        square_size=args.square_size
    )
    
    if args.images1 and args.images2:
        # Offline calibration from images
        calibration = calibrate_from_images(args.images1, args.images2, calibrator)
    else:
        # Live capture
        try:
            cam1 = int(args.cam1)
        except ValueError:
            cam1 = args.cam1
        
        try:
            cam2 = int(args.cam2)
        except ValueError:
            cam2 = args.cam2
        
        calibration = capture_live(cam1, cam2, calibrator, target_pairs=args.target)
    
    if calibration:
        calibrator.save_calibration(calibration, args.output)
        
        print("\n=== Calibration Summary ===")
        print(f"Image size: {calibration['img_size']}")
        print(f"Baseline: {calibration['baseline_m']*100:.1f} cm")
        print(f"Camera 1 RMS: {calibration['cam1_rms']:.4f}")
        print(f"Camera 2 RMS: {calibration['cam2_rms']:.4f}")
        print(f"Stereo RMS: {calibration['stereo_rms']:.4f}")
        print(f"\nSaved to: {args.output}")


if __name__ == '__main__':
    main()
