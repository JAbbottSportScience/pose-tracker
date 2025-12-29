#!/usr/bin/env python3
"""
Camera Diagnostic Tool
Run this to identify camera issues and test your setup.
"""

import cv2
import sys
import platform
import time


def print_header(text):
    print(f"\n{'='*50}")
    print(f" {text}")
    print('='*50)


def get_available_backends():
    """List available OpenCV backends"""
    backends = [
        ("CAP_ANY", cv2.CAP_ANY),
        ("CAP_V4L2", cv2.CAP_V4L2),
        ("CAP_DSHOW", cv2.CAP_DSHOW),
        ("CAP_MSMF", cv2.CAP_MSMF),
        ("CAP_AVFOUNDATION", cv2.CAP_AVFOUNDATION),
        ("CAP_GSTREAMER", cv2.CAP_GSTREAMER),
        ("CAP_FFMPEG", cv2.CAP_FFMPEG),
    ]
    return backends


def check_system_info():
    """Print system and OpenCV info"""
    print_header("System Information")
    print(f"OS: {platform.system()} {platform.release()}")
    print(f"Python: {sys.version}")
    print(f"OpenCV: {cv2.__version__}")
    
    # Check build info for backend support
    build_info = cv2.getBuildInformation()
    
    print("\nOpenCV Backend Support:")
    backends_to_check = ["FFMPEG", "GStreamer", "V4L", "DirectShow", "AVFoundation", "MSMF"]
    for backend in backends_to_check:
        if backend in build_info:
            # Find the line with YES/NO
            for line in build_info.split('\n'):
                if backend in line and ('YES' in line or 'NO' in line):
                    status = "✓" if "YES" in line else "✗"
                    print(f"  {status} {backend}")
                    break


def find_cameras(max_index=10):
    """Scan for available cameras"""
    print_header("Scanning for Cameras")
    
    found_cameras = []
    
    system = platform.system()
    if system == "Windows":
        preferred_backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]
    elif system == "Linux":
        preferred_backends = [cv2.CAP_V4L2, cv2.CAP_ANY]
    elif system == "Darwin":
        preferred_backends = [cv2.CAP_AVFOUNDATION, cv2.CAP_ANY]
    else:
        preferred_backends = [cv2.CAP_ANY]
    
    for idx in range(max_index):
        for backend in preferred_backends:
            cap = cv2.VideoCapture(idx, backend)
            if cap.isOpened():
                # Try to read a frame to confirm it works
                ret, frame = cap.read()
                if ret and frame is not None:
                    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    fps = cap.get(cv2.CAP_PROP_FPS)
                    backend_name = [name for name, val in get_available_backends() if val == backend][0]
                    
                    print(f"✓ Camera {idx}: {w}x{h} @ {fps:.1f}fps (backend: {backend_name})")
                    found_cameras.append((idx, backend, w, h))
                    cap.release()
                    break  # Found working backend, move to next index
                cap.release()
    
    if not found_cameras:
        print("✗ No cameras found!")
        print("\nTroubleshooting:")
        if platform.system() == "Linux":
            print("  - Check permissions: sudo usermod -a -G video $USER")
            print("  - Then log out and back in")
            print("  - List devices: ls -la /dev/video*")
            print("  - Install v4l-utils: sudo apt install v4l-utils")
            print("  - List cameras: v4l2-ctl --list-devices")
        elif platform.system() == "Windows":
            print("  - Check Device Manager for camera")
            print("  - Make sure camera isn't in use by another app")
            print("  - Try unplugging and replugging USB cameras")
        elif platform.system() == "Darwin":
            print("  - Grant camera permissions in System Preferences > Privacy")
            print("  - Check if camera works in Photo Booth app")
    
    return found_cameras


def test_camera_detailed(index=0, backend=None, duration=10):
    """Detailed test of a specific camera with live display"""
    print_header(f"Testing Camera {index}")
    
    if backend is None:
        system = platform.system()
        if system == "Windows":
            backend = cv2.CAP_DSHOW
        elif system == "Linux":
            backend = cv2.CAP_V4L2
        else:
            backend = cv2.CAP_ANY
    
    print(f"Opening camera {index}...")
    cap = cv2.VideoCapture(index, backend)
    
    if not cap.isOpened():
        print(f"✗ Failed to open camera {index} with backend {backend}")
        print("  Trying fallback to CAP_ANY...")
        cap = cv2.VideoCapture(index, cv2.CAP_ANY)
        if not cap.isOpened():
            print("✗ Fallback also failed")
            return False
    
    # Configure camera
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    
    # Read actual settings
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    print(f"✓ Camera opened: {w}x{h} @ {fps:.1f}fps")
    
    # Warm up - discard first few frames
    print("Warming up camera...")
    for _ in range(10):
        cap.read()
    
    print(f"\nShowing live feed for {duration} seconds...")
    print("Press 'q' to quit early, 's' to save a frame\n")
    
    frame_count = 0
    start_time = time.time()
    last_fps_time = start_time
    fps_frame_count = 0
    current_fps = 0
    
    try:
        while True:
            ret, frame = cap.read()
            
            if not ret:
                print("✗ Failed to read frame")
                time.sleep(0.1)
                continue
            
            frame_count += 1
            fps_frame_count += 1
            
            # Calculate FPS every second
            now = time.time()
            if now - last_fps_time >= 1.0:
                current_fps = fps_frame_count / (now - last_fps_time)
                fps_frame_count = 0
                last_fps_time = now
            
            # Add info overlay
            info = f"Frame: {frame_count} | FPS: {current_fps:.1f} | Size: {frame.shape[1]}x{frame.shape[0]}"
            cv2.putText(frame, info, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, "Press 'q' to quit, 's' to save", (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
            
            # Show frame
            cv2.imshow(f"Camera {index} Test", frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("Quit requested")
                break
            elif key == ord('s'):
                filename = f"camera_{index}_frame_{frame_count}.jpg"
                cv2.imwrite(filename, frame)
                print(f"Saved: {filename}")
            
            # Check duration
            if time.time() - start_time >= duration:
                break
                
    except KeyboardInterrupt:
        print("\nInterrupted")
    finally:
        elapsed = time.time() - start_time
        avg_fps = frame_count / elapsed if elapsed > 0 else 0
        
        print(f"\nResults:")
        print(f"  Total frames: {frame_count}")
        print(f"  Duration: {elapsed:.1f}s")
        print(f"  Average FPS: {avg_fps:.1f}")
        
        cap.release()
        cv2.destroyAllWindows()
    
    return True


def test_dual_cameras(idx1=0, idx2=1, duration=10):
    """Test dual camera setup"""
    print_header(f"Testing Dual Cameras {idx1} and {idx2}")
    
    system = platform.system()
    if system == "Windows":
        backend = cv2.CAP_DSHOW
    elif system == "Linux":
        backend = cv2.CAP_V4L2
    else:
        backend = cv2.CAP_ANY
    
    cap1 = cv2.VideoCapture(idx1, backend)
    cap2 = cv2.VideoCapture(idx2, backend)
    
    if not cap1.isOpened():
        print(f"✗ Failed to open camera {idx1}")
        return False
    if not cap2.isOpened():
        print(f"✗ Failed to open camera {idx2}")
        cap1.release()
        return False
    
    # Configure both cameras
    for cap in [cap1, cap2]:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    
    print("✓ Both cameras opened")
    
    # Warm up
    for _ in range(10):
        cap1.read()
        cap2.read()
    
    print(f"Showing dual feed for {duration} seconds...")
    print("Press 'q' to quit\n")
    
    start_time = time.time()
    frame_count = 0
    
    try:
        while time.time() - start_time < duration:
            t1 = time.time()
            ret1, frame1 = cap1.read()
            t2 = time.time()
            ret2, frame2 = cap2.read()
            t3 = time.time()
            
            if not ret1 or not ret2:
                continue
            
            frame_count += 1
            sync_delta_ms = (t2 - t1) * 1000
            
            # Add labels
            cv2.putText(frame1, f"Cam {idx1}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame2, f"Cam {idx2}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame1, f"Sync: {sync_delta_ms:.1f}ms", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
            
            # Stack horizontally
            combined = cv2.hconcat([frame1, frame2])
            cv2.imshow("Dual Camera Test", combined)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    except KeyboardInterrupt:
        pass
    finally:
        cap1.release()
        cap2.release()
        cv2.destroyAllWindows()
        
        elapsed = time.time() - start_time
        print(f"\nCaptured {frame_count} frame pairs in {elapsed:.1f}s")
    
    return True


def simple_test(index=0):
    """Bare minimum camera test - no threading, no frills"""
    print_header(f"Simple Camera Test (index {index})")
    
    print("Step 1: Opening camera...")
    cap = cv2.VideoCapture(index)
    
    if not cap.isOpened():
        print("  ✗ cv2.VideoCapture() failed")
        print("  Trying with explicit backends...")
        
        for name, backend in get_available_backends():
            cap = cv2.VideoCapture(index, backend)
            if cap.isOpened():
                print(f"  ✓ Opened with {name}")
                break
            cap.release()
        else:
            print("  ✗ All backends failed")
            return False
    else:
        print("  ✓ VideoCapture created")
    
    print("\nStep 2: Reading frame...")
    ret, frame = cap.read()
    
    if not ret:
        print("  ✗ cap.read() returned False")
        cap.release()
        return False
    
    if frame is None:
        print("  ✗ Frame is None")
        cap.release()
        return False
    
    print(f"  ✓ Got frame: {frame.shape}")
    
    print("\nStep 3: Displaying frame...")
    cv2.imshow("Test Frame", frame)
    print("  Window should be visible. Press any key to continue...")
    cv2.waitKey(0)
    
    print("\nStep 4: Live feed (5 seconds)...")
    start = time.time()
    count = 0
    while time.time() - start < 5:
        ret, frame = cap.read()
        if ret:
            count += 1
            cv2.imshow("Test Frame", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    print(f"  ✓ Captured {count} frames")
    
    cap.release()
    cv2.destroyAllWindows()
    print("\n✓ Camera test passed!")
    return True


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Camera Diagnostic Tool")
    parser.add_argument("command", nargs="?", default="scan",
                       choices=["scan", "test", "simple", "dual", "info"],
                       help="Command to run")
    parser.add_argument("-c", "--camera", type=int, default=0,
                       help="Camera index (default: 0)")
    parser.add_argument("-c2", "--camera2", type=int, default=1,
                       help="Second camera index for dual mode (default: 1)")
    parser.add_argument("-t", "--time", type=int, default=10,
                       help="Test duration in seconds (default: 10)")
    
    args = parser.parse_args()
    
    if args.command == "info":
        check_system_info()
    elif args.command == "scan":
        check_system_info()
        find_cameras()
    elif args.command == "simple":
        simple_test(args.camera)
    elif args.command == "test":
        test_camera_detailed(args.camera, duration=args.time)
    elif args.command == "dual":
        test_dual_cameras(args.camera, args.camera2, duration=args.time)