#!/usr/bin/env python3
"""
Test script to verify camera access with different methods
"""

import cv2
import time
import os

def test_camera_by_path(device_path):
    """Test camera using device path"""
    print(f"\n{'='*50}")
    print(f"Testing Camera: {device_path}")
    print('='*50)
    
    if not os.path.exists(device_path):
        print(f"❌ Device {device_path} does not exist")
        return False
    
    # Method 1: Default backend (no specification)
    print("\n1. Testing with default backend (automatic)...")
    cap = cv2.VideoCapture(device_path)
    
    if cap.isOpened():
        print(f"✅ Opened with default backend")
        ret, frame = cap.read()
        if ret:
            print(f"✅ Successfully read frame: {frame.shape}")
            
            # Show properties
            width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
            height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
            fps = cap.get(cv2.CAP_PROP_FPS)
            print(f"📊 Properties: {int(width)}x{int(height)} @ {fps}fps")
        else:
            print(f"❌ Opened but cannot read frames")
        cap.release()
        time.sleep(0.3)  # Delay after release
        return ret
    else:
        print(f"❌ Cannot open with default backend")
        return False
    
    # Note: We removed V4L2 and GStreamer tests since they fail on your system

def test_simultaneous():
    """Test opening both cameras simultaneously"""
    print(f"\n{'='*50}")
    print("Testing Simultaneous Camera Access (Default Backend)")
    print('='*50)
    
    print("Opening camera 0...")
    cap0 = cv2.VideoCapture("/dev/video0")
    
    if not cap0.isOpened():
        print("❌ Failed to open camera 0")
        return False
    
    print("✅ Camera 0 opened")
    
    # Critical: Wait before opening second camera
    print("⏳ Waiting 1 second before opening camera 1...")
    time.sleep(1.0)
    
    print("Opening camera 1...")
    cap1 = cv2.VideoCapture("/dev/video1")
    
    if not cap1.isOpened():
        print("❌ Failed to open camera 1")
        cap0.release()
        return False
    
    print("✅ Camera 1 opened")
    
    # Try reading from both
    print("\n📸 Testing frame capture from both cameras...")
    
    ret0, frame0 = cap0.read()
    time.sleep(0.1)  # Small delay between reads
    ret1, frame1 = cap1.read()
    
    if ret0 and ret1:
        print(f"✅ Both cameras reading frames successfully!")
        print(f"   Camera 0: {frame0.shape}")
        print(f"   Camera 1: {frame1.shape}")
        success = True
    else:
        print(f"❌ One or both cameras failed to read frames")
        print(f"   Camera 0: {ret0}")
        print(f"   Camera 1: {ret1}")
        success = False
    
    # Test multiple reads
    if success:
        print("\n🔄 Testing continuous capture (5 frames)...")
        for i in range(5):
            ret0, frame0 = cap0.read()
            ret1, frame1 = cap1.read()
            
            if ret0 and ret1:
                print(f"  Frame {i+1}: ✅")
            else:
                print(f"  Frame {i+1}: ❌ (L:{ret0}, R:{ret1})")
            
            time.sleep(0.1)
    
    cap0.release()
    cap1.release()
    
    return success

def test_sequential_open_close():
    """Test opening cameras sequentially with proper cleanup"""
    print(f"\n{'='*50}")
    print("Testing Sequential Open/Close")
    print('='*50)
    
    for attempt in range(3):
        print(f"\nAttempt {attempt + 1}/3:")
        
        # Open camera 0
        cap0 = cv2.VideoCapture("/dev/video0")
        if cap0.isOpened():
            ret, _ = cap0.read()
            print(f"  Camera 0: {'✅' if ret else '❌'}")
            cap0.release()
            time.sleep(0.5)
        
        # Open camera 1
        cap1 = cv2.VideoCapture("/dev/video1")
        if cap1.isOpened():
            ret, _ = cap1.read()
            print(f"  Camera 1: {'✅' if ret else '❌'}")
            cap1.release()
            time.sleep(0.5)

if __name__ == "__main__":
    print("🎥 Jetson Camera Test Script (Simplified)")
    print("="*50)
    print("Using default OpenCV backend (no V4L2/GStreamer)")
    print()
    
    # Test individual cameras
    cam0_ok = test_camera_by_path("/dev/video0")
    time.sleep(0.5)
    
    cam1_ok = test_camera_by_path("/dev/video1")
    time.sleep(0.5)
    
    # Test sequential access
    test_sequential_open_close()
    time.sleep(0.5)
    
    # Test simultaneous access (the real test)
    if cam0_ok and cam1_ok:
        simultaneous_ok = test_simultaneous()
    else:
        print("\n⚠️  Skipping simultaneous test (individual cameras failed)")
        simultaneous_ok = False
    
    print("\n" + "="*50)
    print("📊 Test Results:")
    print(f"  Camera 0: {'✅ Working' if cam0_ok else '❌ Failed'}")
    print(f"  Camera 1: {'✅ Working' if cam1_ok else '❌ Failed'}")
    print(f"  Simultaneous: {'✅ Working' if simultaneous_ok else '❌ Failed'}")
    
    if simultaneous_ok:
        print("\n🎉 SUCCESS! Both cameras work together")
        print("   You can now run main.py")
    else:
        print("\n⚠️  Simultaneous access failed")
        print("   Try: sudo rmmod <camera_module> && sudo modprobe <camera_module>")
