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
    
    # Method 1: Direct V4L2
    print("\n1. Testing with V4L2 (device path)...")
    cap = cv2.VideoCapture(device_path, cv2.CAP_V4L2)
    
    if cap.isOpened():
        print(f"✅ Opened with V4L2")
        ret, frame = cap.read()
        if ret:
            print(f"✅ Successfully read frame: {frame.shape}")
        else:
            print(f"❌ Opened but cannot read frames")
        cap.release()
    else:
        print(f"❌ Cannot open with V4L2")
    
    time.sleep(0.5)
    
    # Method 2: GStreamer pipeline
    print("\n2. Testing with GStreamer pipeline...")
    
    gst_pipeline = (
        f"v4l2src device={device_path} ! "
        "video/x-raw, width=640, height=480, framerate=30/1 ! "
        "videoconvert ! video/x-raw, format=BGR ! appsink drop=1"
    )
    
    cap = cv2.VideoCapture(gst_pipeline, cv2.CAP_GSTREAMER)
    
    if cap.isOpened():
        print(f"✅ Opened with GStreamer")
        ret, frame = cap.read()
        if ret:
            print(f"✅ Successfully read frame: {frame.shape}")
            return True
        else:
            print(f"❌ Opened but cannot read frames")
    else:
        print(f"❌ Cannot open with GStreamer")
    
    cap.release()
    return False

def test_simultaneous():
    """Test opening both cameras simultaneously"""
    print(f"\n{'='*50}")
    print("Testing Simultaneous Camera Access")
    print('='*50)
    
    # Use GStreamer for both
    gst_pipeline_0 = (
        "v4l2src device=/dev/video0 ! "
        "video/x-raw, width=640, height=480, framerate=30/1 ! "
        "videoconvert ! video/x-raw, format=BGR ! appsink drop=1"
    )
    
    gst_pipeline_1 = (
        "v4l2src device=/dev/video1 ! "
        "video/x-raw, width=640, height=480, framerate=30/1 ! "
        "videoconvert ! video/x-raw, format=BGR ! appsink drop=1"
    )
    
    print("Opening camera 0...")
    cap0 = cv2.VideoCapture(gst_pipeline_0, cv2.CAP_GSTREAMER)
    
    time.sleep(0.5)
    
    print("Opening camera 1...")
    cap1 = cv2.VideoCapture(gst_pipeline_1, cv2.CAP_GSTREAMER)
    
    if cap0.isOpened() and cap1.isOpened():
        print("✅ Both cameras opened")
        
        ret0, frame0 = cap0.read()
        ret1, frame1 = cap1.read()
        
        if ret0 and ret1:
            print(f"✅ Both cameras reading frames")
            print(f"   Camera 0: {frame0.shape}")
            print(f"   Camera 1: {frame1.shape}")
        else:
            print(f"❌ Cameras opened but cannot read frames")
            print(f"   Camera 0: {ret0}")
            print(f"   Camera 1: {ret1}")
    else:
        print(f"❌ Failed to open both cameras")
        print(f"   Camera 0: {cap0.isOpened()}")
        print(f"   Camera 1: {cap1.isOpened()}")
    
    cap0.release()
    cap1.release()

if __name__ == "__main__":
    print("🎥 Enhanced Camera Test Script for Jetson")
    print("="*50)
    
    # Test individual cameras by path
    test_camera_by_path("/dev/video0")
    test_camera_by_path("/dev/video1")
    
    # Test simultaneous access
    test_simultaneous()
    
    print("\n" + "="*50)
    print("✅ Camera tests complete")
    print("\nRecommendation: Use GStreamer pipeline for best Jetson compatibility")
