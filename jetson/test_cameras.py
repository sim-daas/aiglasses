#!/usr/bin/env python3
"""
Test script to verify camera access
"""

import cv2
import time

def test_camera(index):
    """Test a single camera"""
    print(f"\n{'='*50}")
    print(f"Testing Camera {index} (/dev/video{index})")
    print('='*50)
    
    # Try without CAP_V4L2
    print("\n1. Testing with default backend...")
    cap = cv2.VideoCapture(index)
    
    if cap.isOpened():
        print(f"✅ Camera {index} opened with default backend")
        ret, frame = cap.read()
        if ret:
            print(f"✅ Successfully read frame: {frame.shape}")
        else:
            print(f"❌ Camera opened but cannot read frames")
        cap.release()
    else:
        print(f"❌ Cannot open camera {index} with default backend")
    
    time.sleep(0.5)
    
    # Try with CAP_V4L2
    print("\n2. Testing with CAP_V4L2 backend...")
    cap = cv2.VideoCapture(index, cv2.CAP_V4L2)
    
    if cap.isOpened():
        print(f"✅ Camera {index} opened with CAP_V4L2")
        ret, frame = cap.read()
        if ret:
            print(f"✅ Successfully read frame: {frame.shape}")
            
            # Show camera properties
            width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
            height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
            fps = cap.get(cv2.CAP_PROP_FPS)
            print(f"📊 Properties: {int(width)}x{int(height)} @ {fps}fps")
        else:
            print(f"❌ Camera opened but cannot read frames")
        cap.release()
    else:
        print(f"❌ Cannot open camera {index} with CAP_V4L2")

def test_simultaneous():
    """Test opening both cameras simultaneously"""
    print(f"\n{'='*50}")
    print("Testing Simultaneous Camera Access")
    print('='*50)
    
    cap0 = cv2.VideoCapture(0, cv2.CAP_V4L2)
    time.sleep(0.5)  # Small delay
    cap1 = cv2.VideoCapture(1, cv2.CAP_V4L2)
    
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
    print("🎥 Camera Test Script")
    print("="*50)
    
    # Test individual cameras
    test_camera(0)
    test_camera(1)
    
    # Test simultaneous access
    test_simultaneous()
    
    print("\n" + "="*50)
    print("✅ Camera tests complete")
