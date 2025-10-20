#!/usr/bin/env python3
"""
Test script - Use INTEGER indices instead of device paths for Jetson
"""

import cv2
import time

def test_camera_by_index(index):
    """Test camera using INTEGER index (not device path)"""
    print(f"\n{'='*50}")
    print(f"Testing Camera Index: {index}")
    print('='*50)
    
    print(f"Opening camera {index}...")
    cap = cv2.VideoCapture(index)
    
    if cap.isOpened():
        print(f"✅ Opened camera {index}")
        ret, frame = cap.read()
        if ret:
            print(f"✅ Successfully read frame: {frame.shape}")
            
            # Show properties
            width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
            height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
            fps = cap.get(cv2.CAP_PROP_FPS)
            print(f"📊 Properties: {int(width)}x{int(height)} @ {fps}fps")
            
            cap.release()
            time.sleep(0.3)
            return True
        else:
            print(f"❌ Opened but cannot read frames (may be metadata device)")
            cap.release()
            return False
    else:
        print(f"❌ Cannot open camera {index}")
        return False

def test_simultaneous():
    """Test opening both cameras simultaneously using indices"""
    print(f"\n{'='*50}")
    print("Testing Simultaneous Camera Access (Integer Indices)")
    print('='*50)
    
    print("Opening camera 0...")
    cap0 = cv2.VideoCapture(0)
    
    if not cap0.isOpened():
        print("❌ Failed to open camera 0")
        return False
    
    print("✅ Camera 0 opened")
    
    # Critical: Wait before opening second camera
    print("⏳ Waiting 1 second before opening camera 1...")
    time.sleep(1.0)
    
    print("Opening camera 1...")
    cap1 = cv2.VideoCapture(1)
    
    if not cap1.isOpened():
        print("❌ Failed to open camera 1")
        cap0.release()
        return False
    
    print("✅ Camera 1 opened")
    
    # Try reading from both
    print("\n📸 Testing frame capture from both cameras...")
    
    ret0, frame0 = cap0.read()
    time.sleep(0.1)
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
                print(f"  Frame {i+1}: ❌ (Cam0:{ret0}, Cam1:{ret1})")
            
            time.sleep(0.1)
    
    cap0.release()
    cap1.release()
    
    return success

def detect_all_cameras():
    """Detect all available cameras"""
    print(f"\n{'='*50}")
    print("Detecting All Available Cameras")
    print('='*50)
    
    available = []
    
    for idx in range(10):
        try:
            cap = cv2.VideoCapture(idx)
            if cap.isOpened():
                ret, frame = cap.read()
                cap.release()
                
                if ret and frame is not None:
                    available.append(idx)
                    print(f"  ✅ Camera {idx} - Working ({frame.shape})")
                else:
                    print(f"  ⚠️  Camera {idx} - Opens but no frames")
            
            time.sleep(0.2)
        except Exception as e:
            pass
    
    print(f"\n📊 Summary: Found {len(available)} working camera(s): {available}")
    return available

if __name__ == "__main__":
    print("🎥 Jetson Camera Test Script (Integer Indices)")
    print("="*50)
    print("Using cv2.VideoCapture(INTEGER) instead of device paths")
    print()
    
    # First, detect all cameras
    available = detect_all_cameras()
    
    if len(available) == 0:
        print("\n❌ No cameras detected!")
        exit(1)
    
    if len(available) == 1:
        print("\n⚠️  Only 1 camera detected. Need 2 for stereo.")
        print("   Check if both cameras are connected.")
        exit(1)
    
    # Test individual cameras
    print(f"\n{'='*50}")
    print("Testing Individual Cameras")
    print('='*50)
    
    cam0_ok = test_camera_by_index(available[0])
    time.sleep(0.5)
    
    cam1_ok = test_camera_by_index(available[1])
    time.sleep(0.5)
    
    # Test simultaneous access
    if cam0_ok and cam1_ok:
        simultaneous_ok = test_simultaneous()
    else:
        print("\n⚠️  Skipping simultaneous test (individual cameras failed)")
        simultaneous_ok = False
    
    # Results
    print("\n" + "="*50)
    print("📊 Test Results:")
    print(f"  Camera {available[0]}: {'✅ Working' if cam0_ok else '❌ Failed'}")
    if len(available) > 1:
        print(f"  Camera {available[1]}: {'✅ Working' if cam1_ok else '❌ Failed'}")
    print(f"  Simultaneous: {'✅ Working' if simultaneous_ok else '❌ Failed'}")
    
    if simultaneous_ok:
        print("\n🎉 SUCCESS! Both cameras work together")
        print("   You can now run main.py")
    else:
        print("\n⚠️  Issues detected:")
        print("   1. You may only have 1 physical camera")
        print("   2. /dev/video1 may be a metadata device")
        print("   3. Check: v4l2-ctl --list-devices")
