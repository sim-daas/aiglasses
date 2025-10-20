#!/usr/bin/env python3
"""
Test stereo camera using /dev/video paths with live display
"""

import cv2
import time
import sys

def test_single_camera_display(device_path, window_name, duration=5):
    """Test single camera with live display"""
    print(f"\n{'='*50}")
    print(f"Testing: {device_path}")
    print('='*50)
    
    # Try opening WITHOUT backend specification (like in interpreter)
    print(f"Opening {device_path}...")
    cap = cv2.VideoCapture(device_path)
    
    if not cap.isOpened():
        print(f"❌ Failed to open {device_path}")
        return False
    
    print(f"✅ Opened {device_path}")
    
    # Get properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"📊 Properties: {width}x{height} @ {fps}fps")
    
    # Read and display frames
    print(f"📺 Displaying video for {duration} seconds...")
    print("   Press 'q' to quit early")
    
    start_time = time.time()
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        
        if not ret:
            print(f"❌ Failed to read frame")
            break
        
        frame_count += 1
        
        # Add info overlay
        cv2.putText(frame, f"{device_path}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"Frame: {frame_count}", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Display
        cv2.imshow(window_name, frame)
        
        # Check for quit
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("User quit")
            break
        
        # Check duration
        if time.time() - start_time > duration:
            break
    
    cap.release()
    cv2.destroyWindow(window_name)
    
    print(f"✅ Captured {frame_count} frames")
    return True

def test_stereo_camera_display(duration=10):
    """Test both cameras simultaneously with side-by-side display"""
    print(f"\n{'='*50}")
    print("Testing STEREO Camera (Both Feeds)")
    print('='*50)
    
    # Open both cameras WITHOUT backend specification
    print("Opening /dev/video0...")
    cap0 = cv2.VideoCapture("/dev/video0")
    
    if not cap0.isOpened():
        print("❌ Failed to open /dev/video0")
        return False
    
    print("✅ /dev/video0 opened")
    
    # CRITICAL: Small delay before opening second camera
    time.sleep(0.5)
    
    print("Opening /dev/video1...")
    cap1 = cv2.VideoCapture("/dev/video1")
    
    if not cap1.isOpened():
        print("❌ Failed to open /dev/video1")
        cap0.release()
        return False
    
    print("✅ /dev/video1 opened")
    
    # Get properties
    w0 = int(cap0.get(cv2.CAP_PROP_FRAME_WIDTH))
    h0 = int(cap0.get(cv2.CAP_PROP_FRAME_HEIGHT))
    w1 = int(cap1.get(cv2.CAP_PROP_FRAME_WIDTH))
    h1 = int(cap1.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"📊 Camera 0: {w0}x{h0}")
    print(f"📊 Camera 1: {w1}x{h1}")
    
    print(f"\n📺 Displaying STEREO video for {duration} seconds...")
    print("   Press 'q' to quit")
    print("   Press 's' to save a stereo frame pair")
    
    start_time = time.time()
    frame_count = 0
    
    while True:
        # Read from both cameras
        ret0, frame0 = cap0.read()
        ret1, frame1 = cap1.read()
        
        if not ret0 or not ret1:
            print(f"❌ Read failed - Camera 0: {ret0}, Camera 1: {ret1}")
            break
        
        frame_count += 1
        
        # Add labels
        cv2.putText(frame0, "LEFT (/dev/video0)", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame1, "RIGHT (/dev/video1)", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        cv2.putText(frame0, f"Frame: {frame_count}", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(frame1, f"Frame: {frame_count}", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Resize to same height if different
        if h0 != h1:
            target_height = min(h0, h1)
            frame0 = cv2.resize(frame0, (int(w0 * target_height / h0), target_height))
            frame1 = cv2.resize(frame1, (int(w1 * target_height / h1), target_height))
        
        # Combine side by side
        stereo_frame = cv2.hconcat([frame0, frame1])
        
        # Display
        cv2.imshow("Stereo Camera (Left | Right)", stereo_frame)
        
        # Handle keys
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            print("User quit")
            break
        elif key == ord('s'):
            # Save stereo pair
            timestamp = int(time.time())
            filename = f"stereo_frame_{timestamp}.jpg"
            cv2.imwrite(filename, stereo_frame)
            print(f"💾 Saved: {filename}")
        
        # Check duration
        if time.time() - start_time > duration:
            print("Duration complete")
            break
    
    # Cleanup
    cap0.release()
    cap1.release()
    cv2.destroyAllWindows()
    
    print(f"\n✅ Test complete - Captured {frame_count} stereo frame pairs")
    return True

def test_camera_properties(device_path):
    """Test and display all camera properties"""
    print(f"\n{'='*50}")
    print(f"Camera Properties: {device_path}")
    print('='*50)
    
    cap = cv2.VideoCapture(device_path)
    
    if not cap.isOpened():
        print(f"❌ Cannot open {device_path}")
        return False
    
    properties = {
        'FRAME_WIDTH': cv2.CAP_PROP_FRAME_WIDTH,
        'FRAME_HEIGHT': cv2.CAP_PROP_FRAME_HEIGHT,
        'FPS': cv2.CAP_PROP_FPS,
        'FOURCC': cv2.CAP_PROP_FOURCC,
        'BRIGHTNESS': cv2.CAP_PROP_BRIGHTNESS,
        'CONTRAST': cv2.CAP_PROP_CONTRAST,
        'SATURATION': cv2.CAP_PROP_SATURATION,
        'HUE': cv2.CAP_PROP_HUE,
        'GAIN': cv2.CAP_PROP_GAIN,
        'EXPOSURE': cv2.CAP_PROP_EXPOSURE,
        'BUFFERSIZE': cv2.CAP_PROP_BUFFERSIZE,
    }
    
    for name, prop in properties.items():
        value = cap.get(prop)
        if prop == cv2.CAP_PROP_FOURCC:
            # Decode FOURCC code
            fourcc = int(value)
            fourcc_str = "".join([chr((fourcc >> 8 * i) & 0xFF) for i in range(4)])
            print(f"  {name:15s}: {fourcc_str}")
        else:
            print(f"  {name:15s}: {value}")
    
    cap.release()
    return True

if __name__ == "__main__":
    print("🎥 Stereo Camera Test - Device Path Method")
    print("="*50)
    print("This test uses /dev/video* paths directly")
    print("(Same method that works in Python interpreter)")
    print()
    
    # Test individual cameras first
    print("\n🔍 STEP 1: Test individual cameras")
    cam0_ok = test_single_camera_display("/dev/video0", "Camera 0", duration=3)
    time.sleep(0.5)
    
    cam1_ok = test_single_camera_display("/dev/video1", "Camera 1", duration=3)
    time.sleep(0.5)
    
    # Show detailed properties
    print("\n🔍 STEP 2: Camera properties")
    test_camera_properties("/dev/video0")
    test_camera_properties("/dev/video1")
    
    # Test stereo mode
    if cam0_ok and cam1_ok:
        print("\n🔍 STEP 3: Test stereo mode (both cameras)")
        stereo_ok = test_stereo_camera_display(duration=10)
    else:
        print("\n⚠️  Skipping stereo test (individual cameras failed)")
        stereo_ok = False
    
    # Summary
    print("\n" + "="*50)
    print("📊 TEST SUMMARY")
    print("="*50)
    print(f"  /dev/video0: {'✅ PASS' if cam0_ok else '❌ FAIL'}")
    print(f"  /dev/video1: {'✅ PASS' if cam1_ok else '❌ FAIL'}")
    print(f"  Stereo Mode: {'✅ PASS' if stereo_ok else '❌ FAIL'}")
    
    if stereo_ok:
        print("\n🎉 SUCCESS! Stereo camera working!")
        print("   Both /dev/video0 and /dev/video1 are functional")
        print("   You can now proceed with main.py")
    else:
        print("\n❌ STEREO TEST FAILED")
        print("\nPossible issues:")
        print("  1. Try running with sudo: sudo python3 test_cameras.py")
        print("  2. Check permissions: ls -la /dev/video*")
        print("  3. Add user to video group: sudo usermod -a -G video $USER")
        print("  4. Reboot and try again")
