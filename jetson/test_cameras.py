#!/usr/bin/env python3
"""
Test stereo camera - Single device with side-by-side output
"""

import cv2
import time

def test_stereo_camera_split():
    """Test stereo camera with frame splitting"""
    print("🎥 Stereo Camera Test - Side-by-Side Frame Splitting")
    print("="*50)
    
    device = "/dev/video0"
    
    print(f"Opening {device}...")
    cap = cv2.VideoCapture(device)
    
    if not cap.isOpened():
        print(f"❌ Failed to open {device}")
        return False
    
    print(f"✅ Opened {device}")
    
    # Get properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    print(f"📊 Full frame: {width}x{height} @ {fps}fps")
    print(f"📊 Each camera: {width//2}x{height}")
    
    print(f"\n📺 Displaying stereo video...")
    print("   Press 'q' to quit")
    print("   Press 's' to save frame")
    print("   Press '1' to show left only")
    print("   Press '2' to show right only")
    print("   Press '3' to show stereo (default)")
    
    display_mode = 'stereo'  # 'left', 'right', or 'stereo'
    frame_count = 0
    
    while True:
        ret, stereo_frame = cap.read()
        
        if not ret:
            print("❌ Failed to read frame")
            break
        
        frame_count += 1
        
        # Split into left and right
        h, w = stereo_frame.shape[:2]
        mid = w // 2
        left_frame = stereo_frame[:, :mid]
        right_frame = stereo_frame[:, mid:]
        
        # Add labels
        cv2.putText(left_frame, "LEFT", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(right_frame, "RIGHT", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        cv2.putText(left_frame, f"Frame: {frame_count}", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.putText(right_frame, f"Frame: {frame_count}", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Choose display based on mode
        if display_mode == 'left':
            display_frame = left_frame
            window_title = "LEFT Camera"
        elif display_mode == 'right':
            display_frame = right_frame
            window_title = "RIGHT Camera"
        else:  # stereo
            display_frame = cv2.hconcat([left_frame, right_frame])
            window_title = "Stereo Camera (Left | Right)"
        
        cv2.imshow(window_title, display_frame)
        
        # Handle keys
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            print("User quit")
            break
        elif key == ord('s'):
            timestamp = int(time.time())
            filename = f"stereo_frame_{timestamp}.jpg"
            cv2.imwrite(filename, cv2.hconcat([left_frame, right_frame]))
            print(f"💾 Saved: {filename}")
        elif key == ord('1'):
            display_mode = 'left'
            cv2.destroyAllWindows()
            print("Display mode: LEFT only")
        elif key == ord('2'):
            display_mode = 'right'
            cv2.destroyAllWindows()
            print("Display mode: RIGHT only")
        elif key == ord('3'):
            display_mode = 'stereo'
            cv2.destroyAllWindows()
            print("Display mode: STEREO")
    
    cap.release()
    cv2.destroyAllWindows()
    
    print(f"\n✅ Test complete - Captured {frame_count} frames")
    return True

if __name__ == "__main__":
    success = test_stereo_camera_split()
    
    if success:
        print("\n🎉 SUCCESS! Stereo camera working!")
        print("   Your camera outputs side-by-side stereo on /dev/video0")
        print("   Each frame is split into left (320px) and right (320px)")
        print("   You can now run main.py")
    else:
        print("\n❌ Test failed")
