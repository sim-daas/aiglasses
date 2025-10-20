import cv2
import numpy as np
import threading
import time
import os
from config import Config

class StereoCamera:
    def __init__(self):
        self.cap_left = None
        self.cap_right = None
        self.frame_left = None
        self.frame_right = None
        self.depth_map = None
        self.running = False
        self.lock = threading.Lock()
        
        # Stereo matcher
        self.stereo = cv2.StereoBM_create(
            numDisparities=Config.STEREO_NUM_DISPARITIES,
            blockSize=Config.STEREO_BLOCK_SIZE
        )
        
    def initialize(self):
        """Initialize both cameras"""
        print(f"🎥 Initializing cameras...")
        
        # Verify devices exist
        self._verify_camera_devices()
        
        # List available cameras
        self._list_available_cameras()
        
        # Open left camera using device path with GStreamer pipeline
        print(f"📷 Opening left camera ({Config.CAMERA_LEFT_INDEX})...")
        self.cap_left = self._open_camera_with_gstreamer(Config.CAMERA_LEFT_INDEX, "left")
        
        if not self.cap_left or not self.cap_left.isOpened():
            raise RuntimeError(f"Cannot open camera {Config.CAMERA_LEFT_INDEX}")
        
        # Test read from left camera
        ret, frame = self.cap_left.read()
        if not ret:
            self.cap_left.release()
            raise RuntimeError(f"Camera {Config.CAMERA_LEFT_INDEX} opened but cannot read frames")
        
        print(f"✅ Left camera initialized: {frame.shape[1]}x{frame.shape[0]}")
        
        # Small delay before opening second camera
        time.sleep(0.5)
        
        # Open right camera
        print(f"📷 Opening right camera ({Config.CAMERA_RIGHT_INDEX})...")
        self.cap_right = self._open_camera_with_gstreamer(Config.CAMERA_RIGHT_INDEX, "right")
        
        if not self.cap_right or not self.cap_right.isOpened():
            self.cap_left.release()  # Clean up left camera
            raise RuntimeError(f"Cannot open camera {Config.CAMERA_RIGHT_INDEX}")
        
        # Test read from right camera
        ret, frame = self.cap_right.read()
        if not ret:
            self.cap_left.release()
            self.cap_right.release()
            raise RuntimeError(f"Camera {Config.CAMERA_RIGHT_INDEX} opened but cannot read frames")
        
        print(f"✅ Right camera initialized: {frame.shape[1]}x{frame.shape[0]}")
        print(f"✅ Stereo cameras ready at {Config.CAMERA_WIDTH}x{Config.CAMERA_HEIGHT}@{Config.CAMERA_FPS}fps")
    
    def _verify_camera_devices(self):
        """Verify camera device files exist"""
        print("🔍 Verifying camera devices...")
        
        if not os.path.exists(Config.CAMERA_LEFT_INDEX):
            raise RuntimeError(f"Camera device {Config.CAMERA_LEFT_INDEX} does not exist")
        print(f"  ✅ {Config.CAMERA_LEFT_INDEX} exists")
        
        if not os.path.exists(Config.CAMERA_RIGHT_INDEX):
            raise RuntimeError(f"Camera device {Config.CAMERA_RIGHT_INDEX} does not exist")
        print(f"  ✅ {Config.CAMERA_RIGHT_INDEX} exists")
    
    def _open_camera_with_gstreamer(self, device_path, name):
        """
        Open camera using GStreamer pipeline for Jetson compatibility
        Falls back to direct V4L2 if GStreamer fails
        """
        # Try GStreamer pipeline first (best for Jetson)
        gst_pipeline = (
            f"v4l2src device={device_path} ! "
            f"video/x-raw, width={Config.CAMERA_WIDTH}, height={Config.CAMERA_HEIGHT}, framerate={Config.CAMERA_FPS}/1 ! "
            "videoconvert ! video/x-raw, format=BGR ! appsink drop=1"
        )
        
        print(f"  Trying GStreamer pipeline for {name} camera...")
        cap = cv2.VideoCapture(gst_pipeline, cv2.CAP_GSTREAMER)
        
        if cap.isOpened():
            print(f"  ✅ {name.capitalize()} camera opened with GStreamer")
            return cap
        
        print(f"  ⚠️  GStreamer failed, trying V4L2 directly...")
        
        # Fallback: Try direct V4L2 with device path
        cap = cv2.VideoCapture(device_path, cv2.CAP_V4L2)
        
        if cap.isOpened():
            # Set properties
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, Config.CAMERA_WIDTH)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, Config.CAMERA_HEIGHT)
            cap.set(cv2.CAP_PROP_FPS, Config.CAMERA_FPS)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            print(f"  ✅ {name.capitalize()} camera opened with V4L2")
            return cap
        
        print(f"  ❌ Failed to open {name} camera with both methods")
        return None
        
    def _list_available_cameras(self):
        """List available cameras for debugging"""
        print("🔍 Scanning /dev/video* devices...")
        
        for i in range(10):
            device = f"/dev/video{i}"
            if os.path.exists(device):
                # Check if it's a real camera (not a metadata device)
                try:
                    cap = cv2.VideoCapture(device, cv2.CAP_V4L2)
                    if cap.isOpened():
                        ret, frame = cap.read()
                        cap.release()
                        
                        if ret:
                            print(f"  ✅ {device} - Working (captured {frame.shape})")
                        else:
                            print(f"  ⚠️  {device} - Opens but no frames (may be metadata device)")
                    else:
                        print(f"  ❌ {device} - Cannot open with V4L2")
                except Exception as e:
                    print(f"  ❌ {device} - Error: {e}")
            else:
                break  # No more devices
    
    def start(self):
        """Start capture thread"""
        self.running = True
        self.thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.thread.start()
        print("✅ Camera capture thread started")
        
    def _capture_loop(self):
        """Continuous capture loop"""
        while self.running:
            ret_left, frame_left = self.cap_left.read()
            ret_right, frame_right = self.cap_right.read()
            
            if ret_left and ret_right:
                with self.lock:
                    self.frame_left = frame_left
                    self.frame_right = frame_right
                    self._compute_depth(frame_left, frame_right)
            else:
                if not ret_left:
                    print("⚠️  Left camera read failed")
                if not ret_right:
                    print("⚠️  Right camera read failed")
            
            time.sleep(1 / Config.CAMERA_FPS)
    
    def _compute_depth(self, left, right):
        """Compute depth map from stereo pair"""
        try:
            gray_left = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY)
            gray_right = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY)
            
            disparity = self.stereo.compute(gray_left, gray_right).astype(np.float32) / 16.0
            
            # Normalize to 0-1 range
            self.depth_map = cv2.normalize(disparity, None, 0, 1, cv2.NORM_MINMAX)
        except Exception as e:
            print(f"⚠️  Depth computation error: {e}")
    
    def get_frames(self):
        """Get latest frames (thread-safe)"""
        with self.lock:
            return (
                self.frame_left.copy() if self.frame_left is not None else None,
                self.frame_right.copy() if self.frame_right is not None else None,
                self.depth_map.copy() if self.depth_map is not None else None
            )
    
    def get_depth_at_point(self, x, y):
        """Get depth value at specific pixel coordinate"""
        with self.lock:
            if self.depth_map is None:
                return 0.5
            
            h, w = self.depth_map.shape
            x = max(0, min(x, w - 1))
            y = max(0, min(y, h - 1))
            
            # Average in 10x10 region
            region = self.depth_map[
                max(0, y-5):min(h, y+5),
                max(0, x-5):min(w, x+5)
            ]
            return float(np.mean(region))
    
    def stop(self):
        """Stop cameras"""
        self.running = False
        
        if self.cap_left:
            self.cap_left.release()
            print("✅ Left camera released")
        
        if self.cap_right:
            self.cap_right.release()
            print("✅ Right camera released")
        
        cv2.destroyAllWindows()
