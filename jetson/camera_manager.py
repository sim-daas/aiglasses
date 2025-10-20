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
        
        # Open left camera (use default backend, no CAP_V4L2)
        print(f"📷 Opening left camera ({Config.CAMERA_LEFT_INDEX})...")
        self.cap_left = cv2.VideoCapture(Config.CAMERA_LEFT_INDEX)
        
        if not self.cap_left.isOpened():
            raise RuntimeError(f"Cannot open camera {Config.CAMERA_LEFT_INDEX}")
        
        # Set properties for left camera
        self.cap_left.set(cv2.CAP_PROP_FRAME_WIDTH, Config.CAMERA_WIDTH)
        self.cap_left.set(cv2.CAP_PROP_FRAME_HEIGHT, Config.CAMERA_HEIGHT)
        self.cap_left.set(cv2.CAP_PROP_FPS, Config.CAMERA_FPS)
        self.cap_left.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        # Test read from left camera
        ret, frame = self.cap_left.read()
        if not ret:
            self.cap_left.release()
            raise RuntimeError(f"Camera {Config.CAMERA_LEFT_INDEX} opened but cannot read frames")
        
        print(f"✅ Left camera initialized: {frame.shape[1]}x{frame.shape[0]}")
        
        # Important: Delay before opening second camera
        print("⏳ Waiting before opening second camera...")
        time.sleep(1.0)  # Increased delay
        
        # Open right camera (use default backend, no CAP_V4L2)
        print(f"📷 Opening right camera ({Config.CAMERA_RIGHT_INDEX})...")
        self.cap_right = cv2.VideoCapture(Config.CAMERA_RIGHT_INDEX)
        
        if not self.cap_right.isOpened():
            self.cap_left.release()  # Clean up left camera
            raise RuntimeError(f"Cannot open camera {Config.CAMERA_RIGHT_INDEX}")
        
        # Set properties for right camera
        self.cap_right.set(cv2.CAP_PROP_FRAME_WIDTH, Config.CAMERA_WIDTH)
        self.cap_right.set(cv2.CAP_PROP_FRAME_HEIGHT, Config.CAMERA_HEIGHT)
        self.cap_right.set(cv2.CAP_PROP_FPS, Config.CAMERA_FPS)
        self.cap_right.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
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
        
    def _list_available_cameras(self):
        """List available cameras for debugging"""
        print("🔍 Scanning /dev/video* devices...")
        
        for i in range(10):
            device = f"/dev/video{i}"
            if os.path.exists(device):
                # Try opening without backend specification
                try:
                    cap = cv2.VideoCapture(device)
                    if cap.isOpened():
                        ret, frame = cap.read()
                        cap.release()
                        
                        if ret:
                            print(f"  ✅ {device} - Working (captured {frame.shape})")
                        else:
                            print(f"  ⚠️  {device} - Opens but no frames")
                    else:
                        print(f"  ❌ {device} - Cannot open")
                    
                    # Small delay between tests
                    time.sleep(0.2)
                    
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
