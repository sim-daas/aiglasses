import cv2
import numpy as np
import threading
import time
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
        
        self.cap_left = cv2.VideoCapture(Config.CAMERA_LEFT_INDEX)
        self.cap_right = cv2.VideoCapture(Config.CAMERA_RIGHT_INDEX)
        
        if not self.cap_left.isOpened():
            raise RuntimeError(f"Cannot open camera {Config.CAMERA_LEFT_INDEX}")
        if not self.cap_right.isOpened():
            raise RuntimeError(f"Cannot open camera {Config.CAMERA_RIGHT_INDEX}")
        
        # Set camera properties
        for cap in [self.cap_left, self.cap_right]:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, Config.CAMERA_WIDTH)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, Config.CAMERA_HEIGHT)
            cap.set(cv2.CAP_PROP_FPS, Config.CAMERA_FPS)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        print(f"✅ Cameras initialized at {Config.CAMERA_WIDTH}x{Config.CAMERA_HEIGHT}@{Config.CAMERA_FPS}fps")
        
    def start(self):
        """Start capture thread"""
        self.running = True
        self.thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.thread.start()
        print("✅ Camera capture started")
        
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
        if self.cap_right:
            self.cap_right.release()
        print("✅ Cameras stopped")
