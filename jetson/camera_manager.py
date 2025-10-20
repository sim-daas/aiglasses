import cv2
import numpy as np
import threading
import time
from config import Config

class StereoCamera:
    def __init__(self):
        self.cap = None  # Single camera capture
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
        """Initialize stereo camera (single device with side-by-side output)"""
        print(f"🎥 Initializing stereo camera...")
        print(f"   Device: {Config.CAMERA_DEVICE}")
        print(f"   Output: {Config.CAMERA_WIDTH}x{Config.CAMERA_HEIGHT} (stereo side-by-side)")
        
        # Open single camera device
        print(f"📷 Opening {Config.CAMERA_DEVICE}...")
        self.cap = cv2.VideoCapture(Config.CAMERA_DEVICE)
        
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open camera {Config.CAMERA_DEVICE}")
        
        # Set properties for full stereo frame
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, Config.CAMERA_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, Config.CAMERA_HEIGHT)
        self.cap.set(cv2.CAP_PROP_FPS, Config.CAMERA_FPS)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        # Test read
        ret, frame = self.cap.read()
        if not ret:
            self.cap.release()
            raise RuntimeError(f"Camera opened but cannot read frames")
        
        print(f"✅ Camera initialized: {frame.shape[1]}x{frame.shape[0]}")
        print(f"   Left camera: {Config.SINGLE_CAM_WIDTH}x{Config.SINGLE_CAM_HEIGHT}")
        print(f"   Right camera: {Config.SINGLE_CAM_WIDTH}x{Config.SINGLE_CAM_HEIGHT}")
        
        # Test frame splitting
        left, right = self._split_stereo_frame(frame)
        print(f"✅ Frame split successful: L={left.shape}, R={right.shape}")
        
    def _split_stereo_frame(self, stereo_frame):
        """Split side-by-side stereo frame into left and right images"""
        height, width = stereo_frame.shape[:2]
        mid_point = width // 2
        
        # Split into left and right halves
        left_frame = stereo_frame[:, :mid_point]
        right_frame = stereo_frame[:, mid_point:]
        
        return left_frame, right_frame
    
    def start(self):
        """Start capture thread"""
        self.running = True
        self.thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.thread.start()
        print("✅ Camera capture thread started")
        
    def _capture_loop(self):
        """Continuous capture loop"""
        while self.running:
            ret, stereo_frame = self.cap.read()
            
            if ret:
                # Split into left and right frames
                left, right = self._split_stereo_frame(stereo_frame)
                
                with self.lock:
                    self.frame_left = left
                    self.frame_right = right
                    self._compute_depth(left, right)
            else:
                print("⚠️  Camera read failed")
            
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
        """Stop camera capture"""
        logger.info("Stopping camera capture...")
        self.running = False
        
        if self.capture_thread and self.capture_thread.is_alive():
            self.capture_thread.join(timeout=2.0)
        
        if self.cap_left:
            self.cap_left.release()
            logger.info("✅ Left camera released")
        
        if self.cap_right:
            self.cap_right.release()
            logger.info("✅ Right camera released")
        
        # Don't call destroyAllWindows - handled by main app
        logger.info("✅ Camera resources released")
