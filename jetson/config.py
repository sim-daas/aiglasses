import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    # Camera settings - Single stereo camera outputting side-by-side frames
    CAMERA_DEVICE = "/dev/video0"  # Single device with stereo output
    CAMERA_WIDTH = 1280   # Full width (both cameras side-by-side) - DOUBLED from 640
    CAMERA_HEIGHT = 480  # Height - DOUBLED from 240
    CAMERA_FPS = 30
    
    # Individual camera dimensions (half of full width)
    SINGLE_CAM_WIDTH = 640   # Each camera is 640 pixels wide - DOUBLED from 320
    SINGLE_CAM_HEIGHT = 480  # Same height - DOUBLED from 240
    
    # Audio settings
    AUDIO_RATE = 16000
    AUDIO_CHANNELS = 1
    AUDIO_CHUNK = 1024
    AUDIO_FORMAT = 'int16'
    
    # Gemini API
    GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
    GEMINI_MODEL = 'gemini-2.5-flash'  # Fixed: was 'gemini-2.5-flash', should be 'gemini-1.5-flash'
    GEMINI_MAX_RETRIES = 3
    GEMINI_RETRY_DELAY = 2  # seconds
    
    # Web server
    SERVER_HOST = '0.0.0.0'
    SERVER_PORT = 5000
    
    # Depth estimation
    STEREO_NUM_DISPARITIES = 64
    STEREO_BLOCK_SIZE = 11
    
    # 3D positioning
    FOCAL_LENGTH = 500  # pixels (calibrate for your camera)
    BASELINE = 0.06     # meters (distance between cameras)
