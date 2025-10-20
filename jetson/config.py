import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    # Camera settings - use integers, not strings
    CAMERA_LEFT_INDEX = 0  # /dev/video0
    CAMERA_RIGHT_INDEX = 1  # /dev/video1
    CAMERA_WIDTH = 640
    CAMERA_HEIGHT = 480
    CAMERA_FPS = 30
    
    # Audio settings
    AUDIO_RATE = 16000
    AUDIO_CHANNELS = 1
    AUDIO_CHUNK = 1024
    AUDIO_FORMAT = 'int16'
    
    # Gemini API
    GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
    GEMINI_MODEL = 'models/gemini-2.5-flash'  # Best multimodal model
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
