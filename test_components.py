#!/usr/bin/env python3
"""
Test script to verify AI Glasses Pipeline components
"""

import sys
import os
from dotenv import load_dotenv

def test_imports():
    """Test all required imports"""
    print("🧪 Testing imports...")
    
    try:
        import cv2
        print("✅ OpenCV imported successfully")
        print(f"   Version: {cv2.__version__}")
    except ImportError as e:
        print(f"❌ OpenCV import failed: {e}")
        return False
    
    try:
        import numpy as np
        print("✅ NumPy imported successfully")
        print(f"   Version: {np.__version__}")
    except ImportError as e:
        print(f"❌ NumPy import failed: {e}")
        return False
    
    try:
        import pyaudio
        print("✅ PyAudio imported successfully")
    except ImportError as e:
        print(f"❌ PyAudio import failed: {e}")
        return False
    
    try:
        import google.generativeai as genai
        print("✅ Google Generative AI imported successfully")
    except ImportError as e:
        print(f"❌ Google Generative AI import failed: {e}")
        return False
    
    try:
        from deepgram import DeepgramClient
        print("✅ Deepgram imported successfully")
    except ImportError as e:
        print(f"❌ Deepgram import failed: {e}")
        return False
    
    try:
        import tkinter as tk
        print("✅ Tkinter imported successfully")
    except ImportError as e:
        print(f"❌ Tkinter import failed: {e}")
        return False
    
    try:
        from PIL import Image
        print("✅ Pillow imported successfully")
    except ImportError as e:
        print(f"❌ Pillow import failed: {e}")
        return False
    
    return True

def test_cameras():
    """Test camera access"""
    print("\n📷 Testing cameras...")
    
    import cv2
    
    # Test camera 0
    cap0 = cv2.VideoCapture(0)
    if cap0.isOpened():
        ret, frame = cap0.read()
        if ret:
            print("✅ Camera 0 (/dev/video0) is working")
            print(f"   Resolution: {frame.shape[1]}x{frame.shape[0]}")
        else:
            print("⚠️  Camera 0 opened but cannot read frames")
        cap0.release()
    else:
        print("❌ Camera 0 (/dev/video0) cannot be opened")
    
    # Test camera 1
    cap1 = cv2.VideoCapture(1)
    if cap1.isOpened():
        ret, frame = cap1.read()
        if ret:
            print("✅ Camera 1 (/dev/video1) is working")
            print(f"   Resolution: {frame.shape[1]}x{frame.shape[0]}")
        else:
            print("⚠️  Camera 1 opened but cannot read frames")
        cap1.release()
    else:
        print("❌ Camera 1 (/dev/video1) cannot be opened")

def test_audio():
    """Test audio system"""
    print("\n🎤 Testing audio...")
    
    try:
        import pyaudio
        
        p = pyaudio.PyAudio()
        device_count = p.get_device_count()
        print(f"✅ Found {device_count} audio devices")
        
        input_devices = []
        for i in range(device_count):
            info = p.get_device_info_by_index(i)
            if info['maxInputChannels'] > 0:
                input_devices.append((i, info['name']))
                print(f"   Input device {i}: {info['name']} ({info['maxInputChannels']} channels)")
        
        if not input_devices:
            print("⚠️  No input audio devices found")
        
        p.terminate()
        
    except Exception as e:
        print(f"❌ Audio test failed: {e}")

def test_api_keys():
    """Test API key configuration"""
    print("\n🔑 Testing API keys...")
    
    load_dotenv()
    
    deepgram_key = os.getenv('DEEPGRAM_API_KEY')
    if deepgram_key and deepgram_key != 'your_deepgram_api_key_here':
        print("✅ Deepgram API key found")
    else:
        print("❌ Deepgram API key not configured")
    
    gemini_key = os.getenv('GEMINI_API_KEY')
    if gemini_key and gemini_key != 'your_gemini_api_key_here':
        print("✅ Gemini API key found")
    else:
        print("❌ Gemini API key not configured")

def test_custom_modules():
    """Test custom module imports"""
    print("\n📦 Testing custom modules...")
    
    try:
        from visionapi import VisionAPI
        print("✅ VisionAPI module imported successfully")
    except ImportError as e:
        print(f"❌ VisionAPI import failed: {e}")
    
    try:
        from bbox3d_utils import BBox3DEstimator, BirdEyeView
        print("✅ BBox3D utilities imported successfully")
    except ImportError as e:
        print(f"❌ BBox3D utilities import failed: {e}")
    
    try:
        from owl_predict import OwlPredictor
        print("✅ NanoOwl predictor imported successfully")
    except ImportError as e:
        print(f"⚠️  NanoOwl not available (this is optional): {e}")

def main():
    """Run all tests"""
    print("🚀 AI Glasses Pipeline Component Test")
    print("=" * 50)
    
    # Test imports
    if not test_imports():
        print("\n❌ Import tests failed. Please run setup script first.")
        return 1
    
    # Test hardware
    test_cameras()
    test_audio()
    
    # Test configuration
    test_api_keys()
    
    # Test custom modules
    test_custom_modules()
    
    print("\n" + "=" * 50)
    print("🎯 Test Summary:")
    print("   - If all core imports passed, basic functionality should work")
    print("   - Configure API keys in .env file if not done already")
    print("   - Ensure cameras are connected for full functionality")
    print("   - NanoOwl is optional and will use fallback if not available")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())