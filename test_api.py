#!/usr/bin/env python3
"""
Utility script to test Gemini API connection and list available models
"""

import os
import google.generativeai as genai
from dotenv import load_dotenv

def test_gemini_api():
    """Test Gemini API connection and list available models"""
    # Load environment variables
    load_dotenv()
    
    api_key = os.getenv('GEMINI_API_KEY')
    if not api_key:
        print("❌ GEMINI_API_KEY not found in .env file")
        print("Please add your API key to the .env file")
        print("Get your API key from: https://makersuite.google.com/app/apikey")
        return False
    
    try:
        # Configure API
        genai.configure(api_key=api_key)
        print("✅ API key configured successfully")
        
        # List available models
        print("\n📋 Available models:")
        models = genai.list_models()
        
        vision_models = []
        text_models = []
        
        for model in models:
            if 'generateContent' in model.supported_generation_methods:
                model_name = model.name.replace('models/', '')
                if 'vision' in model_name.lower() or 'pro' in model_name.lower() or 'flash' in model_name.lower():
                    vision_models.append(model_name)
                else:
                    text_models.append(model_name)
        
        print("\n🔍 Vision/Multimodal Models (recommended for image analysis):")
        for model in vision_models:
            print(f"  - {model}")
        
        print("\n📝 Text Models:")
        for model in text_models[:5]:  # Show first 5
            print(f"  - {model}")
        
        # Test a simple model initialization
        print("\n🧪 Testing model initialization...")
        
        test_models = [
            'gemini-1.5-flash-latest',
            'gemini-1.5-flash',
            'gemini-1.5-pro-latest',
            'gemini-1.5-pro',
            'gemini-pro-vision'
        ]
        
        working_model = None
        for model_name in test_models:
            try:
                model = genai.GenerativeModel(model_name)
                print(f"✅ {model_name} - Working")
                if working_model is None:
                    working_model = model_name
            except Exception as e:
                print(f"❌ {model_name} - Error: {e}")
        
        if working_model:
            print(f"\n🎯 Recommended model to use: {working_model}")
            return True
        else:
            print("\n❌ No working models found")
            return False
            
    except Exception as e:
        print(f"❌ API connection failed: {e}")
        return False

if __name__ == "__main__":
    print("🔍 Testing Gemini API Connection...")
    print("=" * 50)
    
    success = test_gemini_api()
    
    print("\n" + "=" * 50)
    if success:
        print("✅ API test completed successfully!")
        print("You can now run: python3 visionapi.py")
    else:
        print("❌ API test failed. Please check your configuration.")