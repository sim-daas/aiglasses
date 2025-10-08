#!/usr/bin/env python3
"""
Debug script to test model initialization issue
"""

import os
import google.generativeai as genai
from dotenv import load_dotenv

def debug_model_init():
    """Debug model initialization"""
    load_dotenv()
    
    api_key = os.getenv('GEMINI_API_KEY')
    if not api_key:
        print("❌ No API key found")
        return
    
    print(f"✅ API key found: {api_key[:10]}...")
    
    try:
        genai.configure(api_key=api_key)
        print("✅ API configured")
        
        # List models
        models = genai.list_models()
        vision_models = []
        
        print("\n📋 All available models:")
        for model in models:
            model_name = model.name.replace('models/', '')
            supports_vision = 'generateContent' in model.supported_generation_methods
            print(f"  - {model_name} (Vision: {supports_vision})")
            
            if supports_vision:
                vision_models.append(model_name)
        
        print(f"\n🔍 Vision-capable models: {vision_models}")
        
        # Test each vision model
        for model_name in vision_models[:3]:  # Test first 3
            try:
                print(f"\n🧪 Testing {model_name}...")
                model = genai.GenerativeModel(model_name)
                
                # Try a simple text prompt first
                response = model.generate_content("Hello, can you see images?")
                print(f"✅ {model_name} - Text response: {response.text[:50]}...")
                
                # This model works, let's use it
                print(f"🎯 {model_name} is working!")
                break
                
            except Exception as e:
                print(f"❌ {model_name} failed: {e}")
        
    except Exception as e:
        print(f"❌ General error: {e}")

if __name__ == "__main__":
    debug_model_init()