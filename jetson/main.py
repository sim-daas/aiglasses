#!/usr/bin/env python3
"""
AURA AI Glasses - Main Entry Point
Simplified Gemini-only pipeline with stereo camera + web display
"""

import sys
import time
import threading
import tempfile
import os
import cv2
from camera_manager import StereoCamera
from audio_manager import AudioManager
from gemini_client import GeminiClient
from web_server import WebServer
from config import Config

class AURAGlasses:
    def __init__(self):
        print("🚀 Initializing AURA AI Glasses...")
        
        # Initialize components
        self.camera = StereoCamera()
        self.audio = AudioManager()
        self.gemini = GeminiClient()
        
        # Initialize cameras
        self.camera.initialize()
        self.camera.start()
        
        # Initialize web server
        self.server = WebServer(self.camera)
        
        # State
        self.recording = False
        
        print("✅ AURA AI Glasses initialized!")
        print("🎤 Audio devices:")
        self.audio.list_devices()
        print("\n📝 Instructions:")
        print("   - Open browser to http://<jetson-ip>:5000")
        print("   - Press SPACE to start/stop recording")
        print("   - Press Q to quit\n")
    
    def process_query(self):
        """Process user voice query with Gemini"""
        print("\n" + "="*50)
        print("🎤 Recording... (release to process)")
        
        # Start audio recording
        if not self.audio.start_recording():
            print("❌ Failed to start recording")
            return
        
        self.recording = True
        
    def stop_query(self):
        """Stop recording and process"""
        if not self.recording:
            return
        
        self.recording = False
        
        print("⏹️  Recording stopped, processing...")
        
        # Stop recording and get audio file
        audio_file = self.audio.stop_recording()
        
        if not audio_file:
            print("❌ No audio recorded")
            return
        
        try:
            # Get current frame
            frame_left, frame_right, depth_map = self.camera.get_frames()
            
            if frame_left is None:
                print("❌ No camera frame available")
                return
            
            # Save frame to temporary file
            image_file = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg')
            cv2.imwrite(image_file.name, frame_right)  # Use right camera
            
            print("📸 Frame captured, sending to Gemini...")
            
            # Process with Gemini
            result = self.gemini.process_multimodal_query(
                image_path=image_file.name,
                audio_path=audio_file
            )
            
            # Add depth information
            pos_x = int(result['position']['x'] * Config.CAMERA_WIDTH)
            pos_y = int(result['position']['y'] * Config.CAMERA_HEIGHT)
            depth_value = self.camera.get_depth_at_point(pos_x, pos_y)
            
            result['position']['z'] = depth_value
            result['position']['depth_normalized'] = depth_value
            
            # Display results
            print("\n📊 RESULTS:")
            print(f"   Q: {result['transcription']}")
            print(f"   A: {result['answer']}")
            print(f"   Object: {result['object']}")
            print(f"   Position: ({result['position']['x']:.2f}, {result['position']['y']:.2f}, {depth_value:.2f})")
            print(f"   Confidence: {result['position']['confidence']:.2%}")
            
            # Broadcast to web clients
            self.server.broadcast_result(result)
            
            # Cleanup temp files
            os.unlink(audio_file)
            os.unlink(image_file.name)
            
        except Exception as e:
            print(f"❌ Processing error: {e}")
            import traceback
            traceback.print_exc()
    
    def run(self):
        """Run the main application"""
        # Start web server in separate thread
        server_thread = threading.Thread(target=self.server.run, daemon=True)
        server_thread.start()
        
        print(f"\n🌐 Web interface: http://localhost:{Config.SERVER_PORT}")
        print("   (or http://<jetson-ip>:5000 from tablet)\n")
        
        # Main interaction loop (keyboard-based for now)
        print("Press SPACE to record, Q to quit")
        
        try:
            while True:
                # Display live feed in window for debugging
                frame_left, _, _ = self.camera.get_frames()
                
                if frame_left is not None:
                    cv2.imshow('AURA AI - Left Camera', frame_left)
                
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord(' '):  # Spacebar
                    if not self.recording:
                        self.process_query()
                    else:
                        self.stop_query()
                
                elif key == ord('q'):  # Quit
                    print("\n👋 Shutting down...")
                    break
                
                time.sleep(0.01)
        
        except KeyboardInterrupt:
            print("\n👋 Interrupted, shutting down...")
        
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Cleanup resources"""
        self.camera.stop()
        self.audio.cleanup()
        cv2.destroyAllWindows()
        print("✅ Cleanup complete")

def main():
    """Main entry point"""
    try:
        app = AURAGlasses()
        app.run()
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
