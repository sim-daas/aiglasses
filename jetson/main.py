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
import logging

# Setup logging FIRST (before any logging calls)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

# Suppress warnings
os.environ['ALSA_CARD'] = 'default'
import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)

from camera_manager import StereoCamera
from gemini_client import GeminiClient
from web_server import WebServer
from config import Config
from text_3d_renderer import Text3DRenderer

# Conditional imports
try:
    from gesture_keyboard import GestureKeyboard
    GESTURE_KB_AVAILABLE = True
except ImportError:
    GESTURE_KB_AVAILABLE = False
    logger.warning("⚠️  Gesture keyboard not available. Install with: pip install mediapipe")

class AURAGlasses:
    def __init__(self, test_mode=False, use_gesture_kb=False):
        logger.info("🚀 Initializing AURA AI Glasses...")
        
        # Initialize components
        logger.info("Initializing camera manager...")
        self.camera = StereoCamera()
        
        logger.info("Initializing Gemini client...")
        self.gemini = GeminiClient()
        
        # Initialize cameras
        logger.info("Starting camera capture...")
        self.camera.initialize()
        self.camera.start()
        
        # Initialize web server
        logger.info("Initializing web server...")
        self.server = WebServer(self.camera)
        
        # State
        self.test_mode = test_mode
        self.use_gesture_kb = use_gesture_kb
        self.current_result = None
        self.gesture_keyboard = None
        
        # Initialize 3D text renderer
        logger.info("Initializing 3D text renderer...")
        self.text_renderer = Text3DRenderer()
        
        # Initialize gesture keyboard if requested
        if use_gesture_kb:
            if GESTURE_KB_AVAILABLE:
                logger.info("Initializing gesture keyboard...")
                self.gesture_keyboard = GestureKeyboard()
            else:
                logger.error("❌ Gesture keyboard requested but MediaPipe not available!")
                logger.error("   Install with: pip install mediapipe")
                sys.exit(1)
        
        logger.info("✅ AURA AI Glasses initialized!")
        
        logger.info("\n📝 Instructions:")
        logger.info("   - Open browser to http://<jetson-ip>:5000")
        if test_mode:
            logger.info("   - Press 't' to send TEST query")
        elif use_gesture_kb:
            logger.info("   - Use hand gestures to type query")
            logger.info("   - Pinch + move = select letter")
            logger.info("   - Two fingers = space")
            logger.info("   - Open palm (hold) = backspace")
            logger.info("   - Three fingers (hold) = submit")
        logger.info("   - Press Q to quit\n")
    
    def process_test_query(self):
        """Process a hardcoded test query without audio"""
        logger.info("="*50)
        logger.info("🧪 TEST MODE - Processing hardcoded query")
        logger.info("="*50)
        
        try:
            # Get current frame
            logger.info("Step 1/6: Getting camera frame...")
            frame_left, frame_right, depth_map = self.camera.get_frames()
            
            if frame_left is None:
                logger.error("❌ No camera frame available")
                return
            
            logger.info(f"✅ Frame captured: L={frame_left.shape}, R={frame_right.shape}")
            
            # Save frame to temporary file
            logger.info("Step 2/6: Saving frame to temp file...")
            image_file = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg')
            cv2.imwrite(image_file.name, frame_right)
            logger.info(f"✅ Frame saved to: {image_file.name}")
            
            # Hardcoded test query
            test_query = "What is the screen size of the monitor?"
            logger.info(f"Step 3/6: Using test query: '{test_query}'")
            
            # Process with Gemini
            logger.info("Step 4/6: Sending to Gemini API...")
            logger.info(f"   Image: {image_file.name}")
            logger.info(f"   Query: {test_query}")
            logger.info("   (This may take 5-10 seconds...)")
            
            result = self.gemini.process_multimodal_query(
                image_path=image_file.name,
                text_query=test_query
            )
            
            logger.info("✅ Received response from Gemini")
            
            # Add depth information
            logger.info("Step 5/6: Calculating depth information...")
            pos_x = int(result['position']['x'] * Config.SINGLE_CAM_WIDTH)
            pos_y = int(result['position']['y'] * Config.SINGLE_CAM_HEIGHT)
            depth_value = self.camera.get_depth_at_point(pos_x, pos_y)
            
            result['position']['z'] = depth_value
            result['position']['depth_normalized'] = depth_value
            logger.info(f"✅ Depth at ({pos_x}, {pos_y}): {depth_value:.3f}")
            
            # Display results
            logger.info("Step 6/6: Broadcasting results...")
            logger.info("📊 RESULTS:")
            logger.info(f"   Q: {result['transcription']}")
            logger.info(f"   A: {result['answer']}")
            logger.info(f"   Object: {result['object']}")
            logger.info(f"   Position: ({result['position']['x']:.2f}, {result['position']['y']:.2f}, {depth_value:.2f})")
            logger.info(f"   Confidence: {result['position']['confidence']:.2%}")
            
            # Broadcast to web clients
            self.server.broadcast_result(result)
            self.current_result = result
            
            logger.info("✅ Test query processing complete!")
            
            # Cleanup
            os.unlink(image_file.name)
            
        except Exception as e:
            logger.error(f"❌ Processing error: {e}")
            import traceback
            logger.error(traceback.format_exc())
    
    def process_gesture_query(self, query_text):
        """Process query from gesture keyboard"""
        logger.info("\n" + "="*50)
        logger.info(f"✋ Processing gesture query: '{query_text}'")
        logger.info("="*50)
        
        try:
            # Get current frame
            frame_left, frame_right, depth_map = self.camera.get_frames()
            
            if frame_left is None:
                logger.error("❌ No camera frame available")
                return
            
            # Save frame to temporary file
            image_file = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg')
            cv2.imwrite(image_file.name, frame_right)
            
            logger.info("📸 Frame captured, sending to Gemini...")
            
            # Process with Gemini
            result = self.gemini.process_multimodal_query(
                image_path=image_file.name,
                text_query=query_text
            )
            
            # Add depth information
            pos_x = int(result['position']['x'] * Config.SINGLE_CAM_WIDTH)
            pos_y = int(result['position']['y'] * Config.SINGLE_CAM_HEIGHT)
            depth_value = self.camera.get_depth_at_point(pos_x, pos_y)
            
            result['position']['z'] = depth_value
            result['position']['depth_normalized'] = depth_value
            
            # Display results
            logger.info("📊 RESULTS:")
            logger.info(f"   Q: {result['transcription']}")
            logger.info(f"   A: {result['answer']}")
            logger.info(f"   Object: {result['object']}")
            logger.info(f"   Position: ({result['position']['x']:.2f}, {result['position']['y']:.2f}, {depth_value:.2f})")
            
            # Broadcast to web clients
            self.server.broadcast_result(result)
            self.current_result = result
            
            # Reset gesture keyboard
            if self.gesture_keyboard:
                self.gesture_keyboard.reset_text()
            
            # Cleanup
            os.unlink(image_file.name)
            logger.info("✅ Gesture query processing complete!")
            
        except Exception as e:
            logger.error(f"❌ Processing error: {e}")
            import traceback
            logger.error(traceback.format_exc())
    
    def _map_location_to_position(self, location, frame_width, frame_height):
        """Map Gemini's text location to pixel coordinates"""
        location_map = {
            'top-left': (int(frame_width * 0.20), int(frame_height * 0.20)),
            'top-center': (int(frame_width * 0.50), int(frame_height * 0.20)),
            'top-right': (int(frame_width * 0.80), int(frame_height * 0.20)),
            'center-left': (int(frame_width * 0.20), int(frame_height * 0.50)),
            'center': (int(frame_width * 0.50), int(frame_height * 0.50)),
            'center-right': (int(frame_width * 0.80), int(frame_height * 0.50)),
            'bottom-left': (int(frame_width * 0.20), int(frame_height * 0.80)),
            'bottom-center': (int(frame_width * 0.50), int(frame_height * 0.80)),
            'bottom-right': (int(frame_width * 0.80), int(frame_height * 0.80)),
        }
        return location_map.get(location.lower(), (int(frame_width * 0.5), int(frame_height * 0.5)))
    
    def _draw_3d_text_overlay(self, frame, result):
        """Draw 3D text with proper depth-based perspective scaling"""
        if not result:
            return frame
        
        h, w = frame.shape[:2]
        location = result.get('location', 'center')
        pixel_x, pixel_y = self._map_location_to_position(location, w, h)
        depth_normalized = result.get('position', {}).get('z', 0.5)
        z_depth = 20.0 - (depth_normalized * 15.0)
        
        answer = result['answer']
        object_name = result['object']
        
        # Render main answer
        frame = self.text_renderer.render_3d_text(
            frame, answer, (pixel_x, pixel_y), z_depth=z_depth
        )
        
        # Render label
        label_offset = int(h * 0.08)
        label_y = min(pixel_y + label_offset, h - 50)
        frame = self.text_renderer.render_3d_text(
            frame, f"[{object_name}]", (pixel_x, label_y), z_depth=z_depth * 0.5
        )
        
        # Draw indicators
        cv2.line(frame, (pixel_x - 10, pixel_y), (pixel_x + 10, pixel_y), (0, 255, 0), 2)
        cv2.line(frame, (pixel_x, pixel_y - 10), (pixel_x, pixel_y + 10), (0, 255, 0), 2)
        cv2.circle(frame, (pixel_x, pixel_y), 8, (0, 255, 0), 2)
        
        return frame
    
    def run(self):
        """Run the main application"""
        # Start web server
        logger.info("Starting web server thread...")
        server_thread = threading.Thread(target=self.server.run, daemon=True)
        server_thread.start()
        time.sleep(2)
        
        logger.info(f"\n🌐 Web interface: http://localhost:{Config.SERVER_PORT}\n")
        
        if self.test_mode:
            logger.info("🧪 TEST MODE ENABLED")
            logger.info("Press 't' to test, 'q' to quit\n")
        elif self.use_gesture_kb:
            logger.info("✋ GESTURE KEYBOARD MODE")
            logger.info("Use hand gestures to type, 'q' to quit\n")
        
        try:
            frame_count = 0
            
            while True:
                # Get frames
                frame_left, frame_right, _ = self.camera.get_frames()
                
                if frame_left is not None:
                    frame_count += 1
                    
                    # Use only left camera for display
                    display_frame = frame_left.copy()
                    
                    # Process gesture keyboard if enabled
                    if self.use_gesture_kb and self.gesture_keyboard:
                        display_frame, status, should_submit = self.gesture_keyboard.process_frame(display_frame)
                        
                        if should_submit:
                            query = self.gesture_keyboard.get_text().strip()
                            if query:
                                self.process_gesture_query(query)
                    
                    # Add 3D text overlay if we have results
                    if self.current_result:
                        display_frame = self._draw_3d_text_overlay(display_frame, self.current_result)
                    
                    # Add frame counter
                    cv2.putText(display_frame, f"Frame: {frame_count}", (10, 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    
                    # Show single camera view
                    cv2.imshow('AURA AI Glasses', display_frame)
                
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('t') and self.test_mode:
                    self.process_test_query()
                elif key == ord('q'):
                    logger.info("'q' pressed - shutting down")
                    break
                
                time.sleep(0.01)
        
        except KeyboardInterrupt:
            logger.info("\n👋 Interrupted, shutting down...")
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Cleanup resources"""
        logger.info("Cleaning up resources...")
        self.camera.stop()
        if self.gesture_keyboard:
            self.gesture_keyboard.cleanup()
        cv2.destroyAllWindows()
        logger.info("✅ Cleanup complete")

def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="AURA AI Glasses")
    parser.add_argument('--test', action='store_true',
                       help='Enable test mode (hardcoded queries)')
    parser.add_argument('--gesture', action='store_true',
                       help='Enable gesture keyboard mode')
    
    args = parser.parse_args()
    
    try:
        app = AURAGlasses(test_mode=args.test, use_gesture_kb=args.gesture)
        app.run()
    except Exception as e:
        logger.error(f"❌ Fatal error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)

if __name__ == "__main__":
    main()
