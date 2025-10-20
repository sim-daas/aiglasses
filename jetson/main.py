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
from depth_tape_measure import DepthTapeMeasure

# Conditional imports
try:
    from gesture_keyboard import GestureKeyboard
    GESTURE_KB_AVAILABLE = True
except ImportError:
    GESTURE_KB_AVAILABLE = False
    logger.warning("⚠️  Gesture keyboard not available. Install with: pip install mediapipe")

try:
    from live_translator import LiveTranslator
    TRANSLATOR_AVAILABLE = True
except ImportError:
    TRANSLATOR_AVAILABLE = False
    logger.warning("⚠️  Live translator not available. Install: pip install easyocr langdetect deep-translator")

class AURAGlasses:
    def __init__(self, test_mode=False, use_gesture_kb=False, use_tape_measure=False, headless=True):
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
        
        # State - initialize BEFORE creating objects
        self.test_mode = test_mode
        self.use_gesture_kb = use_gesture_kb
        self.use_tape_measure = use_tape_measure
        self.headless = headless  # Run without OpenCV windows
        self.current_result = None
        self.gesture_keyboard = None
        self.tape_measure = None
        self.translator = None
        self.translation_results = []
        self.last_translate_time = 0
        self.submit_pending = False
        self.submit_time = 0
        self.submit_query = ""
        self.translate_pending = False  # Track pending translate
        self.translate_time = 0  # When translate was pressed
        
        # Initialize tape measure if requested
        if use_tape_measure:
            logger.info("Initializing AR tape measure...")
            self.tape_measure = DepthTapeMeasure(
                baseline_mm=65,
                focal_px=900,
                frame_width=Config.SINGLE_CAM_WIDTH,
                frame_height=Config.SINGLE_CAM_HEIGHT
            )
        
        # Initialize web server with tape measure reference
        logger.info("Initializing web server...")
        self.server = WebServer(self.camera, self.tape_measure)
        
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
        
        # Initialize translator if gesture keyboard is enabled
        if use_gesture_kb and TRANSLATOR_AVAILABLE:
            try:
                logger.info("Initializing live translator...")
                # Use basic language set to avoid compatibility issues
                self.translator = LiveTranslator(
                    target_lang='en',
                    ocr_langs=['en', 'es', 'fr', 'de']  # Start with compatible languages
                )
            except Exception as e:
                logger.warning(f"⚠️  Could not initialize translator: {e}")
                logger.warning("    Translation feature will be disabled")
                self.translator = None
        elif use_gesture_kb and not TRANSLATOR_AVAILABLE:
            logger.warning("⚠️  Translator libraries not available")
            logger.warning("    Install with: pip install easyocr langdetect deep-translator")
        
        if not headless:
            logger.info("⚠️  Display mode enabled (requires X11/GTK)")
        else:
            logger.info("✅ Headless mode - using web interface only")
        
        logger.info("✅ AURA AI Glasses initialized!")
        
        logger.info("\n📝 Instructions:")
        logger.info("   - Open browser to http://<jetson-ip>:5000")
        if test_mode:
            logger.info("   - Press 't' to send TEST query")
        elif use_gesture_kb:
            logger.info("   - Use hand gestures to type query")
        if use_tape_measure:
            logger.info("   - Press '1' to set point 1")
            logger.info("   - Press '2' to set point 2")
            logger.info("   - Press 'a' to place arrow")
            logger.info("   - Press 'r' to reset measurements")
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
        
        if self.use_tape_measure:
            logger.info("📏 AR TAPE MEASURE ENABLED")
            logger.info("Click or use keys to measure distances\n")
        
        # Mouse callback only works with OpenCV windows (not in headless mode)
        mouse_callback_registered = False
        if self.use_tape_measure and not self.headless:
            def mouse_callback(event, x, y, flags, param):
                if event == cv2.EVENT_LBUTTONDOWN:
                    if self.tape_measure.point1 is None:
                        self.tape_measure.set_point1(x, y)
                    elif self.tape_measure.point2 is None:
                        self.tape_measure.set_point2(x, y)
                    else:
                        self.tape_measure.set_point1(x, y)
                        self.tape_measure.point2 = None
                elif event == cv2.EVENT_RBUTTONDOWN:
                    self.tape_measure.set_arrow(x, y)
        
        try:
            frame_count = 0
            
            while True:
                # Get frames
                frame_left, frame_right, depth_map = self.camera.get_frames()
                
                if frame_left is not None:
                    frame_count += 1
                    
                    # Use only left camera for display
                    display_frame = frame_left.copy()
                    
                    # Compute depth map for tape measure
                    if self.use_tape_measure and self.tape_measure and frame_right is not None:
                        self.tape_measure.compute_depth(frame_left, frame_right)
                    
                    # Process gesture keyboard if enabled
                    if self.use_gesture_kb and self.gesture_keyboard:
                        zoom_level = self.gesture_keyboard.get_zoom_level()
                        
                        # Apply digital zoom
                        if zoom_level > 1.0:
                            h, w = display_frame.shape[:2]
                            crop_w = int(w / zoom_level)
                            crop_h = int(h / zoom_level)
                            crop_x = (w - crop_w) // 2
                            crop_y = (h - crop_h) // 2
                            cropped = display_frame[crop_y:crop_y+crop_h, crop_x:crop_x+crop_w]
                            display_frame = cv2.resize(cropped, (w, h), interpolation=cv2.INTER_LINEAR)
                        
                        # Process keyboard overlay
                        display_frame, status, should_submit = self.gesture_keyboard.process_frame(display_frame)
                        
                        # Check for TRANSLATE button
                        current_time = time.time()
                        if 'Typed: TRANSLATE' in status and (current_time - self.last_translate_time) > 2.0:
                            if self.translator:
                                # Start translate countdown
                                self.translate_pending = True
                                self.translate_time = current_time
                                self.last_translate_time = current_time
                                logger.info("⏳ TRANSLATE pressed! Capturing in 5 seconds for OCR...")
                            else:
                                if (current_time - self.last_translate_time) > 10.0:
                                    logger.warning("⚠️  Translator not initialized. Install: pip install easyocr langdetect deep-translator")
                                self.last_translate_time = current_time
                        elif should_submit:
                            # SUBMIT button pressed - start countdown
                            query = self.gesture_keyboard.get_text().strip()
                            if query and not self.submit_pending:
                                self.submit_pending = True
                                self.submit_time = current_time
                                self.submit_query = query
                                logger.info(f"⏳ SUBMIT pressed! Capturing in 5 seconds for query: '{query}'")
                                self.gesture_keyboard.reset_text()
                    
                    # Check if 5 seconds have passed since submit
                    if self.submit_pending:
                        current_time = time.time()
                        elapsed = current_time - self.submit_time
                        remaining = 5.0 - elapsed
                        
                        # Draw countdown on display_frame
                        if remaining > 0:
                            countdown_text = f"Capturing in {remaining:.1f}s..."
                            h, w = display_frame.shape[:2]
                            
                            # Large centered countdown
                            font = cv2.FONT_HERSHEY_SIMPLEX
                            font_scale = 1.5
                            thickness = 3
                            text_size = cv2.getTextSize(countdown_text, font, font_scale, thickness)[0]
                            text_x = (w - text_size[0]) // 2
                            text_y = (h + text_size[1]) // 2
                            
                            # Background rectangle
                            padding = 20
                            cv2.rectangle(display_frame,
                                        (text_x - padding, text_y - text_size[1] - padding),
                                        (text_x + text_size[0] + padding, text_y + padding),
                                        (0, 0, 0), -1)
                            
                            # Countdown text (green for submit)
                            cv2.putText(display_frame, countdown_text, (text_x, text_y),
                                       font, font_scale, (0, 255, 0), thickness, cv2.LINE_AA)
                        else:
                            # Time's up - capture and process
                            logger.info("📸 5 seconds elapsed, capturing frame now...")
                            self.submit_pending = False
                            
                            # Process the query with current frame
                            threading.Thread(target=self._process_delayed_query, 
                                           args=(self.submit_query, frame_left, frame_right, depth_map),
                                           daemon=True).start()
                    
                    # Check if 5 seconds have passed since translate
                    if self.translate_pending:
                        current_time = time.time()
                        elapsed = current_time - self.translate_time
                        remaining = 5.0 - elapsed
                        
                        # Draw countdown on display_frame
                        if remaining > 0:
                            countdown_text = f"Translating in {remaining:.1f}s..."
                            h, w = display_frame.shape[:2]
                            
                            # Large centered countdown
                            font = cv2.FONT_HERSHEY_SIMPLEX
                            font_scale = 1.5
                            thickness = 3
                            text_size = cv2.getTextSize(countdown_text, font, font_scale, thickness)[0]
                            text_x = (w - text_size[0]) // 2
                            text_y = (h + text_size[1]) // 2
                            
                            # Background rectangle
                            padding = 20
                            cv2.rectangle(display_frame,
                                        (text_x - padding, text_y - text_size[1] - padding),
                                        (text_x + text_size[0] + padding, text_y + padding),
                                        (0, 0, 0), -1)
                            
                            # Countdown text (orange for translate)
                            cv2.putText(display_frame, countdown_text, (text_x, text_y),
                                       font, font_scale, (0, 165, 255), thickness, cv2.LINE_AA)
                        else:
                            # Time's up - capture and translate
                            logger.info("📸 5 seconds elapsed, capturing frame for translation...")
                            self.translate_pending = False
                            
                            # Process translation with current frame
                            threading.Thread(target=self._process_delayed_translation, 
                                           args=(frame_left, frame_right, depth_map),
                                           daemon=True).start()
                    
                    # Draw translation overlay if available (not from countdown, from actual results)
                    if self.translation_results and self.translator and not self.translate_pending:
                        display_frame = self.translator.draw_overlay(display_frame, self.translation_results)
                    
                    # Draw tape measure overlay
                    if self.use_tape_measure and self.tape_measure:
                        display_frame = self.tape_measure.draw_overlay(display_frame)
                    
                    # Store processed frame for web streaming
                    self.camera.set_processed_frame(display_frame)
                    
                    # Show single camera view (ONLY if not headless)
                    if not self.headless:
                        try:
                            cv2.imshow('AURA AI Glasses', display_frame)
                            
                            # Register mouse callback AFTER window is created (only once)
                            if self.use_tape_measure and not mouse_callback_registered:
                                cv2.setMouseCallback('AURA AI Glasses', mouse_callback)
                                mouse_callback_registered = True
                                logger.info("✅ Mouse callback registered for tape measure")
                        except cv2.error as e:
                            logger.warning(f"⚠️  OpenCV display error (switching to headless): {e}")
                            self.headless = True  # Switch to headless if display fails
                
                # Handle keyboard input
                if not self.headless:
                    key = cv2.waitKey(1) & 0xFF
                else:
                    import select
                    if select.select([sys.stdin], [], [], 0.01)[0]:
                        key_input = sys.stdin.read(1).lower()
                        key = ord(key_input) if key_input else 255
                    else:
                        key = 255
                
                if key == ord('t') and self.test_mode:
                    self.process_test_query()
                elif key == ord('q'):
                    logger.info("'q' pressed - shutting down")
                    break
                elif key == ord('1') and self.use_tape_measure:
                    if self.tape_measure:
                        self.tape_measure.set_point1(self.tape_measure.cx, self.tape_measure.cy)
                elif key == ord('2') and self.use_tape_measure:
                    if self.tape_measure:
                        self.tape_measure.set_point2(self.tape_measure.cx, self.tape_measure.cy)
                elif key == ord('a') and self.use_tape_measure:
                    if self.tape_measure:
                        self.tape_measure.set_arrow(self.tape_measure.cx, self.tape_measure.cy)
                elif key == ord('r') and self.use_tape_measure:
                    if self.tape_measure:
                        self.tape_measure.clear_measurements()
                
                time.sleep(0.01)
        
        except KeyboardInterrupt:
            logger.info("\n👋 Interrupted, shutting down...")
        finally:
            self.cleanup()
    
    def _process_delayed_translation(self, frame_left, frame_right, depth_map):
        """Process OCR translation in background thread (called after 5-second delay)"""
        logger.info("📝 Processing delayed OCR translation...")
        
        try:
            if frame_left is None:
                logger.error("❌ No camera frame available")
                return
            
            # Save frame to temporary file
            image_file = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg')
            cv2.imwrite(image_file.name, frame_right if frame_right is not None else frame_left)
            
            logger.info("📸 Frame captured, sending to Gemini for translation...")
            
            # Create translation prompt
            translation_query = "Please detect and translate all visible text in this image to English. For each text region, provide: 1) The original text, 2) The detected language, 3) The English translation, 4) The approximate location (top-left, center, etc). Format as a clear list."
            
            # Process with Gemini
            result = self.gemini.process_multimodal_query(
                image_path=image_file.name,
                text_query=translation_query
            )
            
            logger.info("📊 TRANSLATION RESULTS:")
            logger.info(f"   Q: {result['transcription']}")
            logger.info(f"   A: {result['answer']}")
            
            # Parse translation result and create overlay data
            translation_data = {
                'type': 'translation',
                'answer': result['answer'],
                'object': result.get('object', 'text'),
                'location': result.get('location', 'center'),
                'position': result.get('position', {'x': 0.5, 'y': 0.5, 'z': 0.5})
            }
            
            # Broadcast to web clients
            self.server.broadcast_result(translation_data)
            
            # Store for local overlay (simplified format)
            self.translation_results = [{
                'text': result.get('answer', 'No text detected'),
                'translated': result.get('answer', ''),
                'bbox': [[100, 100], [500, 100], [500, 300], [100, 300]],
                'center': [300, 200],
                'confidence': 0.9
            }]
            
            # Cleanup
            os.unlink(image_file.name)
            logger.info("✅ Translation processing complete!")
            
            # Clear translation results after 10 seconds
            threading.Timer(10.0, self._clear_translation_results).start()
            
        except Exception as e:
            logger.error(f"❌ Translation processing error: {e}")
            import traceback
            logger.error(traceback.format_exc())
    
    def _clear_translation_results(self):
        """Clear translation overlay after timeout"""
        self.translation_results = []
        logger.info("🗑️ Translation results cleared")
    
    def _process_delayed_query(self, query_text, frame_left, frame_right, depth_map):
        """Process query in background thread (called after 5-second delay)"""
        logger.info(f"✋ Processing delayed query: '{query_text}'")
        
        try:
            if frame_left is None:
                logger.error("❌ No camera frame available")
                return
            
            # Save frame to temporary file
            image_file = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg')
            cv2.imwrite(image_file.name, frame_right if frame_right is not None else frame_left)
            
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
            
            # Cleanup
            os.unlink(image_file.name)
            logger.info("✅ Query processing complete!")
            
        except Exception as e:
            logger.error(f"❌ Processing error: {e}")
            import traceback
            logger.error(traceback.format_exc())
    
    def cleanup(self):
        """Cleanup resources"""
        logger.info("Cleaning up resources...")
        self.camera.stop()
        if self.gesture_keyboard:
            self.gesture_keyboard.cleanup()
        
        # Only destroy windows if not headless
        if not self.headless:
            try:
                cv2.destroyAllWindows()
            except cv2.error:
                pass  # Ignore if OpenCV GUI not available
        
        logger.info("✅ Cleanup complete")

def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="AURA AI Glasses")
    parser.add_argument('--test', action='store_true',
                       help='Enable test mode (hardcoded queries)')
    parser.add_argument('--gesture', action='store_true',
                       help='Enable gesture keyboard mode')
    parser.add_argument('--measure', action='store_true',
                       help='Enable AR tape measure mode')
    parser.add_argument('--display', action='store_true',
                       help='Enable OpenCV display (requires X11/GTK, default: headless)')
    
    args = parser.parse_args()
    
    try:
        app = AURAGlasses(
            test_mode=args.test,
            use_gesture_kb=args.gesture,
            use_tape_measure=args.measure,
            headless=not args.display  # Default to headless mode
        )
        app.run()
    except Exception as e:
        logger.error(f"❌ Fatal error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)

if __name__ == "__main__":
    main()
