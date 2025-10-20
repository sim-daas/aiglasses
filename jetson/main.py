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

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

# Suppress ALSA warnings
os.environ['ALSA_CARD'] = 'default'
import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)

from camera_manager import StereoCamera
from audio_manager import AudioManager
from gemini_client import GeminiClient
from web_server import WebServer
from config import Config

class AURAGlasses:
    def __init__(self, test_mode=False):
        logger.info("🚀 Initializing AURA AI Glasses...")
        
        # Initialize components
        logger.info("Initializing camera manager...")
        self.camera = StereoCamera()
        
        logger.info("Initializing audio manager...")
        self.audio = AudioManager()
        
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
        self.recording = False
        self.test_mode = test_mode
        self.current_result = None  # Store latest result for overlay
        
        logger.info("✅ AURA AI Glasses initialized!")
        
        if not test_mode:
            logger.info("🎤 Audio devices:")
            self.audio.list_devices()
        
        logger.info("\n📝 Instructions:")
        logger.info("   - Open browser to http://<jetson-ip>:5000")
        if test_mode:
            logger.info("   - Press 't' to send TEST query")
        else:
            logger.info("   - Press SPACE to start/stop recording")
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
            
            # Hardcoded test query - CHANGED
            test_query = "What is the capacity of the water bottle?"
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
            logger.info("Broadcasting to web clients...")
            self.server.broadcast_result(result)
            
            # Store result for overlay
            self.current_result = result
            
            logger.info("✅ Broadcast complete!")
            
            # Cleanup temp files
            os.unlink(image_file.name)
            logger.info("✅ Test query processing complete!")
            
        except Exception as e:
            logger.error(f"❌ Processing error: {e}")
            import traceback
            logger.error(traceback.format_exc())
    
    def process_query(self):
        """Process user voice query with Gemini"""
        logger.info("\n" + "="*50)
        logger.info("🎤 Recording... (release to process)")
        
        # Start audio recording
        if not self.audio.start_recording():
            logger.error("❌ Failed to start recording")
            return
        
        self.recording = True
        
    def stop_query(self):
        """Stop recording and process"""
        if not self.recording:
            return
        
        self.recording = False
        
        logger.info("⏹️  Recording stopped, processing...")
        
        # Stop recording and get audio file
        audio_file = self.audio.stop_recording()
        
        if not audio_file:
            logger.error("❌ No audio recorded")
            return
        
        try:
            # Get current frame
            frame_left, frame_right, depth_map = self.camera.get_frames()
            
            if frame_left is None:
                logger.error("❌ No camera frame available")
                return
            
            # Save frame to temporary file
            image_file = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg')
            cv2.imwrite(image_file.name, frame_right)  # Use right camera
            
            logger.info("📸 Frame captured, sending to Gemini...")
            
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
            logger.info("📊 RESULTS:")
            logger.info(f"   Q: {result['transcription']}")
            logger.info(f"   A: {result['answer']}")
            logger.info(f"   Object: {result['object']}")
            logger.info(f"   Position: ({result['position']['x']:.2f}, {result['position']['y']:.2f}, {depth_value:.2f})")
            logger.info(f"   Confidence: {result['position']['confidence']:.2%}")
            
            # Broadcast to web clients
            logger.info("Broadcasting to web clients...")
            self.server.broadcast_result(result)
            logger.info("✅ Broadcast complete!")
            
            # Cleanup temp files
            os.unlink(audio_file)
            os.unlink(image_file.name)
            
        except Exception as e:
            logger.error(f"❌ Processing error: {e}")
            import traceback
            logger.error(traceback.print_exc())
    
    def _draw_3d_text_overlay(self, frame, result):
        """Draw 3D text overlay on frame based on location"""
        if not result:
            return frame
        
        # Get frame dimensions
        h, w = frame.shape[:2]
        
        # Map location to coordinates
        location_map = {
            'top-left': (int(w * 0.15), int(h * 0.15)),
            'top-center': (int(w * 0.5), int(h * 0.15)),
            'top-right': (int(w * 0.85), int(h * 0.15)),
            'center-left': (int(w * 0.15), int(h * 0.5)),
            'center': (int(w * 0.5), int(h * 0.5)),
            'center-right': (int(w * 0.85), int(h * 0.5)),
            'bottom-left': (int(w * 0.15), int(h * 0.85)),
            'bottom-center': (int(w * 0.5), int(h * 0.85)),
            'bottom-right': (int(w * 0.85), int(h * 0.85)),
        }
        
        location = result.get('location', 'center')
        x, y = location_map.get(location, (int(w * 0.5), int(h * 0.5)))
        
        # Get answer text
        answer = result['answer']
        object_name = result['object']
        
        # Word wrap for long answers
        max_chars = 30
        words = answer.split()
        lines = []
        current_line = ""
        
        for word in words:
            if len(current_line) + len(word) + 1 <= max_chars:
                current_line += word + " "
            else:
                if current_line:
                    lines.append(current_line.strip())
                current_line = word + " "
        
        if current_line:
            lines.append(current_line.strip())
        
        # Calculate text size for background box
        font = cv2.FONT_HERSHEY_DUPLEX
        font_scale = 0.6
        thickness = 2
        
        max_width = 0
        total_height = 0
        line_heights = []
        
        for line in lines:
            (tw, th), baseline = cv2.getTextSize(line, font, font_scale, thickness)
            max_width = max(max_width, tw)
            line_heights.append(th + baseline)
            total_height += th + baseline + 5
        
        # Add object name line
        obj_text = f"[{object_name}]"
        (obj_w, obj_h), obj_baseline = cv2.getTextSize(obj_text, font, 0.5, 1)
        max_width = max(max_width, obj_w)
        total_height += obj_h + obj_baseline + 10
        
        # Draw semi-transparent background
        padding = 15
        box_x1 = x - max_width // 2 - padding
        box_y1 = y - total_height // 2 - padding
        box_x2 = x + max_width // 2 + padding
        box_y2 = y + total_height // 2 + padding
        
        # Ensure box is within frame
        box_x1 = max(5, box_x1)
        box_y1 = max(5, box_y1)
        box_x2 = min(w - 5, box_x2)
        box_y2 = min(h - 5, box_y2)
        
        # Create overlay for transparency
        overlay = frame.copy()
        cv2.rectangle(overlay, (box_x1, box_y1), (box_x2, box_y2), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # Draw border
        cv2.rectangle(frame, (box_x1, box_y1), (box_x2, box_y2), (0, 255, 0), 2)
        
        # Draw text lines
        current_y = y - total_height // 2 + 10
        
        for i, line in enumerate(lines):
            # Draw shadow
            cv2.putText(frame, line, (x - max_width // 2 + 2, current_y + 2),
                       font, font_scale, (0, 0, 0), thickness + 1, cv2.LINE_AA)
            
            # Draw main text
            cv2.putText(frame, line, (x - max_width // 2, current_y),
                       font, font_scale, (0, 255, 0), thickness, cv2.LINE_AA)
            
            current_y += line_heights[i] + 5
        
        # Draw object name
        current_y += 5
        cv2.putText(frame, obj_text, (x - obj_w // 2, current_y),
                   font, 0.5, (0, 200, 255), 1, cv2.LINE_AA)
        
        # Draw location indicator dot
        cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)
        cv2.circle(frame, (x, y), 7, (255, 255, 255), 1)
        
        return frame
    
    def run(self):
        """Run the main application"""
        # Start web server in separate thread
        logger.info("Starting web server thread...")
        server_thread = threading.Thread(target=self.server.run, daemon=True)
        server_thread.start()
        
        # Wait a bit for server to start
        time.sleep(2)
        
        logger.info(f"\n🌐 Web interface: http://localhost:{Config.SERVER_PORT}")
        logger.info("   (or http://<jetson-ip>:5000 from tablet)\n")
        
        if self.test_mode:
            logger.info("🧪 TEST MODE ENABLED")
            logger.info("Press 't' to test, 'q' to quit\n")
        else:
            logger.info("Press SPACE to record, 'q' to quit\n")
        
        try:
            logger.info("Entering main loop...")
            frame_count = 0
            
            while True:
                # Display live feed in window for debugging
                frame_left, frame_right, _ = self.camera.get_frames()
                
                if frame_left is not None:
                    frame_count += 1
                    
                    # Create copies for overlay
                    display_left = frame_left.copy()
                    display_right = frame_right.copy()
                    
                    # Add overlay if we have results
                    if self.current_result:
                        display_left = self._draw_3d_text_overlay(display_left, self.current_result)
                        display_right = self._draw_3d_text_overlay(display_right, self.current_result)
                    
                    # Show stereo view
                    stereo_view = cv2.hconcat([display_left, display_right])
                    
                    # Add frame counter
                    cv2.putText(stereo_view, f"Frame: {frame_count}", (10, 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    
                    cv2.imshow('AURA AI - Stereo Camera', stereo_view)
                
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('t') and self.test_mode:
                    logger.info("🧪 't' key pressed - starting test query")
                    self.process_test_query()
                
                elif key == ord(' ') and not self.test_mode:
                    if not self.recording:
                        logger.info("SPACE pressed - starting recording")
                        self.process_query()
                    else:
                        logger.info("SPACE released - stopping recording")
                        self.stop_query()
                
                elif key == ord('q'):
                    logger.info("'q' pressed - shutting down")
                    break
                
                elif key == ord('b') and self.test_mode:
                    # Broadcast test message
                    logger.info("'b' pressed - broadcasting test message")
                    self.server.test_broadcast()
                
                time.sleep(0.01)
        
        except KeyboardInterrupt:
            logger.info("\n👋 Interrupted, shutting down...")
        
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Cleanup resources"""
        logger.info("Cleaning up resources...")
        self.camera.stop()
        self.audio.cleanup()
        cv2.destroyAllWindows()
        logger.info("✅ Cleanup complete")

def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="AURA AI Glasses")
    parser.add_argument('--test', action='store_true',
                       help='Enable test mode (hardcoded queries, no audio)')
    
    args = parser.parse_args()
    
    try:
        app = AURAGlasses(test_mode=args.test)
        app.run()
    except Exception as e:
        logger.error(f"❌ Fatal error: {e}")
        import traceback
        logger.error(traceback.print_exc())
        sys.exit(1)

if __name__ == "__main__":
    main()
