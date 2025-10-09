#!/usr/bin/env python3
"""
AI Glasses Pipeline
Complete pipeline combining stereo vision, voice recognition, AI vision analysis, and 3D object detection
"""

import os
import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk, messagebox
import threading
import queue
import time
import json
import tempfile
import wave
import pyaudio
from PIL import Image, ImageTk, ImageDraw, ImageFont
import google.generativeai as genai
from deepgram import DeepgramClient, PrerecordedOptions, FileSource
from dotenv import load_dotenv
import subprocess
import sys
from pathlib import Path

# Import custom modules
try:
    from visionapi import VisionAPI
    from bbox3d_utils import BBox3DEstimator, BirdEyeView
except ImportError as e:
    print(f"Warning: Could not import vision modules: {e}")
    print("Make sure visionapi.py and bbox3d_utils.py are in the same directory")

# Try to import NanoOwl components (optional)
try:
    from owl_predict import OwlPredictor
    NANOOWL_AVAILABLE = True
except ImportError:
    print("Warning: NanoOwl not available, will use fallback detection")
    OwlPredictor = None
    NANOOWL_AVAILABLE = False

import os
import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk, messagebox
import threading
import queue
import time
import json
import tempfile
import wave
import pyaudio
from PIL import Image, ImageTk, ImageDraw, ImageFont
import google.generativeai as genai
from deepgram import DeepgramClient, PrerecordedOptions, FileSource
from dotenv import load_dotenv
import subprocess
import sys
from pathlib import Path

# Import custom modules
try:
    from visionapi import VisionAPI
    from bbox3d_utils import BBox3DEstimator, BirdEyeView
except ImportError as e:
    print(f"Warning: Could not import vision modules: {e}")
    print("Make sure visionapi.py and bbox3d_utils.py are in the same directory")

# Try to import NanoOwl components (optional)
try:
    from owl_predict import OwlPredictor
    NANOOWL_AVAILABLE = True
except ImportError:
    print("Warning: NanoOwl not available, will use fallback detection")
    OwlPredictor = None
    NANOOWL_AVAILABLE = False

class AIGlassesPipeline:
    def __init__(self):
        """Initialize the AI Glasses Pipeline"""
        # Load environment variables
        load_dotenv()
        
        # Initialize main window
        self.root = tk.Tk()
        self.root.title("AI Glasses Pipeline")
        self.root.geometry("1400x900")
        self.root.configure(bg='#2b2b2b')
        
        # Apply modern theme
        self.setup_modern_theme()
        
        # Initialize variables
        self.is_running = False
        self.recording = False
        self.current_frame_left = None
        self.current_frame_right = None
        self.depth_map = None
        self.last_query = ""
        self.last_answer = ""
        self.current_detections = []
        
        # Camera variables
        self.cap_left = None
        self.cap_right = None
        
        # Audio variables
        self.audio_format = pyaudio.paInt16
        self.channels = 1
        self.rate = 16000
        self.chunk = 1024
        self.audio = pyaudio.PyAudio()
        self.audio_frames = []
        self.audio_stream = None
        
        # Initialize APIs
        self.init_apis()
        
        # Initialize computer vision components
        self.stereo_matcher = cv2.StereoBM_create(numDisparities=16, blockSize=15)
        self.bbox_estimator = BBox3DEstimator()
        self.bev_visualizer = BirdEyeView()
        
        # Initialize threading queues
        self.frame_queue = queue.Queue(maxsize=5)
        self.detection_queue = queue.Queue(maxsize=5)
        
        # Setup UI
        self.setup_ui()
        
        # Start camera thread
        self.camera_thread = None
        self.processing_thread = None
        
    def setup_modern_theme(self):
        """Setup modern dark theme for tkinter"""
        style = ttk.Style()
        
        # Configure dark theme colors
        bg_color = '#2b2b2b'
        fg_color = '#ffffff'
        select_color = '#404040'
        button_color = '#404040'
        
        style.theme_use('clam')
        
        # Configure styles
        style.configure('TFrame', background=bg_color)
        style.configure('TLabel', background=bg_color, foreground=fg_color)
        style.configure('TButton', background=button_color, foreground=fg_color)
        style.map('TButton', background=[('active', select_color)])
        style.configure('TLabelFrame', background=bg_color, foreground=fg_color)
        style.configure('TText', background='#1e1e1e', foreground=fg_color)
        
    def init_apis(self):
        """Initialize all APIs"""
        try:
            # Initialize Deepgram
            deepgram_key = os.getenv('DEEPGRAM_API_KEY')
            if not deepgram_key:
                raise ValueError("DEEPGRAM_API_KEY not found in .env file")
            self.deepgram = DeepgramClient(deepgram_key)
            print("✅ Deepgram API initialized")
            
            # Initialize Gemini
            gemini_key = os.getenv('GEMINI_API_KEY')
            if not gemini_key:
                raise ValueError("GEMINI_API_KEY not found in .env file")
            genai.configure(api_key=gemini_key)
            self.gemini_model = genai.GenerativeModel("models/gemini-1.5-flash")
            print("✅ Gemini API initialized")
            
            # Initialize NanoOwl (if available)
            try:
                if NANOOWL_AVAILABLE and OwlPredictor:
                    self.owl_predictor = OwlPredictor(
                        "google/owlvit-base-patch32",
                        image_encoder_engine="../data/owl_image_encoder_patch32.engine"
                    )
                    print("✅ NanoOwl initialized")
                else:
                    self.owl_predictor = None
                    print("⚠️  NanoOwl not available, using fallback detection")
            except Exception as e:
                print(f"⚠️  NanoOwl initialization failed: {e}")
                self.owl_predictor = None
                
        except Exception as e:
            messagebox.showerror("API Error", f"Failed to initialize APIs: {e}")
            
    def setup_ui(self):
        """Setup the user interface"""
        # Main container
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Left panel for controls
        control_frame = ttk.LabelFrame(main_frame, text="Controls", width=300)
        control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        control_frame.pack_propagate(False)
        
        # Right panel for video display
        video_frame = ttk.LabelFrame(main_frame, text="Live Feed")
        video_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        # Video display
        self.video_label = ttk.Label(video_frame, text="Camera feed will appear here")
        self.video_label.pack(expand=True, pady=10)
        
        # Control buttons
        button_frame = ttk.Frame(control_frame)
        button_frame.pack(fill=tk.X, pady=10)
        
        self.start_button = ttk.Button(button_frame, text="Start Pipeline", 
                                      command=self.start_pipeline)
        self.start_button.pack(fill=tk.X, pady=2)
        
        self.stop_button = ttk.Button(button_frame, text="Stop Pipeline", 
                                     command=self.stop_pipeline, state='disabled')
        self.stop_button.pack(fill=tk.X, pady=2)
        
        # Microphone button
        mic_frame = ttk.LabelFrame(control_frame, text="Voice Input")
        mic_frame.pack(fill=tk.X, pady=10)
        
        self.mic_button = ttk.Button(mic_frame, text="🎤 Hold to Record", 
                                    state='disabled')
        self.mic_button.pack(fill=tk.X, pady=5)
        
        # Bind mouse events for press-to-talk
        self.mic_button.bind('<Button-1>', self.start_recording)
        self.mic_button.bind('<ButtonRelease-1>', self.stop_recording)
        
        # Status display
        status_frame = ttk.LabelFrame(control_frame, text="Status")
        status_frame.pack(fill=tk.X, pady=10)
        
        self.status_text = tk.Text(status_frame, height=8, width=35, 
                                  bg='#1e1e1e', fg='white', wrap=tk.WORD)
        self.status_text.pack(fill=tk.BOTH, expand=True, pady=5)
        
        # Query and answer display
        qa_frame = ttk.LabelFrame(control_frame, text="Last Query & Answer")
        qa_frame.pack(fill=tk.X, pady=10)
        
        self.qa_text = tk.Text(qa_frame, height=6, width=35,
                              bg='#1e1e1e', fg='white', wrap=tk.WORD)
        self.qa_text.pack(fill=tk.BOTH, expand=True, pady=5)
        
        # Detection info
        detection_frame = ttk.LabelFrame(control_frame, text="Detections")
        detection_frame.pack(fill=tk.X, pady=10)
        
        self.detection_text = tk.Text(detection_frame, height=6, width=35,
                                     bg='#1e1e1e', fg='white', wrap=tk.WORD)
        self.detection_text.pack(fill=tk.BOTH, expand=True, pady=5)
        
    def log_status(self, message):
        """Log status message to the status display"""
        timestamp = time.strftime("%H:%M:%S")
        self.status_text.insert(tk.END, f"[{timestamp}] {message}\n")
        self.status_text.see(tk.END)
        self.root.update_idletasks()
        print(f"[{timestamp}] {message}")
        
    def start_pipeline(self):
        """Start the AI glasses pipeline"""
        try:
            self.log_status("Starting AI Glasses Pipeline...")
            
            # Initialize cameras
            self.cap_left = cv2.VideoCapture(0)
            self.cap_right = cv2.VideoCapture(1)
            
            if not self.cap_left.isOpened():
                raise RuntimeError("Cannot open left camera (/dev/video0)")
            if not self.cap_right.isOpened():
                raise RuntimeError("Cannot open right camera (/dev/video1)")
                
            # Set camera properties
            for cap in [self.cap_left, self.cap_right]:
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                cap.set(cv2.CAP_PROP_FPS, 30)
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                
            self.is_running = True
            
            # Start camera thread
            self.camera_thread = threading.Thread(target=self.camera_loop, daemon=True)
            self.camera_thread.start()
            
            # Start processing thread
            self.processing_thread = threading.Thread(target=self.processing_loop, daemon=True)
            self.processing_thread.start()
            
            # Update UI
            self.start_button.config(state='disabled')
            self.stop_button.config(state='normal')
            self.mic_button.config(state='normal')
            
            self.log_status("Pipeline started successfully!")
            
        except Exception as e:
            self.log_status(f"Error starting pipeline: {e}")
            messagebox.showerror("Error", f"Failed to start pipeline: {e}")
            
    def stop_pipeline(self):
        """Stop the AI glasses pipeline"""
        self.log_status("Stopping pipeline...")
        
        self.is_running = False
        
        # Release cameras
        if self.cap_left:
            self.cap_left.release()
        if self.cap_right:
            self.cap_right.release()
            
        # Update UI
        self.start_button.config(state='normal')
        self.stop_button.config(state='disabled')
        self.mic_button.config(state='disabled')
        
        self.log_status("Pipeline stopped")
        
    def camera_loop(self):
        """Camera capture loop"""
        while self.is_running:
            try:
                ret_left, frame_left = self.cap_left.read()
                ret_right, frame_right = self.cap_right.read()
                
                if ret_left and ret_right:
                    self.current_frame_left = frame_left
                    self.current_frame_right = frame_right
                    
                    # Calculate depth map
                    self.calculate_depth_map(frame_left, frame_right)
                    
                    # Put frame in queue for processing
                    if not self.frame_queue.full():
                        self.frame_queue.put((frame_left.copy(), frame_right.copy()))
                        
                time.sleep(0.033)  # ~30 FPS
                
            except Exception as e:
                self.log_status(f"Camera error: {e}")
                break
                
    def calculate_depth_map(self, frame_left, frame_right):
        """Calculate depth map from stereo frames"""
        try:
            # Convert to grayscale
            gray_left = cv2.cvtColor(frame_left, cv2.COLOR_BGR2GRAY)
            gray_right = cv2.cvtColor(frame_right, cv2.COLOR_BGR2GRAY)
            
            # Compute disparity
            disparity = self.stereo_matcher.compute(gray_left, gray_right).astype(np.float32) / 16.0
            
            # Normalize and convert to depth map
            self.depth_map = cv2.normalize(disparity, None, 0, 1, cv2.NORM_MINMAX)
            
        except Exception as e:
            self.log_status(f"Depth calculation error: {e}")
            
    def processing_loop(self):
        """Main processing loop for display updates"""
        while self.is_running:
            try:
                # Display current frame with overlays
                if self.current_frame_left is not None:
                    display_frame = self.current_frame_left.copy()
                    
                    # Draw 3D bounding boxes if detections exist
                    if self.current_detections:
                        display_frame = self.draw_3d_detections(display_frame)
                        
                    # Draw answer text if available
                    if self.last_answer:
                        self.draw_answer_text(display_frame)
                        
                    # Update display
                    self.update_video_display(display_frame)
                    
                time.sleep(0.05)  # 20 FPS display update
                
            except Exception as e:
                self.log_status(f"Processing error: {e}")
                
    def start_recording(self, event):
        """Start audio recording"""
        if not self.is_running:
            return
            
        try:
            self.recording = True
            self.audio_frames = []
            self.mic_button.config(text="🔴 Recording...")
            self.log_status("Recording started...")
            
            # Start audio stream
            self.audio_stream = self.audio.open(
                format=self.audio_format,
                channels=self.channels,
                rate=self.rate,
                input=True,
                frames_per_buffer=self.chunk
            )
            
            # Start recording thread
            recording_thread = threading.Thread(target=self.record_audio, daemon=True)
            recording_thread.start()
            
        except Exception as e:
            self.log_status(f"Recording start error: {e}")
            
    def record_audio(self):
        """Record audio while button is pressed"""
        while self.recording:
            try:
                data = self.audio_stream.read(self.chunk)
                self.audio_frames.append(data)
            except Exception as e:
                self.log_status(f"Audio recording error: {e}")
                break
                
    def stop_recording(self, event):
        """Stop audio recording and process"""
        if not self.recording:
            return
            
        try:
            self.recording = False
            self.mic_button.config(text="🎤 Hold to Record")
            
            if self.audio_stream:
                self.audio_stream.stop_stream()
                self.audio_stream.close()
                
            self.log_status("Recording stopped, processing...")
            
            # Save audio file
            audio_file = self.save_audio_file()
            
            if audio_file:
                # Save current frame for vision analysis
                if self.current_frame_right is not None:
                    frame_file = self.save_current_frame()
                    
                    # Process in separate thread to avoid blocking UI
                    process_thread = threading.Thread(
                        target=self.process_voice_and_vision,
                        args=(audio_file, frame_file),
                        daemon=True
                    )
                    process_thread.start()
                    
        except Exception as e:
            self.log_status(f"Recording stop error: {e}")
            
    def save_audio_file(self):
        """Save recorded audio to temporary file"""
        try:
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
            
            with wave.open(temp_file.name, 'wb') as wf:
                wf.setnchannels(self.channels)
                wf.setsampwidth(self.audio.get_sample_size(self.audio_format))
                wf.setframerate(self.rate)
                wf.writeframes(b''.join(self.audio_frames))
                
            return temp_file.name
            
        except Exception as e:
            self.log_status(f"Audio save error: {e}")
            return None
            
    def save_current_frame(self):
        """Save current frame to temporary file"""
        try:
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg')
            cv2.imwrite(temp_file.name, self.current_frame_right)
            return temp_file.name
            
        except Exception as e:
            self.log_status(f"Frame save error: {e}")
            return None
            
    def process_voice_and_vision(self, audio_file, frame_file):
        """Process voice recognition and vision analysis"""
        try:
            self.log_status("Processing voice recognition...")
            
            # Transcribe audio with Deepgram
            query_text = self.transcribe_audio(audio_file)
            
            if query_text:
                self.last_query = query_text
                self.log_status(f"Query: {query_text}")
                
                # Process vision analysis
                self.log_status("Processing vision analysis...")
                vision_result = self.analyze_frame_with_gemini(frame_file, query_text)
                
                if vision_result:
                    self.last_answer = vision_result.get('answer', 'No answer')
                    object_label = vision_result.get('label', 'unknown')
                    
                    self.log_status(f"Answer: {self.last_answer}")
                    self.log_status(f"Object: {object_label}")
                    
                    # Update Q&A display
                    self.update_qa_display(query_text, self.last_answer)
                    
                    # Run object detection
                    self.run_object_detection(frame_file, object_label)
                    
            # Cleanup temporary files
            try:
                os.unlink(audio_file)
                os.unlink(frame_file)
            except:
                pass
                
        except Exception as e:
            self.log_status(f"Voice/Vision processing error: {e}")
            
    def transcribe_audio(self, audio_file):
        """Transcribe audio using Deepgram"""
        try:
            with open(audio_file, 'rb') as file:
                buffer_data = file.read()
                
            payload = FileSource(buffer_data)
            
            options = PrerecordedOptions(
                model="nova-2",
                language="en",
                smart_format=True,
            )
            
            response = self.deepgram.listen.rest.v("1").transcribe_file(
                payload, options
            )
            
            # Extract transcript
            if response.results and response.results.channels:
                alternatives = response.results.channels[0].alternatives
                if alternatives and len(alternatives) > 0:
                    return alternatives[0].transcript
                    
            return None
            
        except Exception as e:
            self.log_status(f"Transcription error: {e}")
            return None
            
    def analyze_frame_with_gemini(self, frame_file, query_text):
        """Analyze frame with Gemini Vision API"""
        try:
            # Load image
            image = Image.open(frame_file)
            
            # Create prompt similar to visionapi.py
            prompt = f"""
            Analyze this image and answer the user's query CONCISELY. You must respond in valid JSON format with exactly these two keys:
            
            User Query: {query_text}
            
            Required JSON format:
            {{
                "answer": "Your CONCISE response (max 10 words) - always provide your best guess",
                "label": "Primary object name (single word or short phrase)"
            }}
            
            Rules:
            - Keep "answer" extremely brief - maximum 10 words
            - ALWAYS provide an answer - never use "N/A" or "unknown"
            - If information is not clearly visible, make your best educated guess
            - The "label" should contain only the main object/thing visible in the image
            - Always output valid JSON format
            - Do not include any text outside the JSON structure
            """
            
            response = self.gemini_model.generate_content([prompt, image])
            
            # Parse JSON response
            response_text = response.text.strip()
            
            # Clean response if it has markdown formatting
            if response_text.startswith('```json'):
                response_text = response_text.replace('```json', '').replace('```', '').strip()
                
            result = json.loads(response_text)
            return result
            
        except Exception as e:
            self.log_status(f"Gemini analysis error: {e}")
            return None
            
    def run_object_detection(self, frame_file, object_label):
        """Run object detection using NanoOwl or fallback"""
        try:
            self.log_status(f"Detecting objects: {object_label}")
            
            # Load image
            image = Image.open(frame_file)
            
            if self.owl_predictor:
                # Use NanoOwl
                detections = self.detect_with_nanoowl(image, object_label)
            else:
                # Fallback to simple detection
                detections = self.detect_with_fallback(image, object_label)
                
            if detections:
                self.current_detections = self.create_3d_detections(detections)
                self.update_detection_display()
                self.log_status(f"Found {len(detections)} objects")
            else:
                self.current_detections = []
                self.log_status("No objects detected")
                
        except Exception as e:
            self.log_status(f"Object detection error: {e}")
            
    def detect_with_nanoowl(self, image, object_label):
        """Detect objects using NanoOwl"""
        try:
            # Prepare text for detection
            text = [object_label]
            threshold = 0.1
            
            # Encode text
            text_encodings = self.owl_predictor.encode_text(text)
            
            # Run prediction
            output = self.owl_predictor.predict(
                image=image,
                text=text,
                text_encodings=text_encodings,
                threshold=threshold,
                pad_square=False
            )
            
            # Convert output to standard format
            detections = []
            if hasattr(output, 'boxes') and len(output.boxes) > 0:
                for i, box in enumerate(output.boxes):
                    x1, y1, x2, y2 = box
                    score = output.scores[i] if hasattr(output, 'scores') else 0.5
                    
                    detections.append({
                        'bbox': [int(x1), int(y1), int(x2), int(y2)],
                        'score': float(score),
                        'class_name': object_label,
                        'class_id': 0
                    })
                    
            return detections
            
        except Exception as e:
            self.log_status(f"NanoOwl detection error: {e}")
            return []
            
    def detect_with_fallback(self, image, object_label):
        """Fallback detection method"""
        try:
            # Convert PIL to OpenCV
            cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            
            # Simple template matching or contour detection
            # This is a placeholder - in practice, you might use YOLO or other detectors
            gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
            
            # Find contours as a simple detection method
            contours, _ = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            detections = []
            for i, contour in enumerate(contours[:3]):  # Limit to 3 largest
                if cv2.contourArea(contour) > 1000:  # Minimum area threshold
                    x, y, w, h = cv2.boundingRect(contour)
                    
                    detections.append({
                        'bbox': [x, y, x + w, y + h],
                        'score': 0.7,
                        'class_name': object_label,
                        'class_id': 0
                    })
                    
            return detections
            
        except Exception as e:
            self.log_status(f"Fallback detection error: {e}")
            return []
            
    def create_3d_detections(self, detections):
        """Create 3D detections with depth information"""
        detections_3d = []
        
        for i, detection in enumerate(detections):
            try:
                bbox_2d = detection['bbox']
                class_name = detection['class_name']
                score = detection['score']
                
                # Get depth value from depth map
                depth_value = self.get_depth_at_bbox(bbox_2d)
                
                # Create 3D bounding box
                box_3d = self.bbox_estimator.estimate_3d_box(
                    bbox_2d=bbox_2d,
                    depth_value=depth_value,
                    class_name=class_name,
                    object_id=i
                )
                
                # Add additional information
                box_3d['score'] = score
                box_3d['depth_value'] = depth_value
                box_3d['depth_method'] = 'stereo'
                
                detections_3d.append(box_3d)
                
            except Exception as e:
                self.log_status(f"3D detection creation error: {e}")
                
        return detections_3d
        
    def get_depth_at_bbox(self, bbox):
        """Get depth value at bounding box center"""
        try:
            if self.depth_map is None:
                return 0.5  # Default depth
                
            x1, y1, x2, y2 = bbox
            center_x = int((x1 + x2) / 2)
            center_y = int((y1 + y2) / 2)
            
            # Ensure coordinates are within bounds
            height, width = self.depth_map.shape
            center_x = max(0, min(center_x, width - 1))
            center_y = max(0, min(center_y, height - 1))
            
            # Get depth value (average in small region around center)
            region_size = 10
            y_start = max(0, center_y - region_size)
            y_end = min(height, center_y + region_size)
            x_start = max(0, center_x - region_size)
            x_end = min(width, center_x + region_size)
            
            depth_region = self.depth_map[y_start:y_end, x_start:x_end]
            return float(np.mean(depth_region))
            
        except Exception as e:
            self.log_status(f"Depth extraction error: {e}")
            return 0.5
            
    def draw_3d_detections(self, frame):
        """Draw 3D bounding boxes on frame"""
        try:
            for detection in self.current_detections:
                frame = self.bbox_estimator.draw_box_3d(frame, detection)
                
                # Draw answer text near the object
                bbox = detection['bbox_2d']
                x1, y1, x2, y2 = bbox
                
                # Position text above the bounding box
                text_x = x1
                text_y = y1 - 30
                
                if self.last_answer and text_y > 20:
                    # Draw background rectangle for text
                    text_size = cv2.getTextSize(self.last_answer, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                    cv2.rectangle(frame, 
                                (text_x, text_y - text_size[1] - 5),
                                (text_x + text_size[0] + 10, text_y + 5),
                                (0, 0, 0), -1)
                    
                    # Draw text
                    cv2.putText(frame, self.last_answer, (text_x + 5, text_y), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    
        except Exception as e:
            self.log_status(f"3D drawing error: {e}")
            
        return frame
        
    def draw_answer_text(self, frame):
        """Draw answer text in corner of frame"""
        try:
            if self.last_answer:
                # Draw in top-left corner
                text_lines = self.last_answer.split(' ')
                line_height = 25
                y_offset = 30
                
                for i, line in enumerate(text_lines):
                    if i * line_height + y_offset > frame.shape[0] - line_height:
                        break
                        
                    # Draw background
                    text_size = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
                    cv2.rectangle(frame,
                                (10, y_offset + i * line_height - text_size[1] - 5),
                                (10 + text_size[0] + 10, y_offset + i * line_height + 5),
                                (0, 0, 0), -1)
                    
                    # Draw text
                    cv2.putText(frame, line, (15, y_offset + i * line_height),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    
        except Exception as e:
            self.log_status(f"Answer text drawing error: {e}")
            
    def update_video_display(self, frame):
        """Update video display in GUI"""
        try:
            # Resize frame for display
            display_height = 480
            display_width = int(frame.shape[1] * display_height / frame.shape[0])
            frame_resized = cv2.resize(frame, (display_width, display_height))
            
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
            
            # Convert to PIL Image
            pil_image = Image.fromarray(frame_rgb)
            photo = ImageTk.PhotoImage(pil_image)
            
            # Update label
            self.video_label.configure(image=photo, text="")
            self.video_label.image = photo  # Keep reference
            
        except Exception as e:
            self.log_status(f"Display update error: {e}")
            
    def update_qa_display(self, query, answer):
        """Update Q&A display"""
        try:
            self.qa_text.delete(1.0, tk.END)
            self.qa_text.insert(tk.END, f"Q: {query}\n\n")
            self.qa_text.insert(tk.END, f"A: {answer}\n")
            
        except Exception as e:
            self.log_status(f"Q&A display error: {e}")
            
    def update_detection_display(self):
        """Update detection display"""
        try:
            self.detection_text.delete(1.0, tk.END)
            
            for i, detection in enumerate(self.current_detections):
                class_name = detection['class_name']
                score = detection.get('score', 0)
                depth = detection.get('depth_value', 0)
                
                self.detection_text.insert(tk.END, 
                    f"{i+1}. {class_name}\n   Score: {score:.2f}\n   Depth: {depth:.2f}\n\n")
                    
        except Exception as e:
            self.log_status(f"Detection display error: {e}")
            
    def run(self):
        """Run the application"""
        try:
            self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
            self.root.mainloop()
        except KeyboardInterrupt:
            self.on_closing()
            
    def on_closing(self):
        """Handle application closing"""
        try:
            self.stop_pipeline()
            
            # Cleanup audio
            if hasattr(self, 'audio'):
                self.audio.terminate()
                
            self.root.destroy()
            
        except Exception as e:
            print(f"Cleanup error: {e}")

def main():
    """Main function"""
    try:
        print("Starting AI Glasses Pipeline...")
        
        # Check for required environment variables
        load_dotenv()
        required_keys = ['DEEPGRAM_API_KEY', 'GEMINI_API_KEY']
        missing_keys = [key for key in required_keys if not os.getenv(key)]
        
        if missing_keys:
            print(f"Error: Missing required environment variables: {missing_keys}")
            print("Please add them to your .env file")
            return
            
        # Create and run application
        app = AIGlassesPipeline()
        app.run()
        
    except Exception as e:
        print(f"Application error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()