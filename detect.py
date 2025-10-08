import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk, ImageDraw, ImageFont, ImageFilter
import json
import os
from ultralytics import YOLO

class ObjectDetector3D:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("3D Object Detection Labels")
        self.root.geometry("1400x900")
        
        # Load YOLO model
        try:
            self.model = YOLO('yolo11n.pt')  # Using nano model for speed
            print("YOLO model loaded successfully")
        except Exception as e:
            print(f"Error loading YOLO model: {e}")
            messagebox.showerror("Error", "Could not load YOLO model. Make sure ultralytics is installed.")
            return
        
        # Variables
        self.image_path = ""
        self.original_image = None
        self.current_detections = []
        self.depth_z = tk.DoubleVar(value=5.0)  # Global depth for all labels
        self.confidence_threshold = tk.DoubleVar(value=0.5)
        self.show_confidence = tk.BooleanVar(value=True)
        
        # Webcam variables
        self.cap = None
        self.is_webcam_running = False
        self.webcam_thread = None
        self.fps = tk.IntVar(value=15)  # FPS for webcam
        self.show_bbox = tk.BooleanVar(value=True)
        
        # Load 3D text parameters from params.json
        self.text_params = self.load_text_parameters()
        
        self.create_ui()
        
    def load_text_parameters(self):
        """Load text parameters from params.json"""
        params_path = "/home/ubuntu/githubrepos/aiglasses/params.json"
        default_params = {
            "font_size": 48,
            "scale_factor": 100.0,
            "perspective_type": "taper_top",
            "perspective_skew": 0.1,
            "shadow_offset_x": 5,
            "shadow_offset_y": 5,
            "shadow_blur": 3,
            "shadow_opacity": 0.6,
            "text_color_r": 255,
            "text_color_g": 255,
            "text_color_b": 255,
            "shadow_color_r": 0,
            "shadow_color_g": 0,
            "shadow_color_b": 0,
            "depth_color_r": 128,
            "depth_color_g": 128,
            "depth_color_b": 128,
            "enable_shadow": True,
            "enable_depth": True,
            "enable_outline": True,
            "enable_gradient": False,
            "enable_bevel": False,
            "enable_perspective": False,
            "enable_emboss": False,
            "auto_shadow_direction": True
        }
        
        try:
            if os.path.exists(params_path):
                with open(params_path, 'r') as f:
                    params = json.load(f)
                print(f"Loaded parameters from {params_path}")
                return {**default_params, **params}  # Merge with defaults
            else:
                print(f"Parameters file not found at {params_path}, using defaults")
                return default_params
        except Exception as e:
            print(f"Error loading parameters: {e}, using defaults")
            return default_params
    
    def create_ui(self):
        # Main frame
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Left panel for controls
        control_frame = ttk.Frame(main_frame, width=300)
        control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        control_frame.pack_propagate(False)
        
        # Right panel for image display
        self.image_frame = ttk.Frame(main_frame)
        self.image_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        # Image label
        self.image_label = ttk.Label(self.image_frame, text="Load an image to start detection")
        self.image_label.pack(expand=True)
        
        # File controls
        file_frame = ttk.LabelFrame(control_frame, text="Input Source")
        file_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Static image controls
        ttk.Button(file_frame, text="Load Image", command=self.load_image).pack(pady=2)
        ttk.Button(file_frame, text="Run Detection", command=self.run_detection).pack(pady=2)
        
        ttk.Separator(file_frame, orient='horizontal').pack(fill=tk.X, pady=5)
        
        # Webcam controls
        webcam_frame = ttk.Frame(file_frame)
        webcam_frame.pack(fill=tk.X, pady=2)
        
        self.webcam_button = ttk.Button(webcam_frame, text="Start Webcam", 
                                       command=self.toggle_webcam)
        self.webcam_button.pack(side=tk.LEFT, padx=(0, 5))
        
        ttk.Label(webcam_frame, text="FPS:").pack(side=tk.LEFT)
        fps_scale = ttk.Scale(webcam_frame, from_=5, to=30, variable=self.fps, 
                             orient=tk.HORIZONTAL, length=100)
        fps_scale.pack(side=tk.LEFT, padx=5)
        
        # Detection controls
        detect_frame = ttk.LabelFrame(control_frame, text="Detection Settings")
        detect_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(detect_frame, text="Confidence Threshold:").pack()
        ttk.Scale(detect_frame, from_=0.1, to=1.0, variable=self.confidence_threshold, 
                 orient=tk.HORIZONTAL, command=self.update_detection).pack(fill=tk.X)
        
        ttk.Checkbutton(detect_frame, text="Show Confidence Scores", 
                       variable=self.show_confidence, command=self.update_detection).pack(anchor=tk.W)
        
        # 3D Text controls
        text_frame = ttk.LabelFrame(control_frame, text="3D Text Settings")
        text_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(text_frame, text="Global Depth (Z):").pack()
        ttk.Scale(text_frame, from_=0.5, to=20.0, variable=self.depth_z, 
                 orient=tk.HORIZONTAL, command=self.update_detection).pack(fill=tk.X)
        
        # Parameter display
        params_frame = ttk.LabelFrame(control_frame, text="Loaded Parameters")
        params_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Show key loaded parameters
        info_text = f"Font Size: {self.text_params.get('font_size', 48)}\n"
        info_text += f"Scale Factor: {self.text_params.get('scale_factor', 100):.1f}\n"
        info_text += f"Shadow: {'On' if self.text_params.get('enable_shadow', True) else 'Off'}\n"
        info_text += f"Depth Effect: {'On' if self.text_params.get('enable_depth', True) else 'Off'}\n"
        info_text += f"Perspective: {self.text_params.get('perspective_type', 'none')}"
        
        ttk.Label(params_frame, text=info_text, justify=tk.LEFT).pack(pady=5)
        
        # Reload button
        ttk.Button(params_frame, text="Reload Parameters", 
                  command=self.reload_parameters).pack(pady=2)
        
    def load_image(self):
        file_path = filedialog.askopenfilename(
            title="Select Image",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff")]
        )
        
        if file_path:
            self.image_path = file_path
            self.original_image = cv2.imread(file_path)
            if self.original_image is not None:
                # Clear previous detections
                self.current_detections = []
                self.display_image(self.original_image)
            else:
                messagebox.showerror("Error", "Could not load image")
    
    def run_detection(self):
        if self.original_image is None:
            messagebox.showwarning("Warning", "Please load an image first")
            return
        
        try:
            # Run YOLO detection
            results = self.model(self.original_image, conf=self.confidence_threshold.get())
            
            # Extract detections
            self.current_detections = []
            if len(results) > 0 and results[0].boxes is not None:
                boxes = results[0].boxes
                for i in range(len(boxes)):
                    # Get bounding box coordinates
                    x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy()
                    confidence = boxes.conf[i].cpu().numpy()
                    class_id = int(boxes.cls[i].cpu().numpy())
                    
                    # Get class name
                    class_name = self.model.names[class_id]
                    
                    self.current_detections.append({
                        'bbox': (int(x1), int(y1), int(x2), int(y2)),
                        'confidence': float(confidence),
                        'class_name': class_name,
                        'class_id': class_id
                    })
            
            print(f"Detected {len(self.current_detections)} objects")
            self.update_detection()
            
        except Exception as e:
            messagebox.showerror("Error", f"Detection failed: {str(e)}")
    
    def calculate_auto_shadow_direction(self, text_x, text_y, image_width, image_height, z_depth):
        """Calculate shadow direction based on text position relative to image center"""
        center_x = image_width / 2
        center_y = image_height / 2
        
        rel_x = (text_x - center_x) / center_x
        rel_y = (text_y - center_y) / center_y
        
        base_distance = max(3, min(15, z_depth * 2))
        
        # Scale shadow by 1.5x for better visibility
        shadow_x = rel_x * base_distance * 1.1
        shadow_y = rel_y * base_distance * 1.1
        
        # Increased bias for more visible shadows even near center
        shadow_x += base_distance * 0.35  # Increased from 0.3
        shadow_y += base_distance * 0.25   # Increased from 0.2
        
        return int(shadow_x), int(shadow_y)
    
    def create_3d_text_label(self, image, text, position, z_depth):
        """Create 3D text label using the loaded parameters"""
        if not text:
            return image
            
        img_cv = image.copy()
        img_pil = Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))
        
        x, y = position
        
        # Calculate depth-based scaling
        k_factor = self.text_params.get('scale_factor', 100.0)
        calculated_scale = k_factor / (z_depth + 1e-3)
        
        # Calculate final font size
        base_font_size = self.text_params.get('font_size', 48)
        final_font_size = int(base_font_size * calculated_scale / 100.0)
        final_font_size = max(12, min(100, final_font_size))  # Reasonable range for detection labels
        
        # Load font
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", final_font_size)
        except:
            try:
                font = ImageFont.load_default()
            except:
                font = ImageFont.load_default()
        
        # Get text dimensions
        bbox = font.getbbox(text)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        
        # Create overlay
        overlay_size = (img_pil.width, img_pil.height)
        text_overlay = Image.new('RGBA', overlay_size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(text_overlay)
        
        # Colors from parameters
        text_color = (
            self.text_params.get('text_color_r', 255),
            self.text_params.get('text_color_g', 255),
            self.text_params.get('text_color_b', 255),
            255
        )
        
        shadow_color = (
            self.text_params.get('shadow_color_r', 0),
            self.text_params.get('shadow_color_g', 0),
            self.text_params.get('shadow_color_b', 0),
            int(255 * self.text_params.get('shadow_opacity', 0.6))
        )
        
        depth_color = (
            self.text_params.get('depth_color_r', 128),
            self.text_params.get('depth_color_g', 128),
            self.text_params.get('depth_color_b', 128),
            200
        )
        
        # Calculate shadow direction
        if self.text_params.get('auto_shadow_direction', True):
            auto_shadow_x, auto_shadow_y = self.calculate_auto_shadow_direction(
                x, y, img_pil.width, img_pil.height, z_depth
            )
            actual_shadow_x = auto_shadow_x
            actual_shadow_y = auto_shadow_y
        else:
            actual_shadow_x = self.text_params.get('shadow_offset_x', 5)
            actual_shadow_y = self.text_params.get('shadow_offset_y', 5)
        
        # Draw effects
        # Shadow
        if self.text_params.get('enable_shadow', True):
            shadow_x = x + actual_shadow_x
            shadow_y = y + actual_shadow_y
            draw.text((shadow_x, shadow_y), text, font=font, fill=shadow_color)
        
        # Depth layers
        if self.text_params.get('enable_depth', True):
            depth_layers = min(6, int(z_depth))
            for i in range(depth_layers, 0, -1):
                depth_x = x - i * 1
                depth_y = y - i * 1
                depth_alpha = int(120 * (depth_layers - i + 1) / depth_layers)
                layer_color = (depth_color[0], depth_color[1], depth_color[2], depth_alpha)
                draw.text((depth_x, depth_y), text, font=font, fill=layer_color)
        
        # Outline
        if self.text_params.get('enable_outline', True):
            outline_color = (0, 0, 0, 255)
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    if dx != 0 or dy != 0:
                        draw.text((x + dx, y + dy), text, font=font, fill=outline_color)
        
        # Main text
        draw.text((x, y), text, font=font, fill=text_color)
        
        # Apply blur to shadow if needed
        shadow_blur = self.text_params.get('shadow_blur', 0)
        if self.text_params.get('enable_shadow', True) and shadow_blur > 0:
            text_overlay = text_overlay.filter(ImageFilter.GaussianBlur(radius=shadow_blur))
        
        # Convert back and blend
        overlay_cv = cv2.cvtColor(np.array(text_overlay), cv2.COLOR_RGBA2BGRA)
        alpha = overlay_cv[:, :, 3] / 255.0
        alpha_3ch = np.dstack([alpha, alpha, alpha])
        overlay_bgr = overlay_cv[:, :, :3]
        result = img_cv * (1 - alpha_3ch) + overlay_bgr * alpha_3ch
        
        return result.astype(np.uint8)
    
    def update_detection(self, *args):
        if self.original_image is None or len(self.current_detections) == 0:
            return
        
        # Start with original image
        result_image = self.original_image.copy()
        
        # Get current depth
        current_depth = self.depth_z.get()
        
        # Process each detection
        for detection in self.current_detections:
            if detection['confidence'] >= self.confidence_threshold.get():
                x1, y1, x2, y2 = detection['bbox']
                class_name = detection['class_name']
                confidence = detection['confidence']
                
                # Create label text
                if self.show_confidence.get():
                    label_text = f"{class_name} {confidence:.2f}"
                else:
                    label_text = class_name
                
                # Position label at top-left corner of bounding box
                label_x = x1
                label_y = y1 - 10  # Slightly above the box
                
                # Ensure label is within image bounds
                if label_y < 20:
                    label_y = y1 + 25  # Below the box if no space above
                
                # Apply 3D text effect
                result_image = self.create_3d_text_label(
                    result_image, 
                    label_text, 
                    (label_x, label_y), 
                    current_depth
                )
                
                # Draw bounding box (optional, simple line)
                cv2.rectangle(result_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        self.display_image(result_image)
    
    def display_image(self, cv_image):
        # Convert BGR to RGB
        rgb_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        
        # Resize to fit display
        height, width = rgb_image.shape[:2]
        max_width, max_height = 1000, 700
        
        if width > max_width or height > max_height:
            scale = min(max_width / width, max_height / height)
            new_width = int(width * scale)
            new_height = int(height * scale)
            rgb_image = cv2.resize(rgb_image, (new_width, new_height))
        
        # Convert to PhotoImage
        pil_image = Image.fromarray(rgb_image)
        photo = ImageTk.PhotoImage(pil_image)
        
        # Update display
        self.image_label.configure(image=photo, text="")
        self.image_label.image = photo
    
    def reload_parameters(self):
        """Reload parameters from params.json"""
        self.text_params = self.load_text_parameters()
        messagebox.showinfo("Success", "Parameters reloaded from params.json")
        self.update_detection()  # Refresh display with new parameters
    
    def toggle_webcam(self):
        """Toggle webcam on/off"""
        if not self.is_webcam_running:
            self.start_webcam()
        else:
            self.stop_webcam()
    
    def start_webcam(self):
        """Start webcam capture and detection"""
        try:
            # Initialize webcam
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                messagebox.showerror("Error", "Could not open webcam")
                return
            
            # Set webcam properties
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            
            self.is_webcam_running = True
            self.webcam_button.config(text="Stop Webcam")
            
            # Start webcam processing
            self.process_webcam_feed()
            
        except Exception as e:
            messagebox.showerror("Error", f"Could not start webcam: {str(e)}")
    
    def stop_webcam(self):
        """Stop webcam capture"""
        self.is_webcam_running = False
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        
        self.webcam_button.config(text="Start Webcam")
        self.image_label.configure(text="Webcam stopped", image="")
        self.image_label.image = None
    
    def process_webcam_feed(self):
        """Process webcam feed with real-time detection"""
        if not self.is_webcam_running or self.cap is None:
            return
        
        try:
            ret, frame = self.cap.read()
            if not ret:
                print("Failed to read from webcam")
                self.stop_webcam()
                return
            
            # Flip frame horizontally for mirror effect
            frame = cv2.flip(frame, 1)
            
            # Run YOLO detection on current frame
            results = self.model(frame, conf=self.confidence_threshold.get(), verbose=False)
            
            # Process detections
            result_image = frame.copy()
            current_depth = self.depth_z.get()
            
            if len(results) > 0 and results[0].boxes is not None:
                boxes = results[0].boxes
                for i in range(len(boxes)):
                    # Get detection info
                    x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy()
                    confidence = boxes.conf[i].cpu().numpy()
                    class_id = int(boxes.cls[i].cpu().numpy())
                    
                    # Filter by confidence
                    if confidence < self.confidence_threshold.get():
                        continue
                    
                    # Get class name
                    class_name = self.model.names[class_id]
                    
                    # Create label text
                    if self.show_confidence.get():
                        label_text = f"{class_name} {confidence:.2f}"
                    else:
                        label_text = class_name
                    
                    # Position label
                    label_x = int(x1)
                    label_y = int(y1) - 10
                    if label_y < 20:
                        label_y = int(y1) + 25
                    
                    # Apply 3D text effect
                    result_image = self.create_3d_text_label(
                        result_image, 
                        label_text, 
                        (label_x, label_y), 
                        current_depth
                    )
                    
                    # Draw bounding box if enabled
                    if self.show_bbox.get():
                        cv2.rectangle(result_image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            
            # Display the processed frame
            self.display_image(result_image)
            
            # Schedule next frame processing
            delay = int(1000 / self.fps.get())  # Convert FPS to milliseconds
            self.root.after(delay, self.process_webcam_feed)
            
        except Exception as e:
            print(f"Error in webcam processing: {e}")
            self.stop_webcam()
    
    def run(self):
        self.root.mainloop()

if __name__ == "__main__":
    try:
        print("Starting 3D Object Detection Application...")
        app = ObjectDetector3D()
        app.run()
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()