import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import os

class TextEffect3D:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("3D Text Effect on Image")
        self.root.geometry("1200x800")
        
        # Variables for text and effects
        self.image_path = ""
        self.original_image = None
        self.current_image = None
        self.text = tk.StringVar(value="SAMPLE TEXT")
        
        # Effect parameters
        self.depth = tk.IntVar(value=8)
        self.shadow_offset_x = tk.IntVar(value=5)
        self.shadow_offset_y = tk.IntVar(value=5)
        self.shadow_blur = tk.IntVar(value=3)
        self.shadow_opacity = tk.DoubleVar(value=0.6)
        self.font_scale = tk.DoubleVar(value=3.0)
        self.thickness = tk.IntVar(value=4)
        self.text_x = tk.IntVar(value=100)
        self.text_y = tk.IntVar(value=200)
        self.perspective_skew = tk.DoubleVar(value=0.2)
        self.bevel_size = tk.IntVar(value=2)
        self.selected_font = tk.StringVar(value="HERSHEY_TRIPLEX")
        
        # Color variables
        self.text_color_r = tk.IntVar(value=255)
        self.text_color_g = tk.IntVar(value=255)
        self.text_color_b = tk.IntVar(value=255)
        self.shadow_color_r = tk.IntVar(value=0)
        self.shadow_color_g = tk.IntVar(value=0)
        self.shadow_color_b = tk.IntVar(value=0)
        self.depth_color_r = tk.IntVar(value=128)
        self.depth_color_g = tk.IntVar(value=128)
        self.depth_color_b = tk.IntVar(value=128)
        
        # Effect toggles
        self.enable_shadow = tk.BooleanVar(value=True)
        self.enable_depth = tk.BooleanVar(value=True)
        self.enable_outline = tk.BooleanVar(value=True)
        self.enable_gradient = tk.BooleanVar(value=False)
        self.enable_bevel = tk.BooleanVar(value=True)
        self.enable_perspective = tk.BooleanVar(value=False)
        self.enable_emboss = tk.BooleanVar(value=False)
        
        self.create_ui()
        
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
        self.image_label = ttk.Label(self.image_frame, text="Load an image to start")
        self.image_label.pack(expand=True)
        
        # File selection
        file_frame = ttk.LabelFrame(control_frame, text="Image")
        file_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Button(file_frame, text="Load Image", command=self.load_image).pack(pady=5)
        
        # Text input
        text_frame = ttk.LabelFrame(control_frame, text="Text")
        text_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Entry(text_frame, textvariable=self.text, width=25).pack(pady=5)
        
        # Position controls
        pos_frame = ttk.LabelFrame(control_frame, text="Position")
        pos_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(pos_frame, text="X:").pack()
        ttk.Scale(pos_frame, from_=0, to=1000, variable=self.text_x, 
                 orient=tk.HORIZONTAL, command=self.update_image).pack(fill=tk.X)
        
        ttk.Label(pos_frame, text="Y:").pack()
        ttk.Scale(pos_frame, from_=0, to=800, variable=self.text_y, 
                 orient=tk.HORIZONTAL, command=self.update_image).pack(fill=tk.X)
        
        # Font controls
        font_frame = ttk.LabelFrame(control_frame, text="Font")
        font_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(font_frame, text="Font:").pack()
        font_combo = ttk.Combobox(font_frame, textvariable=self.selected_font, 
                                 values=["HERSHEY_SIMPLEX", "HERSHEY_PLAIN", "HERSHEY_DUPLEX", 
                                        "HERSHEY_COMPLEX", "HERSHEY_TRIPLEX", "HERSHEY_COMPLEX_SMALL",
                                        "HERSHEY_SCRIPT_SIMPLEX", "HERSHEY_SCRIPT_COMPLEX"], 
                                 state="readonly", width=20)
        font_combo.pack(pady=2)
        font_combo.bind('<<ComboboxSelected>>', self.update_image)
        
        ttk.Label(font_frame, text="Scale:").pack()
        ttk.Scale(font_frame, from_=0.5, to=5.0, variable=self.font_scale, 
                 orient=tk.HORIZONTAL, command=self.update_image).pack(fill=tk.X)
        
        ttk.Label(font_frame, text="Thickness:").pack()
        ttk.Scale(font_frame, from_=1, to=10, variable=self.thickness, 
                 orient=tk.HORIZONTAL, command=self.update_image).pack(fill=tk.X)
        
        # Effect toggles
        effects_frame = ttk.LabelFrame(control_frame, text="Effects")
        effects_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Checkbutton(effects_frame, text="Enable Shadow", 
                       variable=self.enable_shadow, command=self.update_image).pack(anchor=tk.W)
        ttk.Checkbutton(effects_frame, text="Enable Depth", 
                       variable=self.enable_depth, command=self.update_image).pack(anchor=tk.W)
        ttk.Checkbutton(effects_frame, text="Enable Outline", 
                       variable=self.enable_outline, command=self.update_image).pack(anchor=tk.W)
        ttk.Checkbutton(effects_frame, text="Enable Gradient", 
                       variable=self.enable_gradient, command=self.update_image).pack(anchor=tk.W)
        ttk.Checkbutton(effects_frame, text="Enable Bevel", 
                       variable=self.enable_bevel, command=self.update_image).pack(anchor=tk.W)
        ttk.Checkbutton(effects_frame, text="Enable Perspective", 
                       variable=self.enable_perspective, command=self.update_image).pack(anchor=tk.W)
        ttk.Checkbutton(effects_frame, text="Enable Emboss", 
                       variable=self.enable_emboss, command=self.update_image).pack(anchor=tk.W)
        
        # 3D Effect controls
        effect_frame = ttk.LabelFrame(control_frame, text="3D Parameters")
        effect_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(effect_frame, text="Depth:").pack()
        ttk.Scale(effect_frame, from_=0, to=20, variable=self.depth, 
                 orient=tk.HORIZONTAL, command=self.update_image).pack(fill=tk.X)
        
        ttk.Label(effect_frame, text="Shadow X:").pack()
        ttk.Scale(effect_frame, from_=-20, to=20, variable=self.shadow_offset_x, 
                 orient=tk.HORIZONTAL, command=self.update_image).pack(fill=tk.X)
        
        ttk.Label(effect_frame, text="Shadow Y:").pack()
        ttk.Scale(effect_frame, from_=-20, to=20, variable=self.shadow_offset_y, 
                 orient=tk.HORIZONTAL, command=self.update_image).pack(fill=tk.X)
        
        ttk.Label(effect_frame, text="Shadow Blur:").pack()
        ttk.Scale(effect_frame, from_=0, to=10, variable=self.shadow_blur, 
                 orient=tk.HORIZONTAL, command=self.update_image).pack(fill=tk.X)
        
        ttk.Label(effect_frame, text="Shadow Opacity:").pack()
        ttk.Scale(effect_frame, from_=0.0, to=1.0, variable=self.shadow_opacity, 
                 orient=tk.HORIZONTAL, command=self.update_image).pack(fill=tk.X)
        
        ttk.Label(effect_frame, text="Perspective Skew:").pack()
        ttk.Scale(effect_frame, from_=0.0, to=1.0, variable=self.perspective_skew, 
                 orient=tk.HORIZONTAL, command=self.update_image).pack(fill=tk.X)
        
        ttk.Label(effect_frame, text="Bevel Size:").pack()
        ttk.Scale(effect_frame, from_=0, to=5, variable=self.bevel_size, 
                 orient=tk.HORIZONTAL, command=self.update_image).pack(fill=tk.X)
        
        # Color controls (simplified)
        color_frame = ttk.LabelFrame(control_frame, text="Colors")
        color_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Text color
        ttk.Label(color_frame, text="Text Color (RGB):").pack()
        color_text_frame = ttk.Frame(color_frame)
        color_text_frame.pack(fill=tk.X)
        ttk.Scale(color_text_frame, from_=0, to=255, variable=self.text_color_r, 
                 orient=tk.HORIZONTAL, command=self.update_image, length=80).pack(side=tk.LEFT)
        ttk.Scale(color_text_frame, from_=0, to=255, variable=self.text_color_g, 
                 orient=tk.HORIZONTAL, command=self.update_image, length=80).pack(side=tk.LEFT)
        ttk.Scale(color_text_frame, from_=0, to=255, variable=self.text_color_b, 
                 orient=tk.HORIZONTAL, command=self.update_image, length=80).pack(side=tk.LEFT)
        
        # Bind text change
        self.text.trace('w', self.update_image)
        
    def load_image(self):
        file_path = filedialog.askopenfilename(
            title="Select Image",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff")]
        )
        
        if file_path:
            self.image_path = file_path
            self.original_image = cv2.imread(file_path)
            if self.original_image is not None:
                self.update_image()
            else:
                messagebox.showerror("Error", "Could not load image")
    
    def create_3d_text_effect(self, image, text, position, font_scale, thickness):
        if not text:
            return image
            
        img = image.copy()
        
        # Map font selection to OpenCV fonts
        font_map = {
            "HERSHEY_SIMPLEX": cv2.FONT_HERSHEY_SIMPLEX,
            "HERSHEY_PLAIN": cv2.FONT_HERSHEY_PLAIN,
            "HERSHEY_DUPLEX": cv2.FONT_HERSHEY_DUPLEX,
            "HERSHEY_COMPLEX": cv2.FONT_HERSHEY_COMPLEX,
            "HERSHEY_TRIPLEX": cv2.FONT_HERSHEY_TRIPLEX,
            "HERSHEY_COMPLEX_SMALL": cv2.FONT_HERSHEY_COMPLEX_SMALL,
            "HERSHEY_SCRIPT_SIMPLEX": cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,
            "HERSHEY_SCRIPT_COMPLEX": cv2.FONT_HERSHEY_SCRIPT_COMPLEX
        }
        
        font = font_map.get(self.selected_font.get(), cv2.FONT_HERSHEY_TRIPLEX)
        x, y = position
        
        # Get text size for positioning
        (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
        
        # Create separate layers for better compositing
        shadow_layer = np.zeros_like(img, dtype=np.uint8)
        depth_layer = np.zeros_like(img, dtype=np.uint8)
        text_layer = np.zeros_like(img, dtype=np.uint8)
        
        # Shadow effect with proper blending
        if self.enable_shadow.get():
            shadow_pos = (x + self.shadow_offset_x.get(), y + self.shadow_offset_y.get())
            shadow_color = (self.shadow_color_b.get(), self.shadow_color_g.get(), self.shadow_color_r.get())
            
            # Draw shadow on separate layer
            cv2.putText(shadow_layer, text, shadow_pos, font, font_scale, shadow_color, thickness)
            
            # Apply blur if needed
            if self.shadow_blur.get() > 0:
                blur_size = self.shadow_blur.get() * 2 + 1
                shadow_layer = cv2.GaussianBlur(shadow_layer, (blur_size, blur_size), 0)
            
            # Blend shadow with main image using opacity
            shadow_mask = cv2.cvtColor(shadow_layer, cv2.COLOR_BGR2GRAY) > 0
            opacity = self.shadow_opacity.get()
            img[shadow_mask] = cv2.addWeighted(
                img[shadow_mask], 1.0 - opacity, 
                shadow_layer[shadow_mask], opacity, 0
            )
        
        # Depth effect - Enhanced with proper perspective
        if self.enable_depth.get():
            depth = self.depth.get()
            depth_color = (self.depth_color_b.get(), self.depth_color_g.get(), self.depth_color_r.get())
            
            # Create depth layers with perspective adjustment
            for i in range(depth, 0, -1):
                # Add perspective skew if enabled
                if self.enable_perspective.get():
                    skew = self.perspective_skew.get()
                    depth_pos = (x - i - int(i * skew), y - i + int(i * skew * 0.3))
                else:
                    depth_pos = (x - i, y - i)
                
                # Progressive color darkening for depth illusion
                depth_ratio = (depth - i + 1) / depth
                layer_color = tuple(max(0, int(c * (0.2 + 0.6 * depth_ratio))) for c in depth_color)
                
                cv2.putText(img, text, depth_pos, font, font_scale, layer_color, thickness + 1)
        
        # Bevel effect - creates raised edge appearance
        if self.enable_bevel.get():
            bevel_size = self.bevel_size.get()
            text_color = (self.text_color_b.get(), self.text_color_g.get(), self.text_color_r.get())
            
            # Light source from top-left
            highlight_color = tuple(min(255, int(c * 1.3)) for c in text_color)
            shadow_bevel_color = tuple(max(0, int(c * 0.7)) for c in text_color)
            
            # Draw bevel highlights and shadows
            for offset in range(1, bevel_size + 1):
                # Top-left highlight
                cv2.putText(img, text, (x - offset, y - offset), font, font_scale, 
                           highlight_color, thickness // 2)
                # Bottom-right shadow
                cv2.putText(img, text, (x + offset, y + offset), font, font_scale, 
                           shadow_bevel_color, thickness // 2)
        
        # Emboss effect
        if self.enable_emboss.get():
            # Create embossed appearance by drawing offset lighter/darker versions
            text_color = (self.text_color_b.get(), self.text_color_g.get(), self.text_color_r.get())
            light_color = tuple(min(255, int(c * 1.4)) for c in text_color)
            dark_color = tuple(max(0, int(c * 0.6)) for c in text_color)
            
            # Light emboss
            cv2.putText(img, text, (x - 1, y - 1), font, font_scale, light_color, thickness)
            # Dark emboss
            cv2.putText(img, text, (x + 1, y + 1), font, font_scale, dark_color, thickness)
        
        # Outline effect
        if self.enable_outline.get():
            outline_color = (0, 0, 0)  # Black outline
            cv2.putText(img, text, (x, y), font, font_scale, outline_color, thickness + 2)
        
        # Main text with gradient or solid color
        text_color = (self.text_color_b.get(), self.text_color_g.get(), self.text_color_r.get())
        
        if self.enable_gradient.get():
            # Vertical gradient effect
            gradient_steps = max(3, text_height // 10)
            for i in range(gradient_steps):
                # Calculate gradient position and color
                gradient_ratio = i / (gradient_steps - 1) if gradient_steps > 1 else 0
                gradient_y = y - int(text_height * gradient_ratio * 0.8)
                gradient_color = tuple(int(c * (0.7 + 0.3 * (1 - gradient_ratio))) for c in text_color)
                
                # Create mask for this gradient section
                temp_img = np.zeros_like(img)
                cv2.putText(temp_img, text, (x, gradient_y), font, font_scale, gradient_color, thickness)
                
                # Blend gradient section
                mask = cv2.cvtColor(temp_img, cv2.COLOR_BGR2GRAY) > 0
                img[mask] = temp_img[mask]
        else:
            cv2.putText(img, text, (x, y), font, font_scale, text_color, thickness)
        
        return img
    
    def update_image(self, *args):
        if self.original_image is None:
            return
            
        # Create 3D text effect
        result_image = self.create_3d_text_effect(
            self.original_image,
            self.text.get(),
            (self.text_x.get(), self.text_y.get()),
            self.font_scale.get(),
            self.thickness.get()
        )
        
        # Convert to display format
        self.display_image(result_image)
    
    def display_image(self, cv_image):
        # Convert BGR to RGB
        rgb_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        
        # Resize image to fit display
        height, width = rgb_image.shape[:2]
        max_width, max_height = 800, 600
        
        if width > max_width or height > max_height:
            scale = min(max_width / width, max_height / height)
            new_width = int(width * scale)
            new_height = int(height * scale)
            rgb_image = cv2.resize(rgb_image, (new_width, new_height))
        
        # Convert to PIL and then to PhotoImage
        pil_image = Image.fromarray(rgb_image)
        photo = ImageTk.PhotoImage(pil_image)
        
        # Update label
        self.image_label.configure(image=photo, text="")
        self.image_label.image = photo  # Keep a reference
    
    def run(self):
        self.root.mainloop()

if __name__ == "__main__":
    app = TextEffect3D()
    app.run()