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
        self.depth = tk.IntVar(value=5)
        self.shadow_offset_x = tk.IntVar(value=3)
        self.shadow_offset_y = tk.IntVar(value=3)
        self.shadow_blur = tk.IntVar(value=2)
        self.font_scale = tk.DoubleVar(value=2.0)
        self.thickness = tk.IntVar(value=3)
        self.text_x = tk.IntVar(value=100)
        self.text_y = tk.IntVar(value=200)
        
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
        font = cv2.FONT_HERSHEY_SIMPLEX
        x, y = position
        
        # Get text size for positioning
        (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
        
        # Depth effect - draw multiple layers
        if self.enable_depth.get():
            depth = self.depth.get()
            depth_color = (self.depth_color_b.get(), self.depth_color_g.get(), self.depth_color_r.get())
            
            for i in range(depth, 0, -1):
                depth_pos = (x - i, y - i)
                # Gradually darken the depth layers
                layer_color = tuple(max(0, int(c * (0.3 + 0.7 * (depth - i) / depth))) for c in depth_color)
                cv2.putText(img, text, depth_pos, font, font_scale, layer_color, thickness + 1)
        
        # Shadow effect
        if self.enable_shadow.get():
            shadow_pos = (x + self.shadow_offset_x.get(), y + self.shadow_offset_y.get())
            shadow_color = (self.shadow_color_b.get(), self.shadow_color_g.get(), self.shadow_color_r.get())
            
            # Create shadow with blur if needed
            if self.shadow_blur.get() > 0:
                shadow_img = np.zeros_like(img)
                cv2.putText(shadow_img, text, shadow_pos, font, font_scale, shadow_color, thickness)
                shadow_img = cv2.GaussianBlur(shadow_img, (self.shadow_blur.get() * 2 + 1, self.shadow_blur.get() * 2 + 1), 0)
                # Blend shadow
                mask = shadow_img > 0
                img[mask] = cv2.addWeighted(img, 0.7, shadow_img, 0.3, 0)[mask]
            else:
                cv2.putText(img, text, shadow_pos, font, font_scale, shadow_color, thickness)
        
        # Outline effect
        if self.enable_outline.get():
            outline_color = (0, 0, 0)  # Black outline
            cv2.putText(img, text, (x, y), font, font_scale, outline_color, thickness + 2)
        
        # Main text
        text_color = (self.text_color_b.get(), self.text_color_g.get(), self.text_color_r.get())
        
        if self.enable_gradient.get():
            # Simple gradient effect by drawing text multiple times with slight variations
            for i in range(3):
                gradient_color = tuple(min(255, int(c * (0.8 + 0.2 * i / 3))) for c in text_color)
                cv2.putText(img, text, (x, y - i), font, font_scale, gradient_color, thickness)
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