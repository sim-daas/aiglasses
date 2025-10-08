import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk, ImageDraw, ImageFont, ImageFilter
import os
import json

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
        self.depth_z = tk.DoubleVar(value=5.0)  # Simulated Z depth
        self.scale_factor = tk.DoubleVar(value=100.0)  # k value for scale = k / (Z + 1e-3)
        self.shadow_offset_x = tk.IntVar(value=5)
        self.shadow_offset_y = tk.IntVar(value=5)
        self.shadow_blur = tk.IntVar(value=3)
        self.shadow_opacity = tk.DoubleVar(value=0.6)
        self.font_size = tk.IntVar(value=48)  # Base font size in pixels
        self.text_x = tk.IntVar(value=100)
        self.text_y = tk.IntVar(value=200)
        self.perspective_skew = tk.DoubleVar(value=0.1)
        self.perspective_type = tk.StringVar(value="taper_top")  # taper_top, taper_bottom, none
        
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
        self.auto_shadow_direction = tk.BooleanVar(value=True)  # Auto calculate shadow direction
        
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
        file_frame = ttk.LabelFrame(control_frame, text="Image & Settings")
        file_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Button(file_frame, text="Load Image", command=self.load_image).pack(pady=2)
        
        # Settings save/load
        settings_frame = ttk.Frame(file_frame)
        settings_frame.pack(fill=tk.X, pady=2)
        ttk.Button(settings_frame, text="Save Settings", command=self.save_settings, width=12).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(settings_frame, text="Load Settings", command=self.load_settings, width=12).pack(side=tk.LEFT)
        
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
        
        ttk.Label(font_frame, text="Base Font Size:").pack()
        ttk.Scale(font_frame, from_=16, to=120, variable=self.font_size, 
                 orient=tk.HORIZONTAL, command=self.update_image).pack(fill=tk.X)
        
        ttk.Label(font_frame, text="Scale Factor (k):").pack()
        ttk.Scale(font_frame, from_=50, to=300, variable=self.scale_factor, 
                 orient=tk.HORIZONTAL, command=self.update_image).pack(fill=tk.X)
        
        ttk.Label(font_frame, text="Perspective Type:").pack()
        perspective_combo = ttk.Combobox(font_frame, textvariable=self.perspective_type, 
                                       values=["none", "taper_top", "taper_bottom"], 
                                       state="readonly", width=20)
        perspective_combo.pack(pady=2)
        perspective_combo.bind('<<ComboboxSelected>>', self.update_image)
        
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
        ttk.Checkbutton(effects_frame, text="Auto Shadow Direction", 
                       variable=self.auto_shadow_direction, command=self.update_image).pack(anchor=tk.W)
        
        # 3D Effect controls
        effect_frame = ttk.LabelFrame(control_frame, text="3D Parameters")
        effect_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(effect_frame, text="Z-Depth:").pack()
        ttk.Scale(effect_frame, from_=0.5, to=20.0, variable=self.depth_z, 
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
        ttk.Scale(effect_frame, from_=0.0, to=0.5, variable=self.perspective_skew, 
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
    
    def create_3d_text_effect(self, image, text, position):
        if not text:
            return image
            
        # Convert OpenCV image to PIL
        img_cv = image.copy()
        img_pil = Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))
        
        x, y = position
        
        # Calculate depth-based scaling: scale = k / (Z + 1e-3)
        z_depth = self.depth_z.get()
        k_factor = self.scale_factor.get()
        calculated_scale = k_factor / (z_depth + 1e-3)
        
        # Auto-calculate shadow direction if enabled
        if self.auto_shadow_direction.get() and self.enable_shadow.get():
            auto_shadow_x, auto_shadow_y = self.calculate_auto_shadow_direction(
                x, y, img_pil.width, img_pil.height, z_depth
            )
            # Update shadow offset variables (but don't trigger UI update)
            actual_shadow_x = auto_shadow_x
            actual_shadow_y = auto_shadow_y
        else:
            actual_shadow_x = self.shadow_offset_x.get()
            actual_shadow_y = self.shadow_offset_y.get()
        
        # Calculate final font size based on depth
        final_font_size = int(self.font_size.get() * calculated_scale / 100.0)
        final_font_size = max(8, min(200, final_font_size))  # Clamp to reasonable range
        
        # Try to load a system font, fallback to default if not available
        try:
            # Try common system fonts
            font = ImageFont.truetype("arial.ttf", final_font_size)
        except:
            try:
                font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", final_font_size)
            except:
                try:
                    font = ImageFont.load_default()
                except:
                    # Use PIL's default font as last resort
                    font = ImageFont.load_default()
        
        # Get text dimensions using PIL
        bbox = font.getbbox(text)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        
        # Create transparent overlay for text effects
        overlay_size = (img_pil.width, img_pil.height)
        text_overlay = Image.new('RGBA', overlay_size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(text_overlay)
        
        # Text colors
        text_color = (self.text_color_r.get(), self.text_color_g.get(), self.text_color_b.get(), 255)
        shadow_color = (self.shadow_color_r.get(), self.shadow_color_g.get(), self.shadow_color_b.get(), int(255 * self.shadow_opacity.get()))
        depth_color = (self.depth_color_r.get(), self.depth_color_g.get(), self.depth_color_b.get(), 200)
        
        # Create perspective transform if enabled
        text_img = text_overlay.copy()
        if self.enable_perspective.get() and self.perspective_type.get() != "none":
            # Create a temporary image for perspective transformation
            temp_size = (text_width + 100, text_height + 100)  # Extra padding for transform
            temp_img = Image.new('RGBA', temp_size, (0, 0, 0, 0))
            temp_draw = ImageDraw.Draw(temp_img)
            temp_x, temp_y = 50, 50  # Center in temp image
            
            # Apply perspective skew based on type
            skew_factor = self.perspective_skew.get()
            
            if self.perspective_type.get() == "taper_top":
                # Make text narrower at the top (receding perspective)
                # Define perspective transformation points
                width, height = temp_size
                transform_points = [
                    (temp_x - int(text_width * skew_factor / 2), temp_y),  # Top-left
                    (temp_x + text_width + int(text_width * skew_factor / 2), temp_y),  # Top-right
                    (temp_x + text_width, temp_y + text_height),  # Bottom-right
                    (temp_x, temp_y + text_height)  # Bottom-left
                ]
            elif self.perspective_type.get() == "taper_bottom":
                # Make text narrower at the bottom
                transform_points = [
                    (temp_x, temp_y),  # Top-left
                    (temp_x + text_width, temp_y),  # Top-right
                    (temp_x + text_width + int(text_width * skew_factor / 2), temp_y + text_height),  # Bottom-right
                    (temp_x - int(text_width * skew_factor / 2), temp_y + text_height)  # Bottom-left
                ]
            
            # Draw text on temp image first
            if self.enable_shadow.get():
                shadow_x = temp_x + actual_shadow_x
                shadow_y = temp_y + actual_shadow_y
                temp_draw.text((shadow_x, shadow_y), text, font=font, fill=shadow_color)
            
            # Depth effect - draw multiple layers for 3D illusion
            if self.enable_depth.get():
                depth_layers = min(8, int(z_depth))  # Limit depth layers based on z-depth
                for i in range(depth_layers, 0, -1):
                    depth_x = temp_x - i * 2
                    depth_y = temp_y - i * 2
                    # Progressive darkening
                    depth_alpha = int(100 * (depth_layers - i + 1) / depth_layers)
                    layer_color = (depth_color[0], depth_color[1], depth_color[2], depth_alpha)
                    temp_draw.text((depth_x, depth_y), text, font=font, fill=layer_color)
            
            # Main text
            temp_draw.text((temp_x, temp_y), text, font=font, fill=text_color)
            
            # Apply perspective transform (simple skew approximation)
            if skew_factor > 0:
                # Use transform with perspective points
                try:
                    # Simple shear transformation for perspective effect
                    if self.perspective_type.get() == "taper_top":
                        # Shear matrix for top tapering
                        shear_matrix = (1, -skew_factor, 0, 0, 1, 0)
                    else:
                        # Shear matrix for bottom tapering  
                        shear_matrix = (1, skew_factor, 0, 0, 1, 0)
                    
                    text_img = temp_img.transform(temp_size, Image.AFFINE, shear_matrix, Image.BILINEAR)
                except:
                    text_img = temp_img  # Fallback if transform fails
            else:
                text_img = temp_img
            
            # Paste the transformed text onto the main overlay
            text_overlay.paste(text_img, (x - 50, y - 50), text_img)
            
        else:
            # No perspective - draw directly on overlay
            if self.enable_shadow.get():
                shadow_x = x + self.shadow_offset_x.get()
                shadow_y = y + self.shadow_offset_y.get()
                draw.text((shadow_x, shadow_y), text, font=font, fill=shadow_color)
            
            # Depth effect - draw multiple layers for 3D illusion
            if self.enable_depth.get():
                depth_layers = min(8, int(z_depth))  # Limit depth layers based on z-depth
                for i in range(depth_layers, 0, -1):
                    depth_x = x - i * 2
                    depth_y = y - i * 2
                    # Progressive darkening
                    depth_alpha = int(100 * (depth_layers - i + 1) / depth_layers)
                    layer_color = (depth_color[0], depth_color[1], depth_color[2], depth_alpha)
                    draw.text((depth_x, depth_y), text, font=font, fill=layer_color)
            
            # Main text with effects
            if self.enable_outline.get():
                # Draw outline by drawing text in multiple positions
                outline_color = (0, 0, 0, 255)
                for dx in [-1, 0, 1]:
                    for dy in [-1, 0, 1]:
                        if dx != 0 or dy != 0:
                            draw.text((x + dx, y + dy), text, font=font, fill=outline_color)
            
            # Gradient effect
            if self.enable_gradient.get():
                # Simple gradient by varying alpha
                gradient_steps = 5
                for i in range(gradient_steps):
                    alpha_ratio = (i + 1) / gradient_steps
                    gradient_alpha = int(255 * alpha_ratio)
                    gradient_color = (text_color[0], text_color[1], text_color[2], gradient_alpha)
                    gradient_y = y - int(text_height * (1 - alpha_ratio) * 0.3)
                    draw.text((x, gradient_y), text, font=font, fill=gradient_color)
            else:
                draw.text((x, y), text, font=font, fill=text_color)
        
        # Apply blur to shadow if needed
        if self.enable_shadow.get() and self.shadow_blur.get() > 0:
            text_overlay = text_overlay.filter(ImageFilter.GaussianBlur(radius=self.shadow_blur.get()))
        
        # Convert back to OpenCV format and blend
        overlay_cv = cv2.cvtColor(np.array(text_overlay), cv2.COLOR_RGBA2BGRA)
        
        # Extract alpha channel and create mask
        alpha = overlay_cv[:, :, 3] / 255.0
        alpha_3ch = np.dstack([alpha, alpha, alpha])
        
        # Blend overlay with original image
        overlay_bgr = overlay_cv[:, :, :3]
        result = img_cv * (1 - alpha_3ch) + overlay_bgr * alpha_3ch
        
        return result.astype(np.uint8)
    
    def update_image(self, *args):
        if self.original_image is None:
            return
            
        # Create 3D text effect
        result_image = self.create_3d_text_effect(
            self.original_image,
            self.text.get(),
            (self.text_x.get(), self.text_y.get())
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
    
    def save_settings(self):
        """Save all current parameters to a JSON file"""
        settings = {
            # Text and position
            "text": self.text.get(),
            "text_x": self.text_x.get(),
            "text_y": self.text_y.get(),
            
            # Font and 3D parameters
            "font_size": self.font_size.get(),
            "depth_z": self.depth_z.get(),
            "scale_factor": self.scale_factor.get(),
            "perspective_type": self.perspective_type.get(),
            "perspective_skew": self.perspective_skew.get(),
            
            # Shadow parameters
            "shadow_offset_x": self.shadow_offset_x.get(),
            "shadow_offset_y": self.shadow_offset_y.get(),
            "shadow_blur": self.shadow_blur.get(),
            "shadow_opacity": self.shadow_opacity.get(),
            
            # Colors
            "text_color_r": self.text_color_r.get(),
            "text_color_g": self.text_color_g.get(),
            "text_color_b": self.text_color_b.get(),
            "shadow_color_r": self.shadow_color_r.get(),
            "shadow_color_g": self.shadow_color_g.get(),
            "shadow_color_b": self.shadow_color_b.get(),
            "depth_color_r": self.depth_color_r.get(),
            "depth_color_g": self.depth_color_g.get(),
            "depth_color_b": self.depth_color_b.get(),
            
            # Effect toggles
            "enable_shadow": self.enable_shadow.get(),
            "enable_depth": self.enable_depth.get(),
            "enable_outline": self.enable_outline.get(),
            "enable_gradient": self.enable_gradient.get(),
            "enable_bevel": self.enable_bevel.get(),
            "enable_perspective": self.enable_perspective.get(),
            "enable_emboss": self.enable_emboss.get(),
            "auto_shadow_direction": self.auto_shadow_direction.get()
        }
        
        file_path = filedialog.asksaveasfilename(
            title="Save Settings",
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
            initialdir="/home/ubuntu/githubrepos/aiglasses"
        )
        
        if file_path:
            try:
                with open(file_path, 'w') as f:
                    json.dump(settings, f, indent=4)
                messagebox.showinfo("Success", f"Settings saved to {file_path}")
            except Exception as e:
                messagebox.showerror("Error", f"Could not save settings: {str(e)}")
    
    def load_settings(self):
        """Load parameters from a JSON file"""
        file_path = filedialog.askopenfilename(
            title="Load Settings",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
            initialdir="/home/ubuntu/githubrepos/aiglasses"
        )
        
        if file_path:
            try:
                with open(file_path, 'r') as f:
                    settings = json.load(f)
                
                # Apply settings to variables
                if "text" in settings:
                    self.text.set(settings["text"])
                if "text_x" in settings:
                    self.text_x.set(settings["text_x"])
                if "text_y" in settings:
                    self.text_y.set(settings["text_y"])
                    
                if "font_size" in settings:
                    self.font_size.set(settings["font_size"])
                if "depth_z" in settings:
                    self.depth_z.set(settings["depth_z"])
                if "scale_factor" in settings:
                    self.scale_factor.set(settings["scale_factor"])
                if "perspective_type" in settings:
                    self.perspective_type.set(settings["perspective_type"])
                if "perspective_skew" in settings:
                    self.perspective_skew.set(settings["perspective_skew"])
                    
                if "shadow_offset_x" in settings:
                    self.shadow_offset_x.set(settings["shadow_offset_x"])
                if "shadow_offset_y" in settings:
                    self.shadow_offset_y.set(settings["shadow_offset_y"])
                if "shadow_blur" in settings:
                    self.shadow_blur.set(settings["shadow_blur"])
                if "shadow_opacity" in settings:
                    self.shadow_opacity.set(settings["shadow_opacity"])
                    
                # Colors
                if "text_color_r" in settings:
                    self.text_color_r.set(settings["text_color_r"])
                if "text_color_g" in settings:
                    self.text_color_g.set(settings["text_color_g"])
                if "text_color_b" in settings:
                    self.text_color_b.set(settings["text_color_b"])
                if "shadow_color_r" in settings:
                    self.shadow_color_r.set(settings["shadow_color_r"])
                if "shadow_color_g" in settings:
                    self.shadow_color_g.set(settings["shadow_color_g"])
                if "shadow_color_b" in settings:
                    self.shadow_color_b.set(settings["shadow_color_b"])
                if "depth_color_r" in settings:
                    self.depth_color_r.set(settings["depth_color_r"])
                if "depth_color_g" in settings:
                    self.depth_color_g.set(settings["depth_color_g"])
                if "depth_color_b" in settings:
                    self.depth_color_b.set(settings["depth_color_b"])
                    
                # Effect toggles
                if "enable_shadow" in settings:
                    self.enable_shadow.set(settings["enable_shadow"])
                if "enable_depth" in settings:
                    self.enable_depth.set(settings["enable_depth"])
                if "enable_outline" in settings:
                    self.enable_outline.set(settings["enable_outline"])
                if "enable_gradient" in settings:
                    self.enable_gradient.set(settings["enable_gradient"])
                if "enable_bevel" in settings:
                    self.enable_bevel.set(settings["enable_bevel"])
                if "enable_perspective" in settings:
                    self.enable_perspective.set(settings["enable_perspective"])
                if "enable_emboss" in settings:
                    self.enable_emboss.set(settings["enable_emboss"])
                if "auto_shadow_direction" in settings:
                    self.auto_shadow_direction.set(settings["auto_shadow_direction"])
                
                # Update the image with new settings
                self.update_image()
                messagebox.showinfo("Success", f"Settings loaded from {file_path}")
                
            except Exception as e:
                messagebox.showerror("Error", f"Could not load settings: {str(e)}")
    
    def calculate_auto_shadow_direction(self, text_x, text_y, image_width, image_height, z_depth):
        """Calculate shadow direction based on text position relative to image center"""
        # Calculate center of image
        center_x = image_width / 2
        center_y = image_height / 2
        
        # Calculate relative position from center (-1 to 1 range)
        rel_x = (text_x - center_x) / center_x
        rel_y = (text_y - center_y) / center_y
        
        # Shadow should be opposite to the light source
        # If text is to the right of center, shadow should go left (negative x)
        # If text is below center, shadow should go up (negative y)
        # Z-depth affects shadow intensity and distance
        
        # Base shadow distance affected by depth
        base_distance = max(3, min(15, z_depth * 2))
        
        # Calculate shadow offsets (opposite to relative position)
        shadow_x = -rel_x * base_distance
        shadow_y = -rel_y * base_distance
        
        # Add some bias to make shadows more natural (light from top-left)
        shadow_x += base_distance * 0.3  # Slight right bias
        shadow_y += base_distance * 0.2  # Slight down bias
        
        return int(shadow_x), int(shadow_y)

    # ...existing code...