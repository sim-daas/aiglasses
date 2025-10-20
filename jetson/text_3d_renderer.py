"""
3D Text Renderer - Extracts 3D text rendering logic from 3dtext.py
Creates text with depth layers, perspective, shadows, and effects
"""
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter

class Text3DRenderer:
    def __init__(self, params=None):
        """Initialize with rendering parameters"""
        self.params = params if params else self.get_default_params()
    
    @staticmethod
    def get_default_params():
        """Default parameters matching detect.py and 3dtext.py"""
        return {
            "font_size": 48,
            "scale_factor": 100.0,
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
            "auto_shadow_direction": True
        }
    
    def calculate_auto_shadow_direction(self, text_x, text_y, image_width, image_height, z_depth):
        """Calculate shadow direction based on text position"""
        center_x = image_width / 2
        center_y = image_height / 2
        
        rel_x = (text_x - center_x) / center_x
        rel_y = (text_y - center_y) / center_y
        
        base_distance = max(3, min(15, z_depth * 2))
        
        shadow_x = rel_x * base_distance * 1.1
        shadow_y = rel_y * base_distance * 1.1
        
        shadow_x += base_distance * 0.35
        shadow_y += base_distance * 0.25
        
        return int(shadow_x), int(shadow_y)
    
    def render_3d_text(self, cv_image, text, position, z_depth=5.0):
        """
        Render 3D text with depth layers on OpenCV image
        
        Args:
            cv_image: OpenCV BGR image
            text: String to render
            position: (x, y) tuple
            z_depth: Depth value for scaling and effects
        
        Returns:
            OpenCV image with 3D text overlay
        """
        if not text:
            return cv_image
        
        # Convert to PIL
        img_pil = Image.fromarray(cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB))
        x, y = position
        
        # Calculate depth-based scaling
        k_factor = self.params.get('scale_factor', 100.0)
        calculated_scale = k_factor / (z_depth + 1e-3)
        
        # Final font size
        base_font_size = self.params.get('font_size', 48)
        final_font_size = int(base_font_size * calculated_scale / 100.0)
        final_font_size = max(12, min(100, final_font_size))
        
        # Load font
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", final_font_size)
        except:
            font = ImageFont.load_default()
        
        # Get text dimensions
        bbox = font.getbbox(text)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        
        # Create overlay
        overlay = Image.new('RGBA', (img_pil.width, img_pil.height), (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        
        # Colors
        text_color = (
            self.params.get('text_color_r', 255),
            self.params.get('text_color_g', 255),
            self.params.get('text_color_b', 255),
            255
        )
        
        shadow_color = (
            self.params.get('shadow_color_r', 0),
            self.params.get('shadow_color_g', 0),
            self.params.get('shadow_color_b', 0),
            int(255 * self.params.get('shadow_opacity', 0.6))
        )
        
        depth_color = (
            self.params.get('depth_color_r', 128),
            self.params.get('depth_color_g', 128),
            self.params.get('depth_color_b', 128),
            200
        )
        
        # Auto shadow direction
        if self.params.get('auto_shadow_direction', True):
            shadow_x, shadow_y = self.calculate_auto_shadow_direction(
                x, y, img_pil.width, img_pil.height, z_depth
            )
        else:
            shadow_x = self.params.get('shadow_offset_x', 5)
            shadow_y = self.params.get('shadow_offset_y', 5)
        
        # Draw shadow
        if self.params.get('enable_shadow', True):
            draw.text((x + shadow_x, y + shadow_y), text, font=font, fill=shadow_color)
        
        # Draw depth layers (THIS IS THE KEY 3D EFFECT)
        if self.params.get('enable_depth', True):
            depth_layers = min(8, int(z_depth))
            for i in range(depth_layers, 0, -1):
                depth_x = x - i * 2
                depth_y = y - i * 2
                depth_alpha = int(120 * (depth_layers - i + 1) / depth_layers)
                layer_color = (depth_color[0], depth_color[1], depth_color[2], depth_alpha)
                draw.text((depth_x, depth_y), text, font=font, fill=layer_color)
        
        # Draw outline
        if self.params.get('enable_outline', True):
            outline_color = (0, 0, 0, 255)
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    if dx != 0 or dy != 0:
                        draw.text((x + dx, y + dy), text, font=font, fill=outline_color)
        
        # Draw main text
        draw.text((x, y), text, font=font, fill=text_color)
        
        # Apply blur to shadow
        shadow_blur = self.params.get('shadow_blur', 0)
        if self.params.get('enable_shadow', True) and shadow_blur > 0:
            overlay = overlay.filter(ImageFilter.GaussianBlur(radius=shadow_blur))
        
        # Convert back and blend
        overlay_cv = cv2.cvtColor(np.array(overlay), cv2.COLOR_RGBA2BGRA)
        alpha = overlay_cv[:, :, 3] / 255.0
        alpha_3ch = np.dstack([alpha, alpha, alpha])
        overlay_bgr = overlay_cv[:, :, :3]
        result = cv_image * (1 - alpha_3ch) + overlay_bgr * alpha_3ch
        
        return result.astype(np.uint8)
