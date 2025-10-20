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
            "font_size": 32,  # Reduced from 48 for better fit
            "scale_factor": 80.0,  # Reduced from 100.0
            "shadow_offset_x": 5,
            "shadow_offset_y": 5,
            "shadow_blur": 3,
            "shadow_opacity": 0.6,
            "text_color_r": 0,
            "text_color_g": 255,
            "text_color_b": 0,
            "shadow_color_r": 0,
            "shadow_color_g": 0,
            "shadow_color_b": 0,
            "depth_color_r": 100,
            "depth_color_g": 100,
            "depth_color_b": 100,
            "enable_shadow": True,
            "enable_depth": True,
            "enable_outline": True,
            "auto_shadow_direction": True,
            "max_text_width_ratio": 0.6,  # Text takes max 60% of frame width
            "max_chars_per_line": 40  # Maximum characters per line
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
    
    def _wrap_text(self, text, font, max_width):
        """
        Wrap text to fit within max_width
        
        Args:
            text: String to wrap
            font: PIL Font object
            max_width: Maximum width in pixels
            
        Returns:
            List of text lines
        """
        words = text.split(' ')
        lines = []
        current_line = []
        
        for word in words:
            test_line = ' '.join(current_line + [word])
            bbox = font.getbbox(test_line)
            width = bbox[2] - bbox[0]
            
            if width <= max_width:
                current_line.append(word)
            else:
                if current_line:
                    lines.append(' '.join(current_line))
                    current_line = [word]
                else:
                    # Single word too long, force it
                    lines.append(word)
        
        if current_line:
            lines.append(' '.join(current_line))
        
        return lines if lines else [text]
    
    def render_3d_text(self, cv_image, text, position, z_depth=5.0):
        """
        Render 3D text with depth layers on OpenCV image
        
        Args:
            cv_image: OpenCV BGR image
            text: String to render
            position: (x, y) tuple in pixels
            z_depth: Depth value for scaling (higher = closer/larger)
        
        Returns:
            OpenCV image with 3D text overlay
        """
        if not text:
            return cv_image
        
        # Convert to PIL
        img_pil = Image.fromarray(cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB))
        x, y = position
        
        img_width = img_pil.width
        img_height = img_pil.height
        
        # Calculate resolution-aware font size
        # Base size scales with image width (640px = 24pt, 1920px = 72pt)
        base_font_size = int((img_width / 640) * 24)
        base_font_size = self.params.get('font_size', base_font_size)
        
        # Apply depth-based scaling
        k_factor = self.params.get('scale_factor', 80.0)
        scale_multiplier = z_depth / 10.0
        calculated_scale = k_factor * scale_multiplier
        
        # Final font size with resolution-aware bounds
        final_font_size = int(base_font_size * calculated_scale / 100.0)
        min_size = max(12, int(img_width / 80))  # Minimum readable
        max_size = max(60, int(img_width / 15))  # Maximum before too large
        final_font_size = max(min_size, min(max_size, final_font_size))
        
        # Load font
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", final_font_size)
        except:
            font = ImageFont.load_default()
        
        # Calculate max text width based on frame size
        max_text_width_ratio = self.params.get('max_text_width_ratio', 0.6)
        max_text_width = int(img_width * max_text_width_ratio)
        
        # Wrap text to multiple lines
        lines = self._wrap_text(text, font, max_text_width)
        
        # Calculate total text block dimensions
        line_height = final_font_size + int(final_font_size * 0.3)
        total_height = len(lines) * line_height
        
        # Get max line width
        max_line_width = 0
        for line in lines:
            bbox = font.getbbox(line)
            line_width = bbox[2] - bbox[0]
            max_line_width = max(max_line_width, line_width)
        
        # Adjust position to center the text block
        start_x = x - max_line_width // 2
        start_y = y - total_height // 2
        
        # Ensure text block stays within bounds
        margin = 20
        start_x = max(margin, min(img_width - max_line_width - margin, start_x))
        start_y = max(margin, min(img_height - total_height - margin, start_y))
        
        # Create overlay
        overlay = Image.new('RGBA', (img_pil.width, img_pil.height), (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        
        # Colors
        text_color = (
            self.params.get('text_color_r', 0),
            self.params.get('text_color_g', 255),
            self.params.get('text_color_b', 0),
            255
        )
        
        shadow_color = (
            self.params.get('shadow_color_r', 0),
            self.params.get('shadow_color_g', 0),
            self.params.get('shadow_color_b', 0),
            int(255 * self.params.get('shadow_opacity', 0.6))
        )
        
        depth_color = (
            self.params.get('depth_color_r', 100),
            self.params.get('depth_color_g', 100),
            self.params.get('depth_color_b', 100),
            200
        )
        
        # Calculate shadow offset
        if self.params.get('auto_shadow_direction', True):
            shadow_x, shadow_y = self.calculate_auto_shadow_direction(
                x, y, img_pil.width, img_pil.height, z_depth
            )
        else:
            shadow_x = self.params.get('shadow_offset_x', 5)
            shadow_y = self.params.get('shadow_offset_y', 5)
        
        # Depth layers configuration
        depth_layers = max(3, min(10, int(z_depth * 0.6)))
        layer_step = max(1, int(2 * scale_multiplier))
        
        # Outline width
        outline_width = max(1, int(1.5 * scale_multiplier))
        
        # Render each line
        current_y = start_y
        
        for line in lines:
            # Get line width for centering
            bbox = font.getbbox(line)
            line_width = bbox[2] - bbox[0]
            line_x = start_x + (max_line_width - line_width) // 2
            
            # Draw shadow
            if self.params.get('enable_shadow', True):
                draw.text((line_x + shadow_x, current_y + shadow_y), line, font=font, fill=shadow_color)
            
            # Draw depth layers
            if self.params.get('enable_depth', True):
                for i in range(depth_layers, 0, -1):
                    depth_x = line_x - i * layer_step
                    depth_y = current_y - i * layer_step
                    depth_alpha = int(120 * (depth_layers - i + 1) / depth_layers)
                    layer_color = (depth_color[0], depth_color[1], depth_color[2], depth_alpha)
                    draw.text((depth_x, depth_y), line, font=font, fill=layer_color)
            
            # Draw outline
            if self.params.get('enable_outline', True):
                outline_color = (0, 0, 0, 255)
                for dx in range(-outline_width, outline_width + 1):
                    for dy in range(-outline_width, outline_width + 1):
                        if dx != 0 or dy != 0:
                            draw.text((line_x + dx, current_y + dy), line, font=font, fill=outline_color)
            
            # Draw main text
            draw.text((line_x, current_y), line, font=font, fill=text_color)
            
            current_y += line_height
        
        # Apply blur
        shadow_blur = self.params.get('shadow_blur', 0)
        if shadow_blur > 0:
            overlay = overlay.filter(ImageFilter.GaussianBlur(radius=shadow_blur))
        
        # Convert back and blend
        overlay_cv = cv2.cvtColor(np.array(overlay), cv2.COLOR_RGBA2BGRA)
        alpha = overlay_cv[:, :, 3] / 255.0
        alpha_3ch = np.dstack([alpha, alpha, alpha])
        overlay_bgr = overlay_cv[:, :, :3]
        result = cv_image * (1 - alpha_3ch) + overlay_bgr * alpha_3ch
        
        return result.astype(np.uint8)
