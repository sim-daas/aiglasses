"""
OpenGL 3D Text Renderer
Renders actual 3D text geometry with depth using OpenGL + FreeType
Similar to the ArUco marker overlay approach
"""
import cv2
import numpy as np
from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *
import moderngl
from PIL import Image, ImageDraw, ImageFont
import logging

logger = logging.getLogger(__name__)

class OpenGLTextRenderer:
    """
    Render 3D text with actual depth using OpenGL
    """
    def __init__(self, frame_width=640, frame_height=480):
        """
        Initialize OpenGL context for offscreen rendering
        
        Args:
            frame_width: Width of the camera frame
            frame_height: Height of the camera frame
        """
        self.width = frame_width
        self.height = frame_height
        
        # Create ModernGL standalone context (offscreen rendering)
        logger.info("Creating OpenGL context...")
        self.ctx = moderngl.create_standalone_context()
        
        # Create framebuffer for rendering
        self.fbo = self.ctx.simple_framebuffer((self.width, self.height))
        self.fbo.use()
        
        # Camera intrinsic matrix (approximate values for stereo camera)
        # These should be calibrated for your specific camera
        self.camera_matrix = np.array([
            [500.0, 0.0, self.width / 2],
            [0.0, 500.0, self.height / 2],
            [0.0, 0.0, 1.0]
        ], dtype=np.float32)
        
        # Shader programs
        self._init_shaders()
        
        logger.info("✅ OpenGL text renderer initialized")
    
    def _init_shaders(self):
        """Initialize OpenGL shaders for 3D text rendering"""
        # Vertex shader with perspective projection
        vertex_shader = """
        #version 330
        uniform mat4 Mvp;
        in vec3 in_vert;
        in vec2 in_tex;
        out vec2 v_tex;
        
        void main() {
            gl_Position = Mvp * vec4(in_vert, 1.0);
            v_tex = in_tex;
        }
        """
        
        # Fragment shader with texture mapping
        fragment_shader = """
        #version 330
        in vec2 v_tex;
        out vec4 fragColor;
        uniform sampler2D Texture;
        
        void main() {
            fragColor = texture(Texture, v_tex);
        }
        """
        
        self.prog = self.ctx.program(
            vertex_shader=vertex_shader,
            fragment_shader=fragment_shader
        )
    
    def _create_text_texture(self, text, font_size=64):
        """
        Create a texture from text using PIL
        
        Args:
            text: String to render
            font_size: Font size in pixels
            
        Returns:
            ModernGL texture object
        """
        # Create PIL image with text
        # Make it larger for better quality
        tex_width = 1024
        tex_height = 512
        
        img = Image.new('RGBA', (tex_width, tex_height), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)
        
        # Load font
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", font_size)
        except:
            font = ImageFont.load_default()
        
        # Calculate text position (centered)
        bbox = draw.textbbox((0, 0), text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        x = (tex_width - text_width) // 2
        y = (tex_height - text_height) // 2
        
        # Draw depth layers for 3D effect
        layer_count = 8
        for i in range(layer_count, 0, -1):
            depth_alpha = int(80 * (layer_count - i + 1) / layer_count)
            layer_color = (100, 100, 100, depth_alpha)
            draw.text((x - i * 2, y - i * 2), text, font=font, fill=layer_color)
        
        # Draw main text with outline
        outline_color = (0, 0, 0, 255)
        for dx in [-2, 0, 2]:
            for dy in [-2, 0, 2]:
                if dx != 0 or dy != 0:
                    draw.text((x + dx, y + dy), text, font=font, fill=outline_color)
        
        # Main text
        text_color = (0, 255, 0, 255)
        draw.text((x, y), text, font=font, fill=text_color)
        
        # Convert to OpenGL texture
        img_data = img.transpose(Image.FLIP_TOP_BOTTOM).tobytes()
        texture = self.ctx.texture((tex_width, tex_height), 4, img_data)
        texture.build_mipmaps()
        
        return texture
    
    def _get_projection_matrix(self):
        """
        Get OpenGL projection matrix from camera intrinsics
        
        Returns:
            4x4 projection matrix
        """
        fx = self.camera_matrix[0, 0]
        fy = self.camera_matrix[1, 1]
        cx = self.camera_matrix[0, 2]
        cy = self.camera_matrix[1, 2]
        
        near = 0.1
        far = 100.0
        
        # OpenGL projection matrix from camera intrinsics
        proj = np.array([
            [2*fx/self.width, 0, (self.width - 2*cx)/self.width, 0],
            [0, 2*fy/self.height, (2*cy - self.height)/self.height, 0],
            [0, 0, -(far + near)/(far - near), -2*far*near/(far - near)],
            [0, 0, -1, 0]
        ], dtype=np.float32)
        
        return proj
    
    def _get_view_matrix(self, world_pos):
        """
        Get view matrix for text at world position
        
        Args:
            world_pos: (x, y, z) position in world coordinates (meters)
            
        Returns:
            4x4 view matrix
        """
        x, y, z = world_pos
        
        # Translation matrix
        view = np.array([
            [1, 0, 0, x],
            [0, 1, 0, y],
            [0, 0, 1, z],
            [0, 0, 0, 1]
        ], dtype=np.float32)
        
        return view
    
    def _create_text_quad(self, width=1.0, height=0.5, depth=0.1):
        """
        Create 3D quad geometry for text with actual depth
        
        Args:
            width: Width in meters
            height: Height in meters
            depth: Depth/thickness in meters
            
        Returns:
            Vertex buffer and index buffer
        """
        # Create vertices for a 3D box with texture coordinates
        # Front face, back face, and connecting edges
        w, h, d = width/2, height/2, depth/2
        
        vertices = np.array([
            # Front face (with texture coords)
            [-w, -h, d,  0, 0],  # bottom-left
            [ w, -h, d,  1, 0],  # bottom-right
            [ w,  h, d,  1, 1],  # top-right
            [-w,  h, d,  0, 1],  # top-left
            
            # Back face (darker, no texture)
            [-w, -h, -d,  0, 0],
            [ w, -h, -d,  1, 0],
            [ w,  h, -d,  1, 1],
            [-w,  h, -d,  0, 1],
        ], dtype='f4')
        
        # Indices for triangles
        indices = np.array([
            # Front face
            0, 1, 2,  2, 3, 0,
            # Back face
            4, 5, 6,  6, 7, 4,
            # Connecting edges (sides)
            0, 4, 7,  7, 3, 0,  # left
            1, 5, 6,  6, 2, 1,  # right
            3, 7, 6,  6, 2, 3,  # top
            0, 4, 5,  5, 1, 0,  # bottom
        ], dtype='i4')
        
        return vertices, indices
    
    def render_3d_text(self, frame, text, world_position, size=0.5):
        """
        Render 3D text at a world position and overlay on frame
        
        Args:
            frame: OpenCV BGR image
            text: String to render
            world_position: (x, y, z) in world coordinates (meters)
            size: Size scale factor
            
        Returns:
            Frame with 3D text overlaid
        """
        try:
            # Clear framebuffer
            self.fbo.use()
            self.fbo.clear(0.0, 0.0, 0.0, 0.0)
            
            # Create text texture
            texture = self._create_text_texture(text)
            texture.use()
            
            # Get projection and view matrices
            proj = self._get_projection_matrix()
            view = self._get_view_matrix(world_position)
            
            # Model matrix (scale by size)
            model = np.eye(4, dtype=np.float32)
            model[0, 0] = size
            model[1, 1] = size
            model[2, 2] = size
            
            # Combined MVP matrix
            mvp = proj @ view @ model
            
            # Set shader uniform
            self.prog['Mvp'].write(mvp.tobytes())
            
            # Create geometry
            vertices, indices = self._create_text_quad()
            
            # Create buffers
            vbo = self.ctx.buffer(vertices.tobytes())
            ibo = self.ctx.buffer(indices.tobytes())
            
            # Create VAO
            vao = self.ctx.simple_vertex_array(
                self.prog, 
                vbo, 
                'in_vert', 'in_tex',
                index_buffer=ibo
            )
            
            # Render
            vao.render()
            
            # Read pixels from framebuffer
            data = self.fbo.read(components=4, alignment=1)
            text_img = np.frombuffer(data, dtype=np.uint8).reshape(self.height, self.width, 4)
            text_img = np.flipud(text_img)  # Flip vertically
            
            # Convert frame to BGRA
            frame_bgra = cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA)
            
            # Alpha blend
            alpha = text_img[:, :, 3:4] / 255.0
            blended = (frame_bgra * (1 - alpha) + text_img * alpha).astype(np.uint8)
            
            # Convert back to BGR
            result = cv2.cvtColor(blended, cv2.COLOR_BGRA2BGR)
            
            # Cleanup
            vao.release()
            vbo.release()
            ibo.release()
            texture.release()
            
            return result
            
        except Exception as e:
            logger.error(f"Error rendering 3D text: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return frame
    
    def world_position_from_depth(self, pixel_x, pixel_y, depth_meters):
        """
        Convert 2D pixel + depth to 3D world coordinates
        
        Args:
            pixel_x: X coordinate in image
            pixel_y: Y coordinate in image
            depth_meters: Depth in meters (from stereo)
            
        Returns:
            (x, y, z) in camera coordinates (meters)
        """
        # Backproject using camera matrix
        fx = self.camera_matrix[0, 0]
        fy = self.camera_matrix[1, 1]
        cx = self.camera_matrix[0, 2]
        cy = self.camera_matrix[1, 2]
        
        x = (pixel_x - cx) * depth_meters / fx
        y = (pixel_y - cy) * depth_meters / fy
        z = depth_meters
        
        return (x, y, z)
