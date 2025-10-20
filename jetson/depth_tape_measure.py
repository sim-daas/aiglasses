"""
AR Tape Measure - 3D distance measurement using stereo depth
Integrates with AURA AI Glasses stereo camera system
"""
import cv2
import numpy as np
import logging

logger = logging.getLogger(__name__)

class DepthTapeMeasure:
    """AR tape measure with depth-based 3D measurements"""
    
    def __init__(self, baseline_mm=65, focal_px=900, frame_width=640, frame_height=480):
        """
        Initialize tape measure
        
        Args:
            baseline_mm: Distance between camera centers in mm
            focal_px: Approximate focal length in pixels
            frame_width: Frame width
            frame_height: Frame height
        """
        self.baseline_mm = baseline_mm
        self.focal_px = focal_px
        self.w = frame_width
        self.h = frame_height
        self.cx = frame_width // 2
        self.cy = frame_height // 2
        
        # Measurement state
        self.point1 = None
        self.point2 = None
        self.arrow_at = None
        self.depth_map = None
        
        # Depth smoothing
        self.depth_history = []
        self.depth_history_size = 10
        self.smoothed_center_depth = 1.0
        
        # Stereo matcher
        self.stereo = self._create_stereo_matcher()
        
        logger.info(f"✅ Depth tape measure initialized (baseline={baseline_mm}mm, focal={focal_px}px)")
    
    def _create_stereo_matcher(self, num_disp=128, block=5):
        """Create StereoSGBM matcher"""
        return cv2.StereoSGBM_create(
            minDisparity=0,
            numDisparities=num_disp,
            blockSize=block,
            P1=8 * 1 * block * block,
            P2=32 * 1 * block * block,
            disp12MaxDiff=1,
            uniquenessRatio=10,
            speckleWindowSize=50,
            speckleRange=1,
            preFilterCap=63,
            mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
        )
    
    def compute_depth(self, frame_left, frame_right):
        """
        Compute depth map from stereo frames
        
        Args:
            frame_left: Left camera frame (BGR)
            frame_right: Right camera frame (BGR)
            
        Returns:
            Depth map in meters
        """
        # Convert to grayscale
        gray_left = cv2.cvtColor(frame_left, cv2.COLOR_BGR2GRAY)
        gray_right = cv2.cvtColor(frame_right, cv2.COLOR_BGR2GRAY)
        
        # Compute disparity
        disp = self.stereo.compute(gray_left, gray_right).astype(np.float32) / 16.0
        
        # Convert to depth (meters): Z = f * B / disp
        depth = (self.focal_px * (self.baseline_mm / 1000.0)) / (disp + 1e-6)
        depth[disp <= 0] = np.nan
        
        self.depth_map = depth
        return depth
    
    def get_depth_at_point(self, x, y, radius=6):
        """Get median depth at point with radius"""
        if self.depth_map is None:
            return np.nan
        
        x = max(0, min(self.w - 1, x))
        y = max(0, min(self.h - 1, y))
        
        patch = self.depth_map[
            max(0, y - radius):min(self.h, y + radius),
            max(0, x - radius):min(self.w, x + radius)
        ]
        
        return float(np.nanmedian(patch)) if np.isfinite(patch).any() else np.nan
    
    def backproject(self, x, y, z):
        """Convert 2D point + depth to 3D coordinates"""
        X = (x - self.cx) * z / self.focal_px
        Y = (y - self.cy) * z / self.focal_px
        return np.array([X, Y, z], dtype=np.float32)
    
    def set_point1(self, x, y):
        """Set first measurement point"""
        self.point1 = (x, y)
        logger.info(f"Point 1 set at ({x}, {y})")
    
    def set_point2(self, x, y):
        """Set second measurement point"""
        self.point2 = (x, y)
        logger.info(f"Point 2 set at ({x}, {y})")
    
    def set_arrow(self, x, y):
        """Set arrow position"""
        self.arrow_at = (x, y)
        logger.info(f"Arrow placed at ({x}, {y})")
    
    def clear_measurements(self):
        """Clear all measurement points"""
        self.point1 = None
        self.point2 = None
        self.arrow_at = None
        logger.info("Measurements cleared")
    
    def draw_overlay(self, frame):
        """
        Draw AR tape measure overlay on frame
        
        Args:
            frame: Frame to draw on
            
        Returns:
            Frame with overlay
        """
        overlay = frame.copy()
        
        # Draw center crosshair with distance
        cv2.circle(overlay, (self.cx, self.cy), 8, (0, 255, 255), 2)
        cv2.line(overlay, (self.cx - 15, self.cy), (self.cx + 15, self.cy), (0, 255, 255), 2)
        cv2.line(overlay, (self.cx, self.cy - 15), (self.cx, self.cy + 15), (0, 255, 255), 2)
        
        # Center distance
        center_depth = self.get_depth_at_point(self.cx, self.cy, radius=12)
        if np.isfinite(center_depth):
            dist_text = f"{center_depth * 100:.0f} cm"
            cv2.putText(overlay, dist_text, (self.cx + 12, self.cy - 12),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (50, 255, 50), 2, cv2.LINE_AA)
        
        # Draw measurement points
        if self.point1 is not None:
            cv2.circle(overlay, self.point1, 6, (255, 200, 0), -1)
            cv2.circle(overlay, self.point1, 8, (255, 255, 255), 2)
        
        if self.point1 is not None and self.point2 is not None:
            cv2.circle(overlay, self.point2, 6, (255, 200, 0), -1)
            cv2.circle(overlay, self.point2, 8, (255, 255, 255), 2)
            
            # Calculate 3D distance
            z1 = self.get_depth_at_point(*self.point1)
            z2 = self.get_depth_at_point(*self.point2)
            
            if np.isfinite(z1) and np.isfinite(z2):
                P1 = self.backproject(self.point1[0], self.point1[1], z1)
                P2 = self.backproject(self.point2[0], self.point2[1], z2)
                distance_3d = np.linalg.norm(P1 - P2)
                
                # Draw line
                cv2.line(overlay, self.point1, self.point2, (0, 255, 0), 2)
                
                # Draw distance label
                mid_x = (self.point1[0] + self.point2[0]) // 2
                mid_y = (self.point1[1] + self.point2[1]) // 2
                dist_text = f"{distance_3d * 100:.1f} cm"
                
                # Background for text
                text_size = cv2.getTextSize(dist_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
                cv2.rectangle(overlay,
                            (mid_x - 5, mid_y - text_size[1] - 5),
                            (mid_x + text_size[0] + 5, mid_y + 5),
                            (0, 0, 0), -1)
                
                cv2.putText(overlay, dist_text, (mid_x, mid_y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
        
        # Draw depth-scaled arrow
        if self.arrow_at is not None:
            za = self.get_depth_at_point(*self.arrow_at)
            if np.isfinite(za):
                # Scale arrow by depth (closer = bigger)
                scale = int(max(20, min(140, 300 / za)))
                x, y = self.arrow_at
                
                # Arrow triangle
                pts = np.array([
                    [x, y - scale],
                    [x - int(scale * 0.5), y + int(scale * 0.7)],
                    [x + int(scale * 0.5), y + int(scale * 0.7)]
                ], dtype=np.int32)
                
                cv2.fillConvexPoly(overlay, pts, (0, 220, 0))
                cv2.polylines(overlay, [pts], True, (0, 0, 0), 2)
                cv2.circle(overlay, self.arrow_at, 4, (0, 0, 0), -1)
                
                # Depth label
                depth_text = f"{za * 100:.0f}cm"
                cv2.putText(overlay, depth_text, (x + 10, y + scale + 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 220, 0), 2, cv2.LINE_AA)
        
        return overlay
    
    def get_smoothed_center_depth(self):
        """Get smoothed center depth for stable scaling"""
        center_depth = self.get_depth_at_point(self.cx, self.cy, radius=12)
        
        if np.isfinite(center_depth) and center_depth > 0:
            self.depth_history.append(center_depth)
            if len(self.depth_history) > self.depth_history_size:
                self.depth_history.pop(0)
            
            # Use median for robustness
            if self.depth_history:
                sorted_depths = sorted(self.depth_history)
                self.smoothed_center_depth = sorted_depths[len(sorted_depths) // 2]
        
        return self.smoothed_center_depth
    
    def get_depth_grid(self, grid_width=32, grid_height=24):
        """
        Get downsampled depth grid for web overlay
        
        Args:
            grid_width: Grid width
            grid_height: Grid height
            
        Returns:
            dict with depth grid data
        """
        if self.depth_map is None:
            return None
        
        # Downsample depth map
        step_x = self.w // grid_width
        step_y = self.h // grid_height
        
        grid = []
        for y in range(grid_height):
            for x in range(grid_width):
                px = min(x * step_x, self.w - 1)
                py = min(y * step_y, self.h - 1)
                depth = self.get_depth_at_point(px, py, radius=3)
                grid.append(depth if np.isfinite(depth) else 6.0)
        
        return {
            'w': self.w,
            'h': self.h,
            'gw': grid_width,
            'gh': grid_height,
            'fx': self.focal_px,
            'fy': self.focal_px,
            'cx': self.cx,
            'cy': self.cy,
            'z': grid
        }
