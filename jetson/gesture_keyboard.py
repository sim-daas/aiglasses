"""
Gesture-based Virtual QWERTY Keyboard using MediaPipe Hand Tracking
Full keyboard layout with point-and-pinch interaction
"""
import math
import time
import cv2
import numpy as np
import logging

try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False
    
logger = logging.getLogger(__name__)

class GestureKeyboard:
    """Full QWERTY keyboard controlled by hand gestures"""
    
    # Full QWERTY keyboard layout (4 rows)
    KEYBOARD_LAYOUT = [
        ['Q', 'W', 'E', 'R', 'T', 'Y', 'U', 'I', 'O', 'P'],
        ['A', 'S', 'D', 'F', 'G', 'H', 'J', 'K', 'L'],
        ['Z', 'X', 'C', 'V', 'B', 'N', 'M'],
        ['SPACE', 'BACK', 'SUBMIT']
    ]
    
    # Key dimensions (relative to frame size)
    KEY_WIDTH_RATIO = 0.065   # Each key is ~6.5% of frame width
    KEY_HEIGHT_RATIO = 0.08   # Each key is ~8% of frame height
    KEY_SPACING_RATIO = 0.01  # 1% spacing between keys
    
    # Gesture timing
    PINCH_COOLDOWN = 0.3      # Prevent rapid repeated selections
    BACKSPACE_HOLD_S = 0.5    # Hold time for continuous backspace
    KEY_DEBOUNCE_S = 1.8      # Minimum time between same key presses
    
    def __init__(self):
        """Initialize gesture keyboard"""
        if not MEDIAPIPE_AVAILABLE:
            raise ImportError("MediaPipe not available. Install with: pip install mediapipe")
        
        # MediaPipe setup
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            max_num_hands=1,
            model_complexity=0,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.5
        )
        self.draw_utils = mp.solutions.drawing_utils
        
        # State
        self.typed_text = ""
        self.last_key_time = 0
        self.backspace_hold_start = None
        self.last_backspace_time = 0
        
        # Per-key debounce tracking
        self.last_key_press_times = {}  # key -> timestamp
        self.current_pinch_key = None   # Track which key is currently being pinched
        
        # Key positions (computed on first frame)
        self.key_rects = {}
        self.keyboard_initialized = False
        
        logger.info("✅ Full QWERTY gesture keyboard initialized")
    
    def reset_text(self):
        """Clear typed text"""
        self.typed_text = ""
    
    def get_text(self):
        """Get current typed text"""
        return self.typed_text
    
    @staticmethod
    def _norm(x):
        """Vector norm with epsilon"""
        return max(1e-6, np.linalg.norm(x))
    
    @staticmethod
    def _distance(p1, p2):
        """2D distance between two points"""
        return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
    
    def _initialize_keyboard_layout(self, frame_width, frame_height):
        """Initialize keyboard layout based on frame size"""
        if self.keyboard_initialized:
            return
        
        # Calculate dimensions
        key_w = int(frame_width * self.KEY_WIDTH_RATIO)
        key_h = int(frame_height * self.KEY_HEIGHT_RATIO)
        spacing = int(frame_width * self.KEY_SPACING_RATIO)
        
        # Center the keyboard vertically in the middle 50% of frame
        start_y = int(frame_height * 0.35)  # Start at 35% from top
        
        # Build key rectangles for each row
        for row_idx, row in enumerate(self.KEYBOARD_LAYOUT):
            # Calculate row width for centering
            if row_idx < 3:  # Letter rows
                row_width = len(row) * key_w + (len(row) - 1) * spacing
            else:  # Special keys row
                # SPACE is 3x wider
                row_width = key_w * 5 + spacing * 2
            
            start_x = (frame_width - row_width) // 2
            current_x = start_x
            current_y = start_y + row_idx * (key_h + spacing)
            
            for key in row:
                if key == 'SPACE':
                    # Space key is wider
                    w = key_w * 3
                elif key in ['BACK', 'SUBMIT']:
                    w = key_w
                else:
                    w = key_w
                
                # Store key rectangle: (x, y, width, height)
                self.key_rects[key] = (current_x, current_y, w, key_h)
                current_x += w + spacing
        
        self.keyboard_initialized = True
        logger.info(f"Keyboard layout initialized: {len(self.key_rects)} keys")
    
    def _get_key_at_point(self, x, y):
        """Get key at given point (x, y)"""
        for key, (kx, ky, kw, kh) in self.key_rects.items():
            if kx <= x <= kx + kw and ky <= y <= ky + kh:
                return key
        return None
    
    def _draw_keyboard(self, frame, hover_key=None):
        """Draw the virtual keyboard on frame"""
        for key, (x, y, w, h) in self.key_rects.items():
            # Determine key color
            if key == hover_key:
                # Highlighted when hovering
                color = (100, 255, 100)
                thickness = 3
                text_color = (0, 255, 0)
            elif key == 'SUBMIT':
                # Same style as other keys
                color = (80, 80, 80)
                thickness = 2
                text_color = (200, 200, 200)
            elif key == 'BACK':
                color = (80, 80, 80)
                thickness = 2
                text_color = (200, 200, 200)
            else:
                color = (80, 80, 80)
                thickness = 2
                text_color = (200, 200, 200)
            
            # Draw key rectangle
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, thickness)
            
            # Draw key label
            label = key if key not in ['BACK', 'SUBMIT', 'SPACE'] else key[:3]
            
            # Calculate text size for centering
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6 if key in ['SPACE', 'BACK', 'SUBMIT'] else 0.8
            text_size = cv2.getTextSize(label, font, font_scale, 2)[0]
            
            text_x = x + (w - text_size[0]) // 2
            text_y = y + (h + text_size[1]) // 2
            
            cv2.putText(frame, label, (text_x, text_y), 
                       font, font_scale, text_color, 2)
    
    def _can_press_key(self, key):
        """Check if enough time has passed since last press of this key"""
        now = time.time()
        
        # Special handling for BACK - allow faster repeat
        if key == 'BACK':
            return True
        
        # Check if this key was pressed recently
        if key in self.last_key_press_times:
            time_since_last = now - self.last_key_press_times[key]
            return time_since_last >= self.KEY_DEBOUNCE_S
        
        return True
    
    def _commit_key(self, key):
        """Add key to typed text"""
        now = time.time()
        
        if key == 'SPACE':
            self.typed_text += " "
        elif key == 'BACK':
            if self.typed_text:
                self.typed_text = self.typed_text[:-1]
        elif key != 'SUBMIT':
            self.typed_text += key.lower()
        
        # Record the time this key was pressed
        self.last_key_press_times[key] = now
        self.last_key_time = now
        logger.info(f"Key pressed: {key} → Text: '{self.typed_text}'")
    
    def process_frame(self, frame):
        """
        Process frame and detect gestures
        
        Args:
            frame: OpenCV BGR image
            
        Returns:
            tuple: (annotated_frame, status_text, should_submit)
        """
        h, w = frame.shape[:2]
        
        # Initialize keyboard layout
        self._initialize_keyboard_layout(w, h)
        
        # Convert to RGB for MediaPipe
        if frame.ndim == 2 or frame.shape[2] == 1:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process hand detection
        results = self.hands.process(rgb)
        
        # Create overlay
        overlay = frame.copy()
        
        # Draw text input area at top
        text_area_h = int(h * 0.12)
        text_area_y = int(h * 0.05)
        cv2.rectangle(overlay, (20, text_area_y), (w - 20, text_area_y + text_area_h), 
                     (20, 20, 20), -1)
        cv2.rectangle(overlay, (20, text_area_y), (w - 20, text_area_y + text_area_h), 
                     (100, 100, 100), 2)
        
        # Display typed text
        display_text = self.typed_text if len(self.typed_text) < 50 else "..." + self.typed_text[-47:]
        text_y = text_area_y + int(text_area_h * 0.7)
        cv2.putText(overlay, display_text, (30, text_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Character count
        char_count = f"{len(self.typed_text)} chars"
        cv2.putText(overlay, char_count, (w - 150, text_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
        
        status = "Point at key and pinch to type"
        should_submit = False
        hover_key = None
        
        # Process hand landmarks
        if results.multi_hand_landmarks:
            hand = results.multi_hand_landmarks[0]
            lm = hand.landmark
            
            # Get index finger tip position (landmark 8)
            idx_tip_x = int(lm[8].x * w)
            idx_tip_y = int(lm[8].y * h)
            
            # Get thumb tip position (landmark 4)
            thumb_tip_x = int(lm[4].x * w)
            thumb_tip_y = int(lm[4].y * h)
            
            # Calculate pinch distance
            pinch_dist = self._distance((idx_tip_x, idx_tip_y), (thumb_tip_x, thumb_tip_y))
            
            # Normalize by hand scale
            wrist = lm[0]
            mid_mcp = lm[9]
            hand_scale = self._distance(
                (int(wrist.x * w), int(wrist.y * h)),
                (int(mid_mcp.x * w), int(mid_mcp.y * h))
            )
            
            normalized_pinch = pinch_dist / (hand_scale + 1e-6)
            is_pinching = normalized_pinch < 0.20
            
            # Check which key is being pointed at
            hover_key = self._get_key_at_point(idx_tip_x, idx_tip_y)
            
            # Draw pointer indicator
            cv2.circle(overlay, (idx_tip_x, idx_tip_y), 8, (0, 255, 255), -1)
            cv2.circle(overlay, (idx_tip_x, idx_tip_y), 12, (255, 255, 0), 2)
            
            # Draw pinch indicator
            if is_pinching:
                cv2.line(overlay, (idx_tip_x, idx_tip_y), (thumb_tip_x, thumb_tip_y),
                        (0, 255, 0), 3)
            
            # Handle pinch gesture
            now = time.time()
            
            if is_pinching and hover_key:
                # Check if this is a new pinch on a different key
                if self.current_pinch_key != hover_key:
                    # New key - reset current pinch tracking
                    self.current_pinch_key = hover_key
                
                # Check debounce for this specific key
                if self._can_press_key(hover_key):
                    if hover_key == 'SUBMIT':
                        # Instant submit - no hold required
                        should_submit = True
                        status = "Submitting query!"
                        # Record press time to prevent repeat
                        self.last_key_press_times[hover_key] = now
                    
                    elif hover_key == 'BACK':
                        # Continuous backspace on hold
                        if self.backspace_hold_start is None:
                            self.backspace_hold_start = now
                            self._commit_key('BACK')
                        elif now - self.backspace_hold_start >= self.BACKSPACE_HOLD_S:
                            if now - self.last_backspace_time > 0.15:  # Repeat rate
                                self._commit_key('BACK')
                                self.last_backspace_time = now
                        status = "Backspace"
                    
                    else:
                        # Regular key press - only if debounce allows
                        self._commit_key(hover_key)
                        status = f"Typed: {hover_key}"
                else:
                    # Key is still in debounce period
                    time_remaining = self.KEY_DEBOUNCE_S - (now - self.last_key_press_times[hover_key])
                    status = f"Wait {time_remaining:.1f}s to press '{hover_key}' again"
            else:
                # Not pinching - reset hold timers and current pinch key
                self.backspace_hold_start = None
                if not is_pinching:
                    self.current_pinch_key = None
            
            # Update status based on hover
            if hover_key and not is_pinching:
                # Show if key is available or in cooldown
                if self._can_press_key(hover_key):
                    status = f"Hovering: {hover_key}"
                else:
                    time_remaining = self.KEY_DEBOUNCE_S - (now - self.last_key_press_times[hover_key])
                    status = f"Hovering: {hover_key} (cooldown: {time_remaining:.1f}s)"
            
            # Draw hand landmarks
            self.draw_utils.draw_landmarks(
                overlay, hand, self.mp_hands.HAND_CONNECTIONS,
                self.draw_utils.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                self.draw_utils.DrawingSpec(color=(255, 255, 0), thickness=2)
            )
        else:
            # No hand detected - reset current pinch key
            self.current_pinch_key = None
        
        # Draw keyboard
        self._draw_keyboard(overlay, hover_key)
        
        # Draw status below keyboard
        status_y = int(h * 0.75)
        cv2.rectangle(overlay, (20, status_y - 30), (w - 20, status_y + 5), 
                     (30, 30, 30), -1)
        cv2.putText(overlay, status, (30, status_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 200, 100), 2)
        
        # Blend overlay
        frame = cv2.addWeighted(overlay, 0.9, frame, 0.1, 0)
        
        return frame, status, should_submit
    
    def cleanup(self):
        """Release resources"""
        self.hands.close()
        logger.info("✅ Gesture keyboard cleaned up")
