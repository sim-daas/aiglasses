"""
Gesture-based Virtual Keyboard using MediaPipe Hand Tracking
Allows text input through hand gestures for AI glasses interface
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
    """Virtual keyboard controlled by hand gestures"""
    
    # Pie sectors (angles in degrees, 0° = right, CCW)
    SECTORS = [
        ("ABC",   0),
        ("DEF",  45),
        ("GHI",  90),
        ("JKL", 135),
        ("MNO", 180),
        ("PQRS", 225),
        ("TUV", 270),
        ("WXYZ", 315),
    ]
    SECTOR_WIDTH_DEG = 45
    
    # Gesture timing
    SUBMIT_HOLD_S = 1.0      # 3-finger hold to submit
    BACKSPACE_HOLD_S = 0.6   # open palm hold to backspace
    SPACE_COOLDOWN = 0.25    # prevent rapid space
    
    def __init__(self):
        """Initialize gesture keyboard"""
        if not MEDIAPIPE_AVAILABLE:
            raise ImportError("MediaPipe not available. Install with: pip install mediapipe")
        
        # MediaPipe setup
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            max_num_hands=1,
            model_complexity=0,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.draw_utils = mp.solutions.drawing_utils
        
        # State
        self.typed_text = ""
        self.last_commit_time = 0
        self.three_hold_start = None
        self.backspace_hold_start = None
        self.pinch_down = False
        self.pinch_start_time = 0
        self.selected_group = None
        
        logger.info("✅ Gesture keyboard initialized")
    
    def reset_text(self):
        """Clear typed text"""
        self.typed_text = ""
    
    def get_text(self):
        """Get current typed text"""
        return self.typed_text
    
    @staticmethod
    def _v2(a, b):
        """Vector from point a to b"""
        return np.array([b.x - a.x, b.y - a.y], dtype=np.float32)
    
    @staticmethod
    def _norm(x):
        """Vector norm with epsilon"""
        return max(1e-6, np.linalg.norm(x))
    
    @staticmethod
    def _clamp(x, a, b):
        """Clamp value between a and b"""
        return a if x < a else b if x > b else x
    
    def _angle_deg(self, v):
        """Get angle in degrees from vector"""
        a = math.degrees(math.atan2(-v[1], v[0]))  # y-up screen space
        return (a + 360.0) % 360.0
    
    def _angle_diff(self, a, b):
        """Get absolute difference between two angles"""
        d = (a - b + 180) % 360 - 180
        return abs(d)
    
    def _sector_from_angle(self, theta):
        """Get sector name from angle"""
        best_diff, name = 1e9, None
        for name_i, center in self.SECTORS:
            d = self._angle_diff(theta, center)
            if d < best_diff:
                best_diff = d
                name = name_i
        return name if best_diff <= self.SECTOR_WIDTH_DEG / 2 else None
    
    def _letter_from_group(self, group, radius_ratio):
        """Choose letter from group based on pinch radius"""
        if len(group) == 3:
            if radius_ratio < 0.33: return group[0]
            if radius_ratio < 0.66: return group[1]
            return group[2]
        else:  # 4 letters
            if radius_ratio < 0.25: return group[0]
            if radius_ratio < 0.5:  return group[1]
            if radius_ratio < 0.75: return group[2]
            return group[3]
    
    def _fingers_state(self, lm):
        """Detect finger states"""
        idx_up = lm[8].y < lm[6].y
        mid_up = lm[12].y < lm[10].y
        rng_up = lm[16].y < lm[14].y
        pnk_up = lm[20].y < lm[18].y
        
        count = sum([idx_up, mid_up, rng_up])
        open_palm = idx_up and mid_up and rng_up and pnk_up
        two_pinch = idx_up and mid_up and (not pnk_up)
        three = idx_up and mid_up and rng_up
        
        return count, open_palm, two_pinch, three
    
    def _commit_char(self, ch):
        """Add character to typed text"""
        self.typed_text += ch
        self.last_commit_time = time.time()
        logger.info(f"Typed: {self.typed_text}")
    
    def _backspace(self):
        """Remove last character"""
        if self.typed_text:
            self.typed_text = self.typed_text[:-1]
            logger.info(f"Backspace: {self.typed_text}")
    
    def process_frame(self, frame):
        """
        Process frame and detect gestures
        
        Args:
            frame: OpenCV BGR image
            
        Returns:
            tuple: (annotated_frame, status_text, should_submit)
        """
        h, w = frame.shape[:2]
        
        # Convert to RGB for MediaPipe
        if frame.ndim == 2 or frame.shape[2] == 1:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process hand detection
        results = self.hands.process(rgb)
        
        # Create overlay
        overlay = frame.copy()
        
        # Draw UI
        cx, cy = w // 2, h // 2
        radius_ui = int(min(w, h) * 0.35)
        
        # Draw pie sectors
        cv2.circle(overlay, (cx, cy), radius_ui, (60, 60, 60), 2)
        for _, ang in self.SECTORS:
            a = math.radians(ang)
            x2 = int(cx + radius_ui * math.cos(a))
            y2 = int(cy - radius_ui * math.sin(a))
            cv2.line(overlay, (cx, cy), (x2, y2), (50, 50, 50), 1)
        
        # Draw text display
        cur_text = self.typed_text if len(self.typed_text) < 36 else "…" + self.typed_text[-35:]
        cv2.rectangle(overlay, (20, 20), (w - 20, 80), (10, 10, 10), -1)
        cv2.putText(overlay, cur_text, (30, 65), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.9, (200, 255, 200), 2)
        
        status = "Idle"
        should_submit = False
        
        # Process hand landmarks
        if results.multi_hand_landmarks:
            hand = results.multi_hand_landmarks[0]
            lm = hand.landmark
            
            # Get key points
            wrist = lm[0]
            idx_tip = lm[8]
            idx_mcp = lm[5]
            mid_mcp = lm[9]
            th_tip = lm[4]
            
            # Orientation vector
            dir_vec = self._v2(wrist, idx_mcp)
            theta = self._angle_deg(dir_vec)
            
            # Hand scale
            scale = self._norm(self._v2(wrist, mid_mcp))
            
            # Pinch distance
            pinch_dist = self._norm(self._v2(th_tip, idx_tip)) / scale
            
            # Finger states
            count, open_palm, two_pinch, three = self._fingers_state(lm)
            
            # Pinch detection
            is_pinch = pinch_dist < 0.35
            
            # Select sector on pinch start
            if is_pinch and not self.pinch_down:
                self.pinch_down = True
                self.pinch_start_time = time.time()
                self.selected_group = self._sector_from_angle(theta)
                logger.info(f"Pinch started: {self.selected_group}")
            
            elif not is_pinch and self.pinch_down:
                # Commit letter on release
                if self.selected_group:
                    r = self._norm(self._v2(wrist, idx_tip)) / (self._norm(self._v2(wrist, idx_mcp)) * 1.6)
                    r = self._clamp(r, 0.0, 1.0)
                    ch = self._letter_from_group(self.selected_group, r)
                    self._commit_char(ch.lower())
                    status = f"Typed: {ch}"
                
                self.pinch_down = False
                self.selected_group = None
            
            # Space / Backspace / Submit
            now = time.time()
            
            # Two-finger pinch for space
            if two_pinch:
                if now - self.last_commit_time > self.SPACE_COOLDOWN:
                    self._commit_char(" ")
                    status = "Space"
            
            # Open palm for backspace
            if open_palm:
                if self.backspace_hold_start is None:
                    self.backspace_hold_start = now
                elif now - self.backspace_hold_start >= self.BACKSPACE_HOLD_S:
                    self._backspace()
                    self.backspace_hold_start = None
                    status = "Backspace"
            else:
                self.backspace_hold_start = None
            
            # Three fingers for submit
            if three:
                if self.three_hold_start is None:
                    self.three_hold_start = now
                elif now - self.three_hold_start >= self.SUBMIT_HOLD_S:
                    should_submit = True
                    status = "Submit!"
                    self.three_hold_start = None
            else:
                self.three_hold_start = None
            
            # Draw hand landmarks
            self.draw_utils.draw_landmarks(overlay, hand, self.mp_hands.HAND_CONNECTIONS)
            
            # Show current sector
            if self.selected_group:
                cv2.putText(overlay, f"{self.selected_group}", (cx - 60, cy - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 220, 180), 2)
                # Show ring thresholds
                cv2.circle(overlay, (cx, cy), int(radius_ui * 0.30), (80, 90, 110), 1)
                cv2.circle(overlay, (cx, cy), int(radius_ui * 0.55), (80, 90, 110), 1)
                cv2.circle(overlay, (cx, cy), int(radius_ui * 0.80), (80, 90, 110), 1)
            
            # Status text
            cv2.putText(overlay, status, (20, h - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 220, 90), 2)
        
        else:
            # No hand detected
            self.pinch_down = False
            self.selected_group = None
        
        # Blend UI
        frame = cv2.addWeighted(overlay, 0.92, frame, 0.08, 0)
        
        return frame, status, should_submit
    
    def cleanup(self):
        """Release resources"""
        self.hands.close()
        logger.info("✅ Gesture keyboard cleaned up")
