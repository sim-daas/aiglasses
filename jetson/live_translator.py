"""
Live OCR Translation using EasyOCR and Google Translate
Integrates with AURA AI Glasses for real-time text translation
"""
import cv2
import numpy as np
import logging
import json
import os
from typing import List, Dict, Optional

try:
    import easyocr
    EASYOCR_AVAILABLE = True
except ImportError:
    EASYOCR_AVAILABLE = False

try:
    from langdetect import detect, LangDetectException
    from deep_translator import GoogleTranslator
    TRANSLATE_AVAILABLE = True
except ImportError:
    TRANSLATE_AVAILABLE = False

logger = logging.getLogger(__name__)

class LiveTranslator:
    """Real-time OCR and translation"""
    
    # Predefined language sets for EasyOCR compatibility
    LANGUAGE_PRESETS = {
        'western': ['en', 'es', 'fr', 'de', 'it', 'pt'],
        'eastern': ['ch_sim', 'en'],  # Chinese requires English
        'japanese': ['ja', 'en'],     # Japanese requires English
        'korean': ['ko', 'en'],       # Korean requires English
        'arabic': ['ar', 'en'],       # Arabic requires English
        'cyrillic': ['en', 'ru'],     # Russian with English
        'mixed': ['en', 'es', 'fr', 'de', 'it', 'pt', 'ru']  # Safe multi-language
    }
    
    def __init__(self, target_lang='en', language_preset='western', memory_file='aura_memory.json'):
        """
        Initialize translator
        
        Args:
            target_lang: Target language code (default: 'en')
            language_preset: Preset name from LANGUAGE_PRESETS (default: 'western')
            memory_file: File to persist user preferences
        """
        if not EASYOCR_AVAILABLE:
            raise ImportError("EasyOCR not available. Install with: pip install easyocr")
        
        if not TRANSLATE_AVAILABLE:
            raise ImportError("Translation libs not available. Install: pip install langdetect deep-translator")
        
        # Get language set from preset
        ocr_langs = self.LANGUAGE_PRESETS.get(language_preset, self.LANGUAGE_PRESETS['western'])
        
        self.target_lang = target_lang
        self.ocr_min_conf = 0.55
        self.show_original = False
        self.memory_file = memory_file
        
        # Load memory/preferences
        self.memory = self._load_memory()
        
        # Initialize EasyOCR reader with compatible language set
        logger.info(f"Initializing EasyOCR with preset '{language_preset}': {ocr_langs}")
        
        try:
            self.reader = easyocr.Reader(ocr_langs, gpu=True, verbose=False)
            logger.info(f"✅ EasyOCR initialized successfully")
        except Exception as e:
            logger.error(f"❌ EasyOCR initialization failed with preset '{language_preset}': {e}")
            logger.info("Falling back to English-only OCR...")
            self.reader = easyocr.Reader(['en'], gpu=True, verbose=False)
        
        logger.info(f"✅ Live translator initialized (target: {target_lang})")
    
    def _load_memory(self) -> dict:
        """Load user preferences from file"""
        if os.path.exists(self.memory_file):
            try:
                with open(self.memory_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Could not load memory: {e}")
        
        return {
            "target_lang": self.target_lang,
            "show_original": self.show_original,
            "ocr_interval": 0.6
        }
    
    def _save_memory(self):
        """Save user preferences to file"""
        try:
            with open(self.memory_file, 'w') as f:
                json.dump(self.memory, f, indent=2)
        except Exception as e:
            logger.warning(f"Could not save memory: {e}")
    
    def set_target_language(self, lang_code: str):
        """Change target translation language"""
        self.target_lang = lang_code
        self.memory['target_lang'] = lang_code
        self._save_memory()
        logger.info(f"Target language set to: {lang_code}")
    
    def detect_and_translate(self, frame: np.ndarray, depth_map: Optional[np.ndarray] = None) -> List[Dict]:
        """
        Detect text in frame and translate
        
        Args:
            frame: Input image (BGR)
            depth_map: Optional depth map for 3D positioning
            
        Returns:
            List of detection dictionaries with translated text and positions
        """
        results = []
        
        try:
            # Run OCR
            detections = self.reader.readtext(frame)
            
            for bbox, raw_text, conf in detections:
                if conf < self.ocr_min_conf:
                    continue
                
                text = raw_text.strip()
                if not text or len(text) < 2:
                    continue
                
                # Detect language
                lang = 'unknown'
                try:
                    lang = detect(text)
                except LangDetectException:
                    pass
                
                # Translate if not target language
                translated_text = text
                if lang != self.target_lang and lang != 'unknown':
                    try:
                        translator = GoogleTranslator(source='auto', target=self.target_lang)
                        translated_text = translator.translate(text)
                    except Exception as e:
                        logger.warning(f"Translation failed: {e}")
                        translated_text = text
                
                # Show original if enabled
                if self.memory.get('show_original', False) and translated_text != text:
                    translated_text = f"{translated_text}\n({text})"
                
                # Calculate bounding box center
                xs = [p[0] for p in bbox]
                ys = [p[1] for p in bbox]
                center_x = float(np.mean(xs))
                center_y = float(np.mean(ys))
                
                # Get depth if available
                depth = None
                if depth_map is not None:
                    h, w = depth_map.shape[:2]
                    px = int(np.clip(center_x, 0, w - 1))
                    py = int(np.clip(center_y, 0, h - 1))
                    depth_val = depth_map[py, px]
                    if np.isfinite(depth_val):
                        depth = float(depth_val)
                
                result = {
                    'text': text,
                    'translated': translated_text,
                    'language': lang,
                    'confidence': float(conf),
                    'bbox': [[float(p[0]), float(p[1])] for p in bbox],
                    'center': [center_x, center_y],
                    'depth': depth
                }
                
                results.append(result)
                
                logger.info(f"OCR: '{text}' ({lang}) -> '{translated_text}' (conf: {conf:.2f})")
            
        except Exception as e:
            logger.error(f"OCR/Translation error: {e}")
            import traceback
            logger.error(traceback.format_exc())
        
        return results
    
    def draw_overlay(self, frame: np.ndarray, detections: List[Dict]) -> np.ndarray:
        """
        Draw translation overlay on frame
        
        Args:
            frame: Input frame
            detections: List of detection results
            
        Returns:
            Frame with overlay
        """
        overlay = frame.copy()
        
        for det in detections:
            bbox = det['bbox']
            translated = det['translated']
            conf = det['confidence']
            
            # Draw bounding box
            pts = np.array(bbox, dtype=np.int32)
            cv2.polylines(overlay, [pts], True, (0, 255, 255), 2)
            
            # Draw translated text
            center_x, center_y = det['center']
            text_pos = (int(center_x), int(center_y) - 10)
            
            # Background for text
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            thickness = 2
            
            # Multi-line text
            lines = translated.split('\n')
            y_offset = 0
            for line in lines:
                text_size = cv2.getTextSize(line, font, font_scale, thickness)[0]
                bg_x1 = text_pos[0] - 5
                bg_y1 = text_pos[1] + y_offset - text_size[1] - 5
                bg_x2 = text_pos[0] + text_size[0] + 5
                bg_y2 = text_pos[1] + y_offset + 5
                
                cv2.rectangle(overlay, (bg_x1, bg_y1), (bg_x2, bg_y2), (0, 0, 0), -1)
                cv2.putText(overlay, line, (text_pos[0], text_pos[1] + y_offset),
                           font, font_scale, (0, 255, 0), thickness, cv2.LINE_AA)
                
                y_offset += text_size[1] + 10
        
        return overlay
