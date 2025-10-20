import google.generativeai as genai
import json
import time
import logging
from PIL import Image
from config import Config

logger = logging.getLogger(__name__)

class GeminiClient:
    def __init__(self):
        if not Config.GEMINI_API_KEY:
            raise ValueError("GEMINI_API_KEY not found in .env file")
        
        logger.info("Configuring Gemini API...")
        genai.configure(api_key=Config.GEMINI_API_KEY)
        self.model = genai.GenerativeModel(Config.GEMINI_MODEL)
        logger.info(f"✅ Gemini initialized: {Config.GEMINI_MODEL}")
    
    def process_multimodal_query(self, image_path, audio_path=None, text_query=None):
        """
        Process image + audio/text query with Gemini
        """
        for attempt in range(Config.GEMINI_MAX_RETRIES):
            try:
                logger.info(f"🔄 Gemini API call (attempt {attempt + 1}/{Config.GEMINI_MAX_RETRIES})")
                
                # Load image
                logger.info(f"Loading image: {image_path}")
                image = Image.open(image_path)
                logger.info(f"Image loaded: {image.size}, mode: {image.mode}")
                
                # Construct prompt
                logger.info("Building prompt...")
                prompt = self._build_prompt(text_query, audio_path is not None)
                
                # Build content list
                content = [prompt, image]
                
                # Add audio if provided
                if audio_path:
                    logger.info(f"Loading audio: {audio_path}")
                    with open(audio_path, 'rb') as f:
                        audio_data = f.read()
                    logger.info(f"Audio loaded: {len(audio_data)} bytes")
                    content.append({
                        "mime_type": "audio/wav",
                        "data": audio_data
                    })
                
                # Make API call
                logger.info("Sending request to Gemini...")
                start_time = time.time()
                
                response = self.model.generate_content(content)
                
                elapsed = time.time() - start_time
                logger.info(f"✅ Gemini responded in {elapsed:.2f}s")
                
                # Parse response
                logger.info("Parsing JSON response...")
                result = self._parse_response(response.text)
                
                logger.info(f"✅ Parsed result: {json.dumps(result, indent=2)}")
                return result
                
            except Exception as e:
                logger.error(f"❌ Gemini error (attempt {attempt + 1}): {e}")
                if attempt < Config.GEMINI_MAX_RETRIES - 1:
                    logger.info(f"Retrying in {Config.GEMINI_RETRY_DELAY} seconds...")
                    time.sleep(Config.GEMINI_RETRY_DELAY)
                else:
                    logger.error("All retry attempts failed")
                    return self._fallback_response()
        
        return self._fallback_response()
    
    def _build_prompt(self, text_query=None, has_audio=False):
        """Build structured prompt for Gemini"""
        base_context = ""
        if has_audio:
            base_context = "You will receive an audio recording and an image."
        elif text_query:
            base_context = f'User query: "{text_query}"'
        
        return f"""
{base_context}

Analyze the image and answer the user's question CONCISELY. You MUST respond in valid JSON format.

Required JSON structure:
{{
    "transcription": "the user's spoken/written question (if audio provided, transcribe it; otherwise use provided text)",
    "answer": "your CONCISE answer (max 15 words) - ALWAYS provide your best guess, never say 'unknown' or 'N/A'",
    "object": "primary object/subject in the image (single word or short phrase, e.g., 'laptop', 'person', 'cup')",
    "location": "grid position in image - choose ONE: 'top-left', 'top-center', 'top-right', 'center-left', 'center', 'center-right', 'bottom-left', 'bottom-center', 'bottom-right'"
}}

**EXAMPLE OUTPUT:**
User asks: "What color is this laptop?"
{{
    "transcription": "What color is this laptop?",
    "answer": "Silver aluminum MacBook Pro",
    "object": "laptop",
    "location": "center"
}}

**LOCATION GRID:**
Divide the image into a 3x3 grid:
- Row 1 (top): top-left, top-center, top-right
- Row 2 (middle): center-left, center, center-right  
- Row 3 (bottom): bottom-left, bottom-center, bottom-right

Choose the cell where the PRIMARY object is located.

**RULES:**
1. Always output ONLY valid JSON (no markdown, no extra text)
2. Keep "answer" under 15 words
3. NEVER use "unknown" - make educated guesses based on visual context
4. "location" MUST be exactly one of the 9 grid positions listed above
5. Be confident and commit to an answer
"""
    
    def _parse_response(self, response_text):
        """Parse and validate JSON response"""
        try:
            # Clean response
            response_text = response_text.strip()
            
            # Remove markdown code blocks if present
            if response_text.startswith('```json'):
                response_text = response_text.replace('```json', '').replace('```', '').strip()
            elif response_text.startswith('```'):
                response_text = response_text.replace('```', '').strip()
            
            # Parse JSON
            result = json.loads(response_text)
            
            # Validate required keys
            required_keys = ['transcription', 'answer', 'object', 'location']
            for key in required_keys:
                if key not in result:
                    raise ValueError(f"Missing required key: {key}")
            
            # Add default position for compatibility (center of image)
            result['position'] = {
                'x': 0.5,  # Center
                'y': 0.5,  # Center
                'z': 0.5,  # Default depth
                'confidence': 0.8,
                'description': result['location']
            }
            
            return result
            
        except json.JSONDecodeError as e:
            logger.error(f"❌ JSON parse error: {e}")
            logger.error(f"Raw response: {response_text[:200]}")
            raise
        except Exception as e:
            logger.error(f"❌ Response validation error: {e}")
            raise
    
    def _fallback_response(self):
        """Fallback response when API fails"""
        return {
            "transcription": "Error processing request",
            "answer": "Unable to process - please try again",
            "object": "unknown",
            "location": "center",
            "position": {
                "x": 0.5,
                "y": 0.5,
                "z": 0.5,
                "confidence": 0.0,
                "description": "center"
            }
        }
