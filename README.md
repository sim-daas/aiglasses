# AI Vision API with Gemini

A modular Python library for real-time image analysis using Google's Gemini AI and webcam input.

## Features

- **Modular Design**: Reusable `VisionAPI` class for integration into other projects
- **Real-time Webcam Capture**: Uses OpenCV for camera access
- **Gemini AI Integration**: Latest Google Generative AI API
- **Dual Interface**: Both GUI (Tkinter) and CLI modes
- **Structured Output**: JSON responses with `answer` and `label` fields
- **Environment Variables**: Secure API key management with `.env` files

## Installation

### Quick Setup (Recommended)
```bash
chmod +x setup.sh
./setup.sh
```

### Manual Installation

1. **Create virtual environment (recommended):**
```bash
python3 -m venv venv
source venv/bin/activate
```

2. **Install packages:**
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

3. **Get Gemini API Key:**
   - Visit [Google AI Studio](https://makersuite.google.com/app/apikey)
   - Create a new API key
   - Copy the key

4. **Setup Environment Variables:**
```bash
cp .env.example .env
# Edit .env file and add your API key:
# GEMINI_API_KEY=your_actual_api_key_here
```

### Troubleshooting NumPy/OpenCV Compatibility
If you encounter NumPy version conflicts:
```bash
pip uninstall numpy opencv-python
pip install --upgrade pip
pip install -r requirements.txt
```

## Required Packages

- `numpy>=1.26.0,<2.0.0` - Numerical computing (compatible with OpenCV)
- `opencv-python>=4.8.0,<5.0.0` - Computer vision and webcam access
- `google-generativeai>=0.8.0` - Google Gemini AI API (latest)
- `python-dotenv>=1.0.0` - Environment variable management
- `Pillow>=10.0.0` - Image processing
- `tkinter` - GUI framework (usually included with Python)

## Usage

### GUI Mode (Recommended)
```bash
python visionapi.py --mode gui
```
or
```bash
python visionapi.py  # GUI is default
```

### CLI Mode
```bash
python visionapi.py --mode cli
```
- Press 'c' to capture and analyze image
- Press 'q' to quit

### Programmatic Usage

```python
from visionapi import VisionAPI

# Initialize
vision = VisionAPI()
vision.initialize_camera()

# Capture and analyze
frame = vision.capture_frame()
result = vision.analyze_image(frame, "What objects do you see?")

print(f"Answer: {result['answer']}")
print(f"Main object: {result['label']}")

# Cleanup
vision.cleanup()
```

### Integration Example

```python
from visionapi import VisionAPI

class MyApp:
    def __init__(self):
        self.vision = VisionAPI(env_file_path="path/to/your/.env")
        self.vision.initialize_camera()
    
    def analyze_scene(self, user_question):
        frame = self.vision.capture_frame()
        return self.vision.analyze_image(frame, user_question)
```

## API Reference

### VisionAPI Class

#### `__init__(env_file_path=".env")`
Initialize with environment file path containing `GEMINI_API_KEY`.

#### `initialize_camera(camera_index=0)`
Initialize webcam. Returns `True` on success.

#### `capture_frame()`
Capture current frame from webcam. Returns OpenCV frame or `None`.

#### `analyze_image(frame, user_query)`
Analyze frame with user query. Returns JSON:
```json
{
    "answer": "Detailed response to user query",
    "label": "primary_object_name"
}
```

#### `cleanup()`
Release camera resources.

### VisionGUI Class

#### `__init__()`
Initialize GUI interface.

#### `run()`
Start the GUI application.

## Output Format

The API always returns a JSON object with two fields:
- `answer`: Detailed response addressing the user's query
- `label`: Single word or short phrase identifying the main object

Example:
```json
{
    "answer": "I can see a red apple sitting on a wooden table. The apple appears fresh and has a glossy surface.",
    "label": "apple"
}
```

## Error Handling

The system handles various error conditions:
- Missing API key
- Camera initialization failures
- Network connectivity issues
- Invalid API responses

Errors are returned in the same JSON format:
```json
{
    "answer": "Error description",
    "label": "error"
}
```

## Troubleshooting

1. **Camera not working**: Check if camera is being used by another application
2. **API key errors**: Verify your `.env` file contains the correct `GEMINI_API_KEY`
3. **Import errors**: Run `pip install -r requirements.txt`
4. **GUI issues**: Ensure tkinter is installed (`sudo apt-get install python3-tk` on Ubuntu)

## Latest API Syntax

This implementation uses the latest Google Generative AI Python SDK (v0.3.2) with the following key features:
- `genai.configure(api_key=api_key)` for authentication
- `genai.GenerativeModel('gemini-1.5-flash')` for the latest model
- `model.generate_content([prompt, image])` for multimodal input
- PIL Image format for image input (converted from OpenCV frames)

## License

MIT License - feel free to use in your projects!