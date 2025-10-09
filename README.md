# AI Glasses Pipeline

Complete pipeline combining stereo vision, voice recognition, AI vision analysis, and 3D object detection.

## Features

- **Stereo Vision**: Uses two cameras (/dev/video0 and /dev/video1) for depth perception
- **Voice Recognition**: Press-to-talk functionality using Deepgram API
- **AI Vision Analysis**: Gemini Vision API for understanding scenes and answering questions
- **3D Object Detection**: NanoOwl for object detection with 3D bounding boxes
- **Real-time Processing**: Live camera feed with overlay information

## Setup

1. **Install Dependencies**:
   ```bash
   chmod +x setup.sh
   ./setup.sh
   ```

2. **Configure API Keys**:
   Edit the `.env` file and add your API keys:
   ```
   DEEPGRAM_API_KEY=your_deepgram_api_key_here
   GEMINI_API_KEY=your_gemini_api_key_here
   ```

3. **Connect Cameras**:
   - Connect stereo cameras to /dev/video0 and /dev/video1
   - Ensure cameras are recognized by the system

## Usage

1. **Start the Pipeline**:
   ```bash
   python3 aipipeline.py
   ```

2. **Using the Interface**:
   - Click "Start Pipeline" to begin camera capture
   - Hold the microphone button and speak your query
   - Release the button to process voice and vision
   - View results in the status panels

## Pipeline Flow

1. **Voice Input**: Hold mic button → record audio → release button
2. **Voice Recognition**: Audio sent to Deepgram API for transcription
3. **Frame Capture**: Current video frame saved for analysis
4. **Vision Analysis**: Frame + query sent to Gemini Vision API
5. **Object Detection**: Object label used with NanoOwl for detection
6. **3D Visualization**: 2D detections + depth map → 3D bounding boxes
7. **Display**: Results overlaid on live video feed

## Components

- `aipipeline.py` - Main pipeline application
- `visionapi.py` - Gemini Vision API integration
- `bbox3d_utils.py` - 3D bounding box utilities
- `owl_predict.py` - NanoOwl object detection
- `depth.py` - Stereo depth calculation
- `detect.py` - Detection visualization
- `3dtext.py` - 3D text effects

## Requirements

- Python 3.8+
- OpenCV 4.8+
- Two USB cameras for stereo vision
- Microphone for voice input
- Deepgram API key
- Gemini API key
- NanoOwl model files (will be downloaded automatically)

## Troubleshooting

- **Camera Issues**: Check `/dev/video0` and `/dev/video1` exist
- **Audio Issues**: Verify microphone permissions and PyAudio installation
- **API Issues**: Ensure API keys are correct in `.env` file
- **Model Issues**: NanoOwl models will be downloaded on first run

## API Keys

Get your API keys from:
- [Deepgram Console](https://console.deepgram.com/)
- [Google AI Studio](https://makersuite.google.com/app/apikey)