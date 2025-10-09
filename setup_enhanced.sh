#!/bin/bash
# Enhanced setup script for AI Glasses Pipeline

echo "🔧 Setting up AI Glasses Pipeline..."

# Check if uv is available, otherwise use pip
if command -v uv &> /dev/null; then
    INSTALLER="uv pip install"
    echo "📦 Using uv for package installation..."
else
    INSTALLER="pip install"
    echo "📦 Using pip for package installation..."
fi

# Update system packages
echo "📦 Updating system packages..."
sudo apt update

# Install system dependencies for audio and OpenCV
echo "🎵 Installing system dependencies..."
sudo apt install -y \
    portaudio19-dev \
    python3-pyaudio \
    python3-dev \
    libportaudio2 \
    libportaudiocpp0 \
    ffmpeg \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgl1-mesa-glx

# Install Python dependencies one by one to handle potential issues
echo "🐍 Installing Python dependencies..."

# Core dependencies
$INSTALLER numpy>=1.26.0
$INSTALLER opencv-python>=4.8.0
$INSTALLER Pillow>=10.0.0
$INSTALLER scipy>=1.11.0

# API dependencies
$INSTALLER python-dotenv>=1.0.0
$INSTALLER google-generativeai>=0.8.0
$INSTALLER deepgram-sdk>=3.0.0

# Audio dependency (may need special handling)
echo "🎤 Installing audio dependencies..."
$INSTALLER pyaudio>=0.2.11 || {
    echo "⚠️  pyaudio installation failed, trying alternative method..."
    sudo apt install -y python3-pyaudio
    pip install --force-reinstall pyaudio
}

# ML/AI dependencies
$INSTALLER torch>=2.0.0 torchvision>=0.15.0 --index-url https://download.pytorch.org/whl/cpu
$INSTALLER ultralytics>=8.0.0
$INSTALLER transformers>=4.30.0
$INSTALLER accelerate>=0.20.0
$INSTALLER filterpy>=1.4.5

echo "✅ Package installation complete!"

# Create .env template if it doesn't exist
if [ ! -f .env ]; then
    echo "📝 Creating .env template..."
    cat > .env << EOL
# Add your API keys here
DEEPGRAM_API_KEY=your_deepgram_api_key_here
GEMINI_API_KEY=your_gemini_api_key_here
EOL
    echo "⚠️  Please edit .env file and add your API keys"
fi

# Check for camera devices
echo "📷 Checking camera devices..."
if [ -e /dev/video0 ]; then
    echo "✅ Camera 0 found at /dev/video0"
else
    echo "❌ Camera 0 not found at /dev/video0"
fi

if [ -e /dev/video1 ]; then
    echo "✅ Camera 1 found at /dev/video1"
else
    echo "❌ Camera 1 not found at /dev/video1"
fi

# Test imports
echo "🧪 Testing Python imports..."
python3 -c "
try:
    import cv2
    print('✅ OpenCV imported successfully')
except ImportError as e:
    print(f'❌ OpenCV import failed: {e}')

try:
    import pyaudio
    print('✅ PyAudio imported successfully')
except ImportError as e:
    print(f'❌ PyAudio import failed: {e}')

try:
    import google.generativeai
    print('✅ Google Generative AI imported successfully')
except ImportError as e:
    print(f'❌ Google Generative AI import failed: {e}')

try:
    from deepgram import DeepgramClient
    print('✅ Deepgram imported successfully')
except ImportError as e:
    print(f'❌ Deepgram import failed: {e}')
"

# Check audio devices
echo "🎤 Checking audio devices..."
python3 -c "
try:
    import pyaudio
    p = pyaudio.PyAudio()
    print(f'Audio devices found: {p.get_device_count()}')
    for i in range(p.get_device_count()):
        info = p.get_device_info_by_index(i)
        if info['maxInputChannels'] > 0:
            print(f'  Input device {i}: {info[\"name\"]}')
    p.terminate()
except Exception as e:
    print(f'Audio device check failed: {e}')
"

echo "✅ Setup complete!"
echo "📋 Next steps:"
echo "   1. Edit .env file and add your API keys"
echo "   2. Ensure cameras are connected to /dev/video0 and /dev/video1"
echo "   3. Run: python3 aipipeline.py"