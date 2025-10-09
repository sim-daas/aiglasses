#!/bin/bash

# Setup script for AI Glasses Pipeline

echo "🔧 Setting up AI Glasses Pipeline..."

# Update system packages
echo "📦 Updating system packages..."
sudo apt update

# Install system dependencies for audio
echo "🎵 Installing audio dependencies..."
sudo apt install -y portaudio19-dev python3-pyaudio

# Install Python dependencies
echo "🐍 Installing Python dependencies..."
pip install -r requirements.txt

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

# Check audio devices
echo "🎤 Checking audio devices..."
python3 -c "
import pyaudio
p = pyaudio.PyAudio()
print(f'Audio devices found: {p.get_device_count()}')
for i in range(p.get_device_count()):
    info = p.get_device_info_by_index(i)
    if info['maxInputChannels'] > 0:
        print(f'  Input device {i}: {info[\"name\"]}')
p.terminate()
"

echo "✅ Setup complete!"
echo "📋 Next steps:"
echo "   1. Edit .env file and add your API keys"
echo "   2. Ensure cameras are connected to /dev/video0 and /dev/video1"
echo "   3. Run: python3 aipipeline.py"