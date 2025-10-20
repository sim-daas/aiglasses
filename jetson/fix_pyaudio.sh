#!/bin/bash

echo "🔧 Quick PyAudio Fix for Jetson"
echo "================================"
echo ""

# Install system dependencies
echo "📦 Installing system dependencies..."
sudo apt update
sudo apt install -y portaudio19-dev python3-pyaudio libasound2-dev

# Try pip install
echo "🎤 Attempting pip install pyaudio..."
pip install pyaudio

if [ $? -ne 0 ]; then
    echo ""
    echo "⚠️  pip install failed, using system package workaround..."
    
    # Install system package
    sudo apt install -y python3-pyaudio
    
    # If in virtual environment, copy system PyAudio
    if [ ! -z "$VIRTUAL_ENV" ]; then
        echo "📋 Copying system PyAudio to virtual environment..."
        
        PYTHON_VERSION=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
        SYSTEM_SITE="/usr/lib/python3/dist-packages"
        VENV_SITE="$VIRTUAL_ENV/lib/python$PYTHON_VERSION/site-packages"
        
        # Copy PyAudio files
        cp -r $SYSTEM_SITE/pyaudio* $VENV_SITE/ 2>/dev/null
        cp -r $SYSTEM_SITE/_portaudio* $VENV_SITE/ 2>/dev/null
        cp -r $SYSTEM_SITE/PyAudio* $VENV_SITE/ 2>/dev/null
        
        echo "✅ Files copied"
    fi
fi

# Test import
echo ""
echo "🧪 Testing PyAudio import..."
python3 -c "import pyaudio; print('✅ PyAudio works! Version:', pyaudio.__version__)"

if [ $? -eq 0 ]; then
    echo ""
    echo "🎉 PyAudio is now working!"
else
    echo ""
    echo "❌ PyAudio still not working"
    echo ""
    echo "Manual steps:"
    echo "  1. sudo apt install python3-pyaudio"
    echo "  2. python3 -c 'import sys; print(sys.path)'"
    echo "  3. Find where pyaudio is installed"
    echo "  4. Copy to your site-packages directory"
fi
