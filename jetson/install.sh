#!/bin/bash

echo "🚀 AURA AI Glasses - Complete Installation"
echo "=========================================="
echo ""

# Check if running in virtual environment
if [ -z "$VIRTUAL_ENV" ]; then
    echo "⚠️  Warning: Not in a virtual environment"
    echo "   Recommended: python3 -m venv aiglass && source aiglass/bin/activate"
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Step 1: System dependencies
echo "📦 Step 1/4: Installing system dependencies..."
sudo apt update
sudo apt install -y \
    portaudio19-dev \
    python3-pyaudio \
    libasound2-dev \
    libportaudio2 \
    libportaudiocpp0 \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    build-essential \
    python3-dev \
    cmake \
    pkg-config

if [ $? -ne 0 ]; then
    echo "❌ System dependencies installation failed!"
    exit 1
fi

echo "✅ System dependencies installed"
echo ""

# Step 2: Upgrade pip
echo "📦 Step 2/4: Upgrading pip..."
python3 -m pip install --upgrade pip setuptools wheel

# Step 3: Install PyAudio separately (most common failure point)
echo "🎤 Step 3/4: Installing PyAudio..."

# Try pip install first
python3 -m pip install pyaudio

if [ $? -ne 0 ]; then
    echo "⚠️  pip install pyaudio failed, trying alternative method..."
    
    # Method 2: Install from system package then copy to venv
    sudo apt install -y python3-pyaudio
    
    # Find system PyAudio and copy to venv if in one
    if [ ! -z "$VIRTUAL_ENV" ]; then
        SYSTEM_PYAUDIO=$(python3 -c "import sys; print([p for p in sys.path if 'dist-packages' in p][0])" 2>/dev/null)
        if [ ! -z "$SYSTEM_PYAUDIO" ]; then
            echo "Copying system PyAudio to virtual environment..."
            cp -r "$SYSTEM_PYAUDIO/pyaudio"* "$VIRTUAL_ENV/lib/python3.10/site-packages/" 2>/dev/null
            cp -r "$SYSTEM_PYAUDIO/_portaudio"* "$VIRTUAL_ENV/lib/python3.10/site-packages/" 2>/dev/null
        fi
    fi
fi

# Verify PyAudio installation
python3 -c "import pyaudio; print('✅ PyAudio version:', pyaudio.__version__)" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "❌ PyAudio installation failed!"
    echo "   Manual fix: sudo apt install python3-pyaudio"
    exit 1
fi

echo ""

# Step 4: Install remaining dependencies
echo "📦 Step 4/4: Installing remaining Python packages..."

# Install packages one by one to catch individual failures
PACKAGES=(
    "opencv-python>=4.8.0"
    "numpy>=1.24.0"
    "pillow>=10.0.0"
    "flask>=3.0.0"
    "flask-socketio>=5.3.0"
    "flask-cors>=4.0.0"
    "python-socketio>=5.10.0"
    "eventlet>=0.33.0"
    "scipy>=1.11.0"
    "google-generativeai>=0.3.0"
    "python-dotenv>=1.0.0"
    "requests>=2.31.0"
)

for pkg in "${PACKAGES[@]}"; do
    echo "Installing $pkg..."
    python3 -m pip install "$pkg"
    if [ $? -ne 0 ]; then
        echo "⚠️  Warning: Failed to install $pkg"
    fi
done

echo ""
echo "✅ Installation complete!"
echo ""

# Step 5: Verify installation
echo "🧪 Verifying installation..."
python3 << EOF
import sys
modules = {
    'cv2': 'OpenCV',
    'numpy': 'NumPy',
    'PIL': 'Pillow',
    'flask': 'Flask',
    'flask_socketio': 'Flask-SocketIO',
    'pyaudio': 'PyAudio',
    'google.generativeai': 'Gemini API',
    'dotenv': 'python-dotenv'
}

failed = []
for module, name in modules.items():
    try:
        __import__(module)
        print(f'✅ {name}')
    except ImportError:
        print(f'❌ {name} - FAILED')
        failed.append(name)

if failed:
    print(f'\n⚠️  Failed modules: {", ".join(failed)}')
    sys.exit(1)
else:
    print('\n✅ All modules imported successfully!')
EOF

if [ $? -ne 0 ]; then
    echo ""
    echo "❌ Some packages failed to install"
    echo "   Check the errors above and install manually"
    exit 1
fi

echo ""
echo "🎉 AURA AI Glasses setup complete!"
echo ""
echo "📝 Next steps:"
echo "   1. Edit .env file: nano ../.env"
echo "   2. Add your GEMINI_API_KEY"
echo "   3. Run: python3 main.py"
echo ""
