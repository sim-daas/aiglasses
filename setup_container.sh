#!/bin/bash

set -e

echo "=========================================="
echo "AI Glasses - Container Setup"
echo "=========================================="

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

# Navigate to aiglasses directory
cd /opt/aiglasses || cd /home/ubuntu/githubrepos/aiglasses || {
    echo -e "${RED}Error: Could not find aiglasses directory${NC}"
    exit 1
}

echo -e "${YELLOW}Current directory: $(pwd)${NC}"

echo -e "${YELLOW}Installing Python dependencies...${NC}"

# Upgrade pip
pip3 install --upgrade pip

# Install core dependencies
echo -e "${YELLOW}Installing core packages...${NC}"
pip3 install \
    numpy \
    opencv-python \
    pillow \
    python-dotenv \
    pyaudio

# Install Google AI packages
echo -e "${YELLOW}Installing Google AI packages...${NC}"
pip3 install \
    google-generativeai \
    deepgram-sdk

# Install additional dependencies
echo -e "${YELLOW}Installing additional packages...${NC}"
pip3 install \
    requests \
    scipy

# Check if running on Jetson (has CUDA support)
if command -v nvcc &> /dev/null; then
    echo -e "${GREEN}✓ CUDA detected, container is GPU-enabled${NC}"
    nvcc --version | head -1
else
    echo -e "${YELLOW}⚠ CUDA not detected in PATH${NC}"
fi

# Verify OpenCV CUDA support
echo -e "${YELLOW}Checking OpenCV CUDA support...${NC}"
python3 -c "import cv2; print('OpenCV Version:', cv2.__version__); print('CUDA Devices:', cv2.cuda.getCudaEnabledDeviceCount())" 2>/dev/null || echo "Could not check OpenCV CUDA"

# Check camera devices
echo -e "${YELLOW}Checking camera devices...${NC}"
if [ -e /dev/video0 ]; then
    echo -e "${GREEN}✓ /dev/video0 available${NC}"
else
    echo -e "${RED}⚠ /dev/video0 not found${NC}"
fi

if [ -e /dev/video1 ]; then
    echo -e "${GREEN}✓ /dev/video1 available${NC}"
else
    echo -e "${RED}⚠ /dev/video1 not found${NC}"
fi

# Check audio devices
echo -e "${YELLOW}Checking audio devices...${NC}"
if [ -e /dev/snd ]; then
    echo -e "${GREEN}✓ /dev/snd available${NC}"
else
    echo -e "${RED}⚠ /dev/snd not found${NC}"
fi

# Check NanoOwl models directory
if [ -d /data/models/nanoowl ]; then
    echo -e "${GREEN}✓ NanoOwl models directory mounted at: /data/models/nanoowl${NC}"
else
    echo -e "${YELLOW}⚠ NanoOwl models directory not found at /data/models/nanoowl${NC}"
fi

# Verify Python imports
echo -e "${YELLOW}Verifying Python imports...${NC}"
python3 << EOF
import sys
packages = {
    'cv2': 'OpenCV',
    'numpy': 'NumPy',
    'PIL': 'Pillow',
    'google.generativeai': 'Google Generative AI',
    'deepgram': 'Deepgram SDK',
    'pyaudio': 'PyAudio',
    'dotenv': 'python-dotenv'
}

failed = []
for module, name in packages.items():
    try:
        __import__(module)
        print(f'✓ {name}')
    except ImportError as e:
        print(f'✗ {name}: {e}')
        failed.append(name)

if failed:
    print(f'\nFailed to import: {", ".join(failed)}')
    sys.exit(1)
else:
    print('\n✓ All packages imported successfully')
EOF

# Check for .env file
if [ -f .env ]; then
    echo -e "${GREEN}✓ .env file found${NC}"
    
    # Verify API keys are set
    if grep -q "DEEPGRAM_API_KEY=" .env && grep -q "GEMINI_API_KEY=" .env; then
        # Check if they're not empty or default values
        if grep -q "DEEPGRAM_API_KEY=$" .env || grep -q "GEMINI_API_KEY=$" .env || \
           grep -q "your_key_here" .env || grep -q "your_deepgram_api_key_here" .env || \
           grep -q "your_gemini_api_key_here" .env; then
            echo -e "${YELLOW}⚠ Warning: API keys appear to be empty or contain default values${NC}"
            echo -e "${YELLOW}  Please update .env with your actual API keys${NC}"
        else
            echo -e "${GREEN}✓ API keys appear to be configured${NC}"
        fi
    else
        echo -e "${RED}⚠ API keys not found in .env file${NC}"
    fi
else
    echo -e "${RED}⚠ .env file not found${NC}"
    echo -e "${YELLOW}Please create .env file on host with your API keys${NC}"
fi

# Make Python scripts executable
chmod +x aipipeline.py 2>/dev/null || true
chmod +x visionapi.py 2>/dev/null || true
chmod +x bbox3d_utils.py 2>/dev/null || true
chmod +x owl_predict.py 2>/dev/null || true

echo ""
echo -e "${GREEN}=========================================="
echo "✓ Container Setup Complete!"
echo "==========================================${NC}"
echo ""
echo "NanoOwl models location: /data/models/nanoowl"
echo ""
echo "You can now run the AI pipeline:"
echo "  python3 aipipeline.py"
echo ""
echo "Or test individual components:"
echo "  python3 visionapi.py"
echo "  python3 owl_predict.py"
echo ""
