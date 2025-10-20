#!/bin/bash

echo "🔧 Setting up AURA AI Glasses System..."

# Update system packages
echo "📦 Updating system packages..."
sudo apt update

# Install system dependencies for PyAudio
echo "🎤 Installing audio system dependencies..."
sudo apt install -y \
    portaudio19-dev \
    python3-pyaudio \
    libasound2-dev \
    libportaudio2 \
    libportaudiocpp0

# Install OpenCV system dependencies
echo "📷 Installing OpenCV system dependencies..."
sudo apt install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1

# Install build tools (needed for some pip packages)
echo "🔨 Installing build tools..."
sudo apt install -y \
    build-essential \
    python3-dev \
    cmake \
    pkg-config

echo "✅ System dependencies installed!"
echo ""
echo "📝 Now run: pip install -r requirements.txt"
