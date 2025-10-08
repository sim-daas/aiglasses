#!/bin/bash

# Setup script for AI Vision API
echo "Setting up AI Vision API..."

# Check if Python 3 is installed
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is not installed. Please install Python 3.8 or higher."
    exit 1
fi

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install packages from requirements.txt
echo "Installing required packages..."
pip install -r requirements.txt

# Create .env file if it doesn't exist
if [ ! -f ".env" ]; then
    echo "Creating .env file..."
    cp .env.example .env
    echo ""
    echo "⚠️  IMPORTANT: Please edit .env file and add your Gemini API key!"
    echo "   Get your API key from: https://makersuite.google.com/app/apikey"
    echo ""
fi

echo "✅ Setup complete!"
echo ""
echo "To run the application:"
echo "1. Activate the virtual environment: source venv/bin/activate"
echo "2. Add your Gemini API key to the .env file"
echo "3. Run: python3 visionapi.py"
echo ""
echo "GUI Mode: python3 visionapi.py"
echo "CLI Mode: python3 visionapi.py --mode cli"