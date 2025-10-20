#!/bin/bash

set -e

echo "=========================================="
echo "AI Glasses - Jetson Host Setup"
echo "=========================================="

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Check if running on Jetson
if [ ! -f /etc/nv_tegra_release ]; then
    echo -e "${RED}Error: This script must be run on a Jetson device${NC}"
    exit 1
fi

echo -e "${GREEN}✓ Running on Jetson device${NC}"

# Display JetPack version
if [ -f /etc/nv_tegra_release ]; then
    echo -e "${YELLOW}JetPack Version:${NC}"
    cat /etc/nv_tegra_release
fi

# Update system
echo -e "${YELLOW}Updating system packages...${NC}"
sudo apt-get update

# Install required system packages
echo -e "${YELLOW}Installing system dependencies...${NC}"
sudo apt-get install -y \
    git \
    curl \
    wget \
    python3-pip \
    v4l-utils \
    libportaudio2 \
    portaudio19-dev \
    usbutils

# Install Docker using JetsonHacks scripts
if ! command -v docker &> /dev/null; then
    echo -e "${YELLOW}Docker not found. Installing Docker using JetsonHacks scripts...${NC}"
    
    # Clone JetsonHacks install-docker repository
    TEMP_DIR=$(mktemp -d)
    cd "$TEMP_DIR"
    git clone https://github.com/jetsonhacks/install-docker.git
    cd install-docker
    
    # Install and configure Docker
    echo -e "${YELLOW}Installing Docker...${NC}"
    bash ./install_nvidia_docker.sh
    
    echo -e "${YELLOW}Configuring Docker (nvidia runtime and user group)...${NC}"
    bash ./configure_nvidia_docker.sh
    
    # Cleanup
    cd ~
    rm -rf "$TEMP_DIR"
    
    echo -e "${GREEN}✓ Docker installed and configured${NC}"
    echo -e "${YELLOW}⚠ Please log out and log back in for docker group changes to take effect${NC}"
else
    echo -e "${GREEN}✓ Docker already installed${NC}"
    
    # Check if nvidia is default runtime
    if ! sudo docker info 2>/dev/null | grep -q "Default Runtime: nvidia"; then
        echo -e "${YELLOW}Setting nvidia as default Docker runtime...${NC}"
        
        # Backup existing daemon.json
        if [ -f /etc/docker/daemon.json ]; then
            sudo cp /etc/docker/daemon.json /etc/docker/daemon.json.backup
        fi
        
        # Create or update daemon.json
        sudo tee /etc/docker/daemon.json > /dev/null <<EOF
{
    "runtimes": {
        "nvidia": {
            "path": "nvidia-container-runtime",
            "runtimeArgs": []
        }
    },
    "default-runtime": "nvidia"
}
EOF
        sudo systemctl restart docker
        echo -e "${GREEN}✓ Docker configured with nvidia runtime${NC}"
    fi
    
    # Add user to docker group if not already
    if ! groups $USER | grep -q docker; then
        echo -e "${YELLOW}Adding $USER to docker group...${NC}"
        sudo usermod -aG docker $USER
        echo -e "${YELLOW}⚠ Please log out and log back in for docker group changes to take effect${NC}"
    fi
fi

# Verify Docker configuration
echo -e "${YELLOW}Verifying Docker configuration...${NC}"
sudo docker info | grep -E "Default Runtime|Docker Root Dir" || true

# Setup jetson-containers if not present
if [ ! -d "$HOME/jetson-containers" ]; then
    echo -e "${YELLOW}Cloning jetson-containers...${NC}"
    cd $HOME
    git clone https://github.com/dusty-nv/jetson-containers
    cd jetson-containers
    bash install.sh
    echo -e "${GREEN}✓ jetson-containers installed${NC}"
else
    echo -e "${GREEN}✓ jetson-containers already present${NC}"
    echo -e "${YELLOW}Updating jetson-containers...${NC}"
    cd "$HOME/jetson-containers"
    git pull || echo "Could not update jetson-containers"
fi

# Check for swap
echo -e "${YELLOW}Checking swap configuration...${NC}"
SWAP_SIZE=$(free -h | grep Swap | awk '{print $2}')
if [ "$SWAP_SIZE" = "0B" ] || [ -z "$SWAP_SIZE" ]; then
    echo -e "${YELLOW}No swap detected. It's recommended to add swap for building containers.${NC}"
    echo -e "${YELLOW}You can manually set up swap using these commands:${NC}"
    echo "  sudo systemctl disable nvzramconfig"
    echo "  sudo fallocate -l 16G /mnt/16GB.swap"
    echo "  sudo mkswap /mnt/16GB.swap"
    echo "  sudo swapon /mnt/16GB.swap"
    echo "  # Add to /etc/fstab: /mnt/16GB.swap  none  swap  sw 0  0"
else
    echo -e "${GREEN}✓ Swap detected: $SWAP_SIZE${NC}"
fi

# Check power mode
echo -e "${YELLOW}Checking power mode...${NC}"
if command -v nvpmodel &> /dev/null; then
    POWER_MODE=$(sudo nvpmodel -q 2>/dev/null | grep "NV Power Mode" | head -1)
    echo -e "${GREEN}Current power mode: $POWER_MODE${NC}"
    echo -e "${YELLOW}For best performance, consider setting to MAXN mode:${NC}"
    echo "  sudo nvpmodel -m 0"
else
    echo -e "${YELLOW}nvpmodel not found${NC}"
fi

# Check camera devices
echo -e "${YELLOW}Checking camera devices...${NC}"
if [ -e /dev/video0 ]; then
    echo -e "${GREEN}✓ /dev/video0 detected${NC}"
    v4l2-ctl --device=/dev/video0 --list-formats-ext 2>/dev/null | head -20 || true
else
    echo -e "${RED}⚠ /dev/video0 not found${NC}"
fi

if [ -e /dev/video1 ]; then
    echo -e "${GREEN}✓ /dev/video1 detected${NC}"
    v4l2-ctl --device=/dev/video1 --list-formats-ext 2>/dev/null | head -20 || true
else
    echo -e "${RED}⚠ /dev/video1 not found${NC}"
fi

# Check audio devices
echo -e "${YELLOW}Checking audio devices...${NC}"
arecord -l 2>/dev/null || echo -e "${RED}⚠ No audio input devices found${NC}"

# Setup project directory structure
REPO_DIR="$HOME/githubrepos/aiglasses"
if [ ! -d "$REPO_DIR" ]; then
    echo -e "${YELLOW}Creating project directory and cloning repository...${NC}"
    mkdir -p "$HOME/githubrepos"
    cd "$HOME/githubrepos"
    git clone https://github.com/sim-daas/aiglasses.git
    echo -e "${GREEN}✓ Repository cloned${NC}"
else
    echo -e "${GREEN}✓ Repository already present${NC}"
    echo -e "${YELLOW}Updating repository...${NC}"
    cd "$REPO_DIR"
    git pull origin main || git pull origin master || echo "Could not pull latest changes"
fi

# Create NanoOwl models directory outside repo (in jetson-containers data directory)
NANOOWL_DATA_DIR="$HOME/jetson-containers/data/models/nanoowl"
mkdir -p "$NANOOWL_DATA_DIR"
echo -e "${GREEN}✓ NanoOwl models directory: $NANOOWL_DATA_DIR${NC}"

# Make scripts executable
chmod +x "$REPO_DIR/setup_container.sh" 2>/dev/null || true
chmod +x "$REPO_DIR/run_container.sh" 2>/dev/null || true

echo ""
echo -e "${GREEN}=========================================="
echo "✓ Jetson Host Setup Complete!"
echo "==========================================${NC}"
echo ""
echo "Next steps:"
echo "1. Log out and log back in (for docker group changes)"
echo "2. Create $REPO_DIR/.env file with your API keys:"
echo "   DEEPGRAM_API_KEY=your_key_here"
echo "   GEMINI_API_KEY=your_key_here"
echo "3. (Optional) Configure swap if not already set up"
echo "4. (Optional) Set power mode to MAXN: sudo nvpmodel -m 0"
echo "5. Run ./run_container.sh to start the container"
echo "6. Inside the container, run ./setup_container.sh"
echo ""
echo "NanoOwl models will be stored in: $NANOOWL_DATA_DIR"
echo ""
