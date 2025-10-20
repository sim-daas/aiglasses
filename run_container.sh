#!/bin/bash

set -e

echo "=========================================="
echo "AI Glasses - Starting Container"
echo "=========================================="

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

# Get the directory where the script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
REPO_DIR="$SCRIPT_DIR"

# Check if repository exists
if [ ! -d "$REPO_DIR" ]; then
    echo -e "${RED}Error: Repository directory not found at $REPO_DIR${NC}"
    exit 1
fi

# Check if .env file exists
if [ ! -f "$REPO_DIR/.env" ]; then
    echo -e "${RED}Error: .env file not found${NC}"
    echo -e "${YELLOW}Please create $REPO_DIR/.env with your API keys:${NC}"
    echo "DEEPGRAM_API_KEY=your_key_here"
    echo "GEMINI_API_KEY=your_key_here"
    exit 1
fi

# Check for jetson-containers
if [ ! -d "$HOME/jetson-containers" ]; then
    echo -e "${RED}Error: jetson-containers not found. Please run setup_jetson_host.sh first${NC}"
    exit 1
fi

# NanoOwl models directory (outside repo)
NANOOWL_DATA_DIR="$HOME/jetson-containers/data/models/nanoowl"
mkdir -p "$NANOOWL_DATA_DIR"

# Check if container is already running
if docker ps --format '{{.Names}}' | grep -q "^aiglasses-container$"; then
    echo -e "${GREEN}Container 'aiglasses-container' is already running${NC}"
    read -p "Attach to running container? (Y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Nn]$ ]]; then
        docker exec -it aiglasses-container bash
        exit 0
    else
        exit 0
    fi
fi

# Check if container exists but is stopped
if docker ps -a --format '{{.Names}}' | grep -q "^aiglasses-container$"; then
    echo -e "${YELLOW}Container 'aiglasses-container' exists but is stopped${NC}"
    read -p "Start existing container? (Y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Nn]$ ]]; then
        docker start aiglasses-container
        docker exec -it aiglasses-container bash
        exit 0
    else
        read -p "Remove and recreate container? (y/N) " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            echo -e "${YELLOW}Removing existing container...${NC}"
            docker rm -f aiglasses-container
        else
            exit 0
        fi
    fi
fi

echo -e "${YELLOW}Starting jetson-containers with nanoowl and opencv:cuda...${NC}"
echo -e "${YELLOW}Repository mounted at: /opt/aiglasses${NC}"
echo -e "${YELLOW}NanoOwl models at: /data/models/nanoowl${NC}"
echo ""

# Change to jetson-containers directory
cd "$HOME/jetson-containers"

# Run container with all necessary mounts and devices
# Using jetson-containers run.sh which wraps docker run
echo -e "${YELLOW}Running container...${NC}"
jetson-containers run \
    --name aiglasses-container \
    --volume "$REPO_DIR:/opt/aiglasses:rw" \
    --volume "$NANOOWL_DATA_DIR:/data/models/nanoowl:rw" \
    --device /dev/video0 \
    --device /dev/video1 \
    --device /dev/snd \
    --env-file "$REPO_DIR/.env" \
    --workdir /opt/aiglasses \
    $(jetson-containers autotag nanoowl) \
    opencv:cuda

echo ""
echo -e "${GREEN}Container session ended.${NC}"
echo ""
echo "To re-attach to the container, run:"
echo "  docker exec -it aiglasses-container bash"
echo ""
echo "To stop the container:"
echo "  docker stop aiglasses-container"
echo ""
echo "To remove the container:"
echo "  docker rm -f aiglasses-container"
echo ""
