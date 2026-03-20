#!/bin/bash

# One-Click Install Script for Fresh Ubuntu / WSL
# Usage: bash install_wsl.sh


# Exit on error
set -e

echo "[Install] Updating System..."
sudo apt-get update

echo "[Install] Installing System Dependencies (SUMO, Git, Python)..."
sudo apt-get install -y software-properties-common
sudo add-apt-repository ppa:sumo/stable -y
sudo apt-get update
sudo apt-get install -y sumo sumo-tools sumo-doc git python3-pip python3-dev libgdal-dev dos2unix

echo "[Install] Robustness (Fixing potential Windows CRLF issues)..."
find . -name "*.sh" -exec dos2unix {} +
find . -name "*.py" -exec dos2unix {} +
chmod +x *.sh


echo "[Install] Installing Python Libraries..."
# Upgrade pip first
python3 -m pip install --upgrade pip --break-system-packages || python3 -m pip install --upgrade pip
# Install requirements
pip3 install -r requirements.txt --break-system-packages || pip3 install -r requirements.txt

# Install libsumo (binary wheel, usually easiest for WSL)
pip3 install libsumo --break-system-packages || pip3 install libsumo

echo "[Install] Configuration..."
# Source the setup script to check env vars immediately
source setup_env.sh

echo ""
echo "========================================================"
echo "✅ Installation Complete!"
echo "Please run the following command to activate the environment for this session:"
echo "source deploy/setup_env.sh"
echo "========================================================"
