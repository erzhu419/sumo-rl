#!/bin/bash

# Cloud Deployment Setup Script
# Usage: source setup_env.sh

echo "[Setup] Configuring Environment Variables..."

# 1. SUMO Setup
if [ -z "$SUMO_HOME" ]; then
    # Try typical locations
    if [ -d "/usr/share/sumo" ]; then
        export SUMO_HOME="/usr/share/sumo"
    elif [ -d "/usr/local/share/sumo" ]; then
        export SUMO_HOME="/usr/local/share/sumo"
    else
        echo "WARNING: SUMO_HOME not found. Please set it manually."
    fi
fi
echo "SUMO_HOME=$SUMO_HOME"

# 2. PYTHONPATH Setup
# Add SUMO tools
if [[ ":$PYTHONPATH:" != *":$SUMO_HOME/tools:"* ]]; then
    export PYTHONPATH="$SUMO_HOME/tools:$PYTHONPATH"
fi

# Add Project Root (Current Directory)
export PROJECT_ROOT=$(pwd)
if [[ ":$PYTHONPATH:" != *":$PROJECT_ROOT:"* ]]; then
    export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"
fi

echo "PYTHONPATH=$PYTHONPATH"

# 3. Check Dependencies
echo "[Setup] Checking Python Dependencies..."
python3 -c "import torch; import gym; import libsumo; print('All key imports successful.')"

echo "[Setup] Ready to run!"
