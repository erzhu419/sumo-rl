# Base Image: Use official Python + Ubuntu base
FROM ubuntu:22.04

# Prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Update and install system dependencies (including SUMO)
RUN apt-get update && apt-get install -y \
    software-properties-common \
    wget \
    git \
    python3-pip \
    python3-dev \
    libgdal-dev \
    && add-apt-repository ppa:sumo/stable \
    && apt-get update \
    && apt-get install -y sumo sumo-tools sumo-doc \
    && rm -rf /var/lib/apt/lists/*

# Set Environment Variables for SUMO
ENV SUMO_HOME=/usr/share/sumo
ENV PYTHONPATH=$SUMO_HOME/tools:$PYTHONPATH

# Create working directory
WORKDIR /app

# Install Python Dependencies first (caching layer)
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt
# Install libsumo specifically
RUN pip3 install libsumo

# Copy Application Code
# Note: You should ideally use a .dockerignore to exclude large data files if needed
COPY . /app

# Set PYTHONPATH to include project root and submodules
ENV PYTHONPATH=/app:$PYTHONPATH

# Default Command (Run Mixed Data Collection)
CMD ["bash", "run_mix_collection.sh"]
