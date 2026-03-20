# SUMO-RL Cloud Deployment Guide

This folder contains a standalone, lean deployment package for the SUMO-RL offline data collection.

## Option 1: Docker (Recommended for Cloud)
The easiest way to run in a consistent environment.

1.  **Build the Image**:
    *Prerequisite: Docker or Docker Desktop must be installed.*
    ```bash
    # Run from inside this folder
    docker build -t sumo-rl-offline .
    ```

2.  **Run the Container**:
    ```bash
    # Run in detached mode (background)
    docker run -d --name sumo-collector -v $(pwd)/offline_sumo/data:/app/offline_sumo/data sumo-rl-offline
    ```

3.  **Monitor Logs**:
    ```bash
    docker logs -f sumo-collector
    ```
    *Note: "Silence Mode" is enabled, so you will only see "Worker X: Finished Episode..." logs.*

---

## Option 2: Windows (WSL) / Ubuntu One-Click Setup (Native - No Docker Needed)

**Best for Windows Users**: This script runs natively on Ubuntu/WSL, avoiding the complexity of installing Docker Desktop.

If you have a fresh machine (e.g. WSL):

1.  **Preparation**:
    *   (Windows only) Ensure WSL is installed: `wsl --install`
    *   Copy this `deploy` folder to the machine.
    *   Enter the folder: `cd deploy`

2.  **One-Click Install**:
    ```bash
    # Automates apt-get install sumo, python, pip install...
    bash install_wsl.sh
    ```

3.  **Run**:
    ```bash
    # Activate environment
    source setup_env.sh
    
    # Run Collection
    ./run_mix_collection.sh
    ```

## Folder Structure
*   `run_mix_collection.sh`: Main entry point.
*   `offline_sumo/`: Core logic & environments.
*   `SUMO_ruiguang/`: Legacy control modules.
*   `logs/`: Contains required Expert Policy models.
*   `setup_env.sh`: Auto-configures `SUMO_HOME` and `PYTHONPATH`.
