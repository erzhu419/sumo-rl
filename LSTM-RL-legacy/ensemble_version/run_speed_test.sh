#!/bin/bash
source ~/anaconda3/etc/profile.d/conda.sh
conda activate LSTM-RL
export SUMO_HOME=/usr/share/sumo
python test_speed_control.py
