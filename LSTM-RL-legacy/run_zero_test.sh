source ~/anaconda3/etc/profile.d/conda.sh
conda activate LSTM-RL

echo "--- VANILLA ZERO ---"
python sac_zero_vanilla.py --train --use_sumo_env --max_episodes 1

echo "--- ENSEMBLE ZERO ---"
python sac_zero_ensemble.py --train --use_sumo_env --max_episodes 1
