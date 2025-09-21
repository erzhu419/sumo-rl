from sb3_contrib import TRPO

from env import SumoEnv

e = SumoEnv(gui=False, noWarnings=True, epLen=50, traffic=10, mixedConfigs=True, bunched=False)

model = TRPO("MlpPolicy", e, verbose=1, learning_rate=0.001, tensorboard_log="tensorboard/trpoTraffic", device="cpu")

model.learn(total_timesteps=100, log_interval=1)  # Complete at least one episode (epLen=50)
model.save("models/trpoTraffic")

e.close()