import gym
from gym import spaces
import numpy as np
import sys
import os
from collections import deque

# Add project root to path to allow imports from SUMO_ruiguang
# Assuming offline_sumo/envs/sumo_env.py -> ../../ -> sumo-rl/
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

try:
    from offline_sumo.envs.bridge_v2 import SumoRLBridge, DecisionEvent
except ImportError as e:
    raise ImportError(f"Could not import local bridge_v2. Error: {e}")

class SumoBusHoldingEnv(gym.Env):
    """
    Gym Wrapper for SUMO Bus Holding Control.
    Uses SumoRLBridge for efficient event-driven simulation.
    """
    metadata = {'render.modes': ['human']}

    def __init__(self, root_dir=None, sumo_cfg="control_sim_traci_period.sumocfg", gui=False, max_steps=18000, seed=None):
        super(SumoBusHoldingEnv, self).__init__()
        
        if root_dir is None:
            # Default to SUMO_ruiguang static files
            self.root_dir = os.path.join(PROJECT_ROOT, "SUMO_ruiguang", "online_control")
        else:
            self.root_dir = root_dir

        self.bridge = SumoRLBridge(
            root_dir=self.root_dir,
            sumo_cfg=sumo_cfg,
            gui=gui,
            max_steps=max_steps,
            update_freq=10,
            seed=seed
        )

        # Observation Space (Matching rl_env.py)
        # 11 features: [line_idx, bus_idx, station_idx, time_idx, dir, fwd_h, bwd_h, wait, target, duration, time]
        self.observation_space = spaces.Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(11,), 
            dtype=np.float32
        )

        # Action Space: Holding time (0 to 60 seconds)
        self.action_space = spaces.Box(
            low=0.0, 
            high=60.0, 
            shape=(1,), 
            dtype=np.float32
        )

        self.last_event = None
        self.event_queue = deque()
        self.done = False

        # Caching for indexing (simple hash maps)
        self._line_map = {}
        self._bus_map = {}
        self._station_map = {}
    
    def reset(self):
        """
        Resets the simulation.
        Returns:
            obs: The observation of the first bus needing a decision.
        """
        self.bridge.reset()
        self.event_queue.clear()
        self.done = False
        self.last_event = None
        
        # Clear maps on reset if needed, or keep persistent
        # self._line_map.clear() 
        
        return self._advance_to_next_decision()

    def step(self, action):
        """
        Apply action to the LAST event vehicle, then advance to NEXT event vehicle.
        Args:
            action: Holding time (float or array)
        Returns:
            obs: Observation for the NEXT vehicle
            reward: Reward for the PREVIOUS vehicle (Simulated delayed reward)
            done: bool
            info: dict containing 'snapshot' and previous event details
        """
        if self.done:
            return np.zeros(self.observation_space.shape), 0.0, True, {}

        # 1. Apply action to the vehicle waiting for decision (self.last_event)
        if self.last_event:
            # Unwrap action if generic
            if isinstance(action, (list, np.ndarray)):
                val = float(action[0])
            else:
                val = float(action)
            
            self.bridge.apply_action(self.last_event, val)
        else:
            # Specific edge case: step called without active event? 
            pass

        # 2. Compute Reward (Standard or H2O+ Logic)
        # Note: In offline RL, we might compute reward LATER during data processing, 
        # but for Gym compatibility we return a proxy here.
        # Ideally, we should calculcate the reward for the *applied* action.
        # However, due to the nature of "decision -> wait -> arrival at next stop",
        # the true reward is only available at next stop.
        # For this Env wrapper, we implement specific immediate reward logic 
        # based on deviation from target, similar to `rl_env.py`.
        reward = 0.0
        if self.last_event:
            reward = self._compute_reward(self.last_event)

        # 3. Advance simulation until new event
        obs = self._advance_to_next_decision()
        
        # 4. Construct Info (Snapshot placeholder)
        info = {
            'last_event_bus_id': self.last_event.bus_id if self.last_event else None,
            'sim_time': self.bridge.current_time,
            'forward_bus_present': self.last_event.forward_bus_present if self.last_event else False,
            'backward_bus_present': self.last_event.backward_bus_present if self.last_event else False,
        }

        return obs, reward, self.done, info

    def _advance_to_next_decision(self):
        """Loops simulation until event queue has items."""
        while not self.event_queue and not self.done:
            events, terminated, departed = self.bridge.fetch_events()
            
            if terminated:
                self.done = True
                break
            
            if events:
                self.event_queue.extend(events)
        
        if self.done and not self.event_queue:
            return np.zeros(self.observation_space.shape, dtype=np.float32)

        # Pop the next event to be processed
        self.last_event = self.event_queue.popleft()
        return self._encode_observation(self.last_event)

    def _encode_observation(self, event: DecisionEvent):
        # Maps IDs to indices dynamically
        line_idx = self._get_id(self._line_map, event.line_id)
        bus_idx = self._get_id(self._bus_map, event.bus_id)
        station_key = f"{event.line_id}_{event.stop_id}"
        station_idx = self._get_id(self._station_map, station_key)
        time_idx = int(event.sim_time // 3600)

        # Vector matching rl_env.py
        obs = np.array([
            float(line_idx),
            float(bus_idx),
            float(station_idx),
            float(time_idx),
            float(event.direction),
            float(event.forward_headway),
            float(event.backward_headway),
            float(event.waiting_passengers),
            float(event.target_forward_headway), # Using target as feature
            float(event.base_stop_duration),
            float(event.sim_time),
        ], dtype=np.float32)
        return obs

    def _get_id(self, map_dict, key):
        if key not in map_dict:
            map_dict[key] = len(map_dict)
        return map_dict[key]

    def _compute_reward(self, event: DecisionEvent):
        """
        Linear negative penalty reward function.
        """
        def headway_reward(headway, target):
            return -abs(headway - target)

        fwd_target = event.target_forward_headway
        bwd_target = event.target_backward_headway

        forward_reward = headway_reward(event.forward_headway, fwd_target) if event.forward_bus_present else None
        backward_reward = headway_reward(event.backward_headway, bwd_target) if event.backward_bus_present else None

        if forward_reward is not None and backward_reward is not None:
            fwd_dev = abs(event.forward_headway - fwd_target)
            bwd_dev = abs(event.backward_headway - bwd_target)
            weight = fwd_dev / (fwd_dev + bwd_dev + 1e-6)
            
            similarity_bonus = -abs(event.forward_headway - event.backward_headway) * 0.5
            reward = forward_reward * weight + backward_reward * (1 - weight) + similarity_bonus
        elif forward_reward is not None:
            reward = forward_reward
        elif backward_reward is not None:
            reward = backward_reward
        else:
            reward = -50.0

        if (event.forward_bus_present and abs(event.forward_headway - fwd_target) > 180) or \
           (event.backward_bus_present and abs(event.backward_headway - bwd_target) > 180):
            reward -= 20.0
            
        return reward

    def close(self):
        self.bridge.close()
