
"""Single worker: evaluates checkpoints using EXACT training loop logic."""
import sys, os, json, time, copy
import numpy as np, torch
from collections import defaultdict

ENSEMBLE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(ENSEMBLE_DIR, '..', '..', '..'))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'LSTM-RL-legacy'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
    sys.path.insert(0, os.path.join(PROJECT_ROOT, 'LSTM-RL-legacy'))

# Spoof sys.argv to match training configuration
sys.argv = [
    'sac_ensemble_SUMO_linear_penalty.py',
    '--use_sumo_env',
    '--sumo_bridge', 'SUMO_ruiguang.online_control.rl_bridge:build_bridge',
    '--use_residual_control',
    '--bang_bang',
    '--run_name', 'Production_Augmented_BangBang_V7_Long',
]

from ensemble_version.sac_ensemble_SUMO_linear_penalty import (
    env, sac_trainer, args, safe_initialize_state, safe_step,
    build_action_template, action_dim, get_reward_value, device,
)


def collect_eval_metrics(sac_trainer, states_for_metrics):
    """Compute auxiliary metrics from collected eval states."""
    metrics = {
        'reg_norm1': 0.0, 'reg_norm2': 0.0,
        'q_value_mean': 0.0, 'q_std_mean': 0.0,
        'ood_loss': 0.0, 'log_prob_mean': 0.0,
    }
    try:
        reg_norm = sac_trainer.compute_reg_norm(sac_trainer.target_soft_q_net)
        if reg_norm.numel() > 1:
            metrics['reg_norm1'] = args.weight_reg * reg_norm[0].item()
            metrics['reg_norm2'] = args.weight_reg * reg_norm[1].item()
        else:
            metrics['reg_norm1'] = args.weight_reg * reg_norm.item()
            metrics['reg_norm2'] = args.weight_reg * reg_norm.item()
    except Exception as e:
        print(f"  reg_norm failed: {e}", flush=True)

    if len(states_for_metrics) > 0:
        try:
            state_batch = torch.stack(states_for_metrics[:256]).to(device)
            with torch.no_grad():
                new_action, log_prob = sac_trainer.policy_net.evaluate(state_batch)
                q_pred = sac_trainer.soft_q_net(state_batch, new_action)
                metrics['q_value_mean'] = q_pred.mean().item()
                metrics['q_std_mean'] = q_pred.std(dim=0).mean().item()
                metrics['ood_loss'] = q_pred.std(dim=0).mean().item()
                metrics['log_prob_mean'] = -log_prob.mean().item()
        except Exception as e:
            print(f"  eval metrics failed: {e}", flush=True)
    return metrics


def eval_episode(ep, model_dir):
    """
    Evaluate a single checkpoint using the EXACT same logic as the training loop
    (sac_ensemble_SUMO_linear_penalty.py lines 878-1100), minus gradient updates.
    """
    ckpt = os.path.join(model_dir, f"checkpoint_episode_{ep}")
    if not os.path.exists(ckpt + "_policy"):
        return None

    # Load checkpoint
    sac_trainer.load_model(ckpt)
    sac_trainer.policy_net.eval()
    sac_trainer.soft_q_net.eval()
    sac_trainer.target_soft_q_net.load_state_dict(sac_trainer.soft_q_net.state_dict())

    # ── Exact copy of training loop init (lines 878-892) ──
    env.reset()
    state_dict, reward_dict, _ = safe_initialize_state(env, render=False)

    done = False
    episode_steps = 0
    action_dict = build_action_template(state_dict)
    last_action_history = defaultdict(
        lambda: defaultdict(lambda: np.zeros(action_dim, dtype=np.float32))
    )
    episode_reward = 0
    station_feature_idx = sac_trainer.station_feature_idx if sac_trainer.station_feature_idx is not None else 1
    step = 0
    states_for_metrics = []

    # ── Exact copy of training loop body (lines 894-1082) ──
    while not done:
        action_dict = build_action_template(state_dict, action_dict)

        for line_id, buses in state_dict.items():
            for bus_id, history in buses.items():
                if len(history) == 0:
                    continue

                if len(history) == 1:
                    if action_dict[line_id][bus_id] is None:
                        last_action = last_action_history[line_id][bus_id]
                        state_vec = np.concatenate([history[0], last_action])
                        if args.use_state_norm:
                            state_vec = sac_trainer.state_norm(state_vec)
                        sv_tensor = torch.from_numpy(state_vec).float()
                        states_for_metrics.append(sv_tensor)
                        action = sac_trainer.policy_net.get_action(sv_tensor, deterministic=True)
                        action_dict[line_id][bus_id] = action

                elif len(history) >= 2:
                    # ── Lines 930-940: Build state vectors ──
                    prev_action = last_action_history[line_id][bus_id]
                    current_action = action_dict[line_id][bus_id]
                    if current_action is None:
                        current_action = np.zeros(action_dim, dtype=np.float32)
                    elif np.isscalar(current_action):
                        current_action = np.array([current_action], dtype=np.float32)
                    else:
                        current_action = np.array(current_action, dtype=np.float32).reshape(-1)

                    # ── Line 944: Only count reward when station changes ──
                    if history[0][station_feature_idx] != history[1][station_feature_idx]:
                        current_reward = get_reward_value(reward_dict, line_id, bus_id)
                        episode_steps += 1
                        step += 1
                        episode_reward += current_reward  # ← THE KEY LINE

                    # ── Line 986: Truncate state_dict ──
                    state_dict[line_id][bus_id] = history[1:]

                    # ── Line 987: Build next state from truncated state ──
                    state_vec_next = np.concatenate([state_dict[line_id][bus_id][0], current_action])

                    # ── Line 990: Update action history ──
                    last_action_history[line_id][bus_id] = current_action

                    # ── Lines 991-994: Compute next action ──
                    if args.use_state_norm:
                        state_vec_next = sac_trainer.state_norm(state_vec_next)
                    sv_tensor = torch.from_numpy(state_vec_next).float()
                    states_for_metrics.append(sv_tensor)
                    action = sac_trainer.policy_net.get_action(sv_tensor, deterministic=True)
                    action_dict[line_id][bus_id] = action

        # ── Lines 1011-1068: Map actions to env format (deepcopy + residual control) ──
        env_action_dict = copy.deepcopy(action_dict)
        for line_id, buses in env_action_dict.items():
            for bus_id, action_val in buses.items():
                if action_val is not None:
                    if args.use_residual_control:
                        a_hold = action_val[0]
                        a_speed = action_val[1]

                        # Holding: Continuous [0, 120]
                        hold = np.clip((a_hold + 1.0) * 60.0, 0.0, 120.0)

                        # Speed: Deterministic selection for evaluation
                        if a_speed > 0.6:
                            speed = 1.2
                        elif a_speed > 0.2:
                            speed = 1.1
                        elif a_speed > -0.2:
                            speed = 1.0
                        elif a_speed > -0.6:
                            speed = 0.9
                        else:
                            speed = 0.8

                        env_action_dict[line_id][bus_id] = [hold, speed]

        # ── Line 1082: Step environment ──
        state_dict, reward_dict, done, _ = safe_step(env, env_action_dict, render=False)

    # Collect auxiliary metrics
    ep_metrics = collect_eval_metrics(sac_trainer, states_for_metrics)

    return {
        'reward': episode_reward,
        'steps': episode_steps,
        'reg_norm1': ep_metrics['reg_norm1'],
        'reg_norm2': ep_metrics['reg_norm2'],
        'q_value': ep_metrics['q_value_mean'],
        'q_std': ep_metrics['q_std_mean'],
        'ood_loss': ep_metrics['ood_loss'],
        'log_prob': ep_metrics['log_prob_mean'],
    }


def main():
    worker_id = int(os.environ['WORKER_ID'])
    episodes  = json.loads(os.environ['EPISODES'])
    model_dir = os.environ['MODEL_DIR']
    out_file  = os.environ['OUT_FILE']

    results = {}
    for ep in episodes:
        r = eval_episode(ep, model_dir)
        if r is None:
            print(f"W{worker_id} SKIP ep {ep}: no checkpoint", flush=True)
            results[ep] = {k: float('nan') for k in
                ['reward','steps','reg_norm1','reg_norm2','q_value','q_std','ood_loss','log_prob']}
            continue

        print(f"W{worker_id} ep {ep}: reward={r['reward']:.2f} steps={r['steps']} "
              f"q={r['q_value']:.2f} reg1={r['reg_norm1']:.4f}", flush=True)
        results[ep] = r

    env.close()
    with open(out_file, 'w') as f:
        json.dump(results, f)
    print(f"W{worker_id} DONE. Saved to {out_file}", flush=True)

if __name__ == '__main__':
    main()
