"""
Unified data collection script for MiniGrid environments.
Supports both random sampling and policy-based sampling.
"""

import numpy as np
import torch
import os
import sys
from tqdm import tqdm
import gym
from gym_minigrid.minigrid import MiniGridEnv
from gym_minigrid.wrappers import FullyObsWrapper

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)

from env_utils import Environment, Minigrid2Image
from policies import BaseExpertPolicy


def _get_core_env(env: Environment):
    """Return the underlying Minigrid environment for accessing internal states."""
    return env.gym_env.unwrapped


def _get_interact_allowed_actions(env: Environment):
    """Generate allowed actions for InteractEmpty environments based on front object and carrying state."""
    core_env = _get_core_env(env)
    grid = core_env.grid
    fwd_pos = tuple(core_env.front_pos)
    in_bounds = (
        0 <= fwd_pos[0] < grid.width and
        0 <= fwd_pos[1] < grid.height
    )
    fwd_cell = grid.get(*fwd_pos) if in_bounds else None
    has_front_obj = fwd_cell is not None
    front_pickable = has_front_obj and fwd_cell.can_pickup()
    carrying = core_env.carrying
    has_carry = carrying is not None
    toggle_whitelist = ['box', 'key', 'ball']
    front_toggleable = has_front_obj and (fwd_cell.type in toggle_whitelist)

    allowed = [
        int(MiniGridEnv.Actions.left),
        int(MiniGridEnv.Actions.right)
    ]
    if in_bounds and fwd_cell is None:
        allowed.append(int(MiniGridEnv.Actions.forward))

    if front_pickable and not has_carry:
        allowed.append(int(MiniGridEnv.Actions.pickup))

    if in_bounds and fwd_cell is None and has_carry:
        allowed.append(int(MiniGridEnv.Actions.drop))

    if front_toggleable:
        allowed.append(int(MiniGridEnv.Actions.toggle))

    if front_pickable and has_carry:
        allowed.append(int(MiniGridEnv.Actions.done))

    seen = set()
    filtered = []
    for act in allowed:
        if act not in seen:
            filtered.append(act)
            seen.add(act)
    return filtered


def _balanced_sample(choices, action_counts):
    """
    Weighted random sampling based on 1/(count+1) to prioritize under-sampled actions.
    Enhanced version: gives higher weights to interaction actions (3/4/5/6) to encourage more interaction data collection.
    """
    counts = np.array([action_counts[c] for c in choices], dtype=np.float64)
    
    base_weights = 1.0 / (counts + 1.0)
    
    interaction_actions = [3, 4, 5, 6]
    base_interaction_bonus = 10.0
    action6_extra_bonus = 100.0
    
    enhanced_weights = base_weights.copy()
    for i, action in enumerate(choices):
        if action in interaction_actions:
            bonus = base_interaction_bonus
            if action == 6:
                bonus = action6_extra_bonus
            enhanced_weights[i] *= bonus
    
    probs = enhanced_weights / enhanced_weights.sum()
    return int(np.random.choice(choices, p=probs))


def collect_trajectory_data(
    env_name, 
    num_seeds=50, 
    steps_per_seed=20, 
    save_dir="data", 
    fullobs=True,
    policy=None,
    seed_offset=0,
    filename_suffix=None
):
    """
    Collect (s, a, s', r, done) data for training Forward/Inverse Model.
    
    Args:
        env_name: Environment name
        num_seeds: Number of seeds to collect
        steps_per_seed: Number of steps per seed
        save_dir: Save directory
        fullobs: Whether to use fully observable wrapper
        policy: Policy object for action selection. If None, uses balanced random sampling.
        seed_offset: Offset for seed values (default: 0 for random, 10000 for policy)
        filename_suffix: Optional suffix for filename
    """
    os.makedirs(save_dir, exist_ok=True)
    
    buffer = {
        'states': [],
        'carried': [],
        'actions': [],
        'next_states': [],
        'next_carried': [],
        'rewards': [],
        'dones': []
    }

    use_policy = policy is not None
    if use_policy:
        print(f"Starting collection with policy for: {env_name}")
    else:
        print(f"Starting collection for: {env_name}")

    gym_env = gym.make(env_name)
    if fullobs:
        gym_env = Minigrid2Image(FullyObsWrapper(gym_env))
    else:
        gym_env = Minigrid2Image(gym_env)

    action_counts = np.zeros(gym_env.action_space.n, dtype=np.int64)
    for seed in tqdm(range(seed_offset, num_seeds + seed_offset), desc="Seeds"):
        env = Environment(gym_env, fix_seed=True, env_seed=seed)
        
        if use_policy and hasattr(policy, 'reset'):
            policy.reset()
            episode_policy = policy
        
        curr_output = env.initial()
        
        curr_frame = curr_output['frame'].cpu().numpy() if isinstance(curr_output['frame'], torch.Tensor) else curr_output['frame']
        curr_carried = np.array([
            curr_output['carried_col'].item(), 
            curr_output['carried_obj'].item()
        ])

        for _ in range(steps_per_seed):
            if use_policy:
                if isinstance(curr_frame, torch.Tensor):
                    curr_frame_np = curr_frame.cpu().numpy()
                else:
                    curr_frame_np = curr_frame
                
                if len(curr_frame_np.shape) == 4 and curr_frame_np.shape[0] == 1:
                    curr_frame_np = curr_frame_np[0]
                
                a = episode_policy.get_action(curr_frame_np.squeeze(0), curr_carried)
            else:
                if 'Interact' in env_name:
                    allowed_actions = _get_interact_allowed_actions(env)
                elif 'Random' in env_name:
                    allowed_actions = [0, 1, 2, 3, 4]
                elif 'KeyCorridor' in env_name:
                    allowed_actions = [0, 1, 2, 3]
                else:
                    allowed_actions = list(range(env.gym_env.action_space.n))

                a = _balanced_sample(allowed_actions, action_counts)
            
            action_counts[a] += 1

            action_tensor = torch.tensor(a)
            next_output = env.step(action_tensor)
            
            next_frame = next_output['frame'].cpu().numpy() if isinstance(next_output['frame'], torch.Tensor) else next_output['frame']
            next_carried = np.array([
                next_output['carried_col'].item(), 
                next_output['carried_obj'].item()
            ])
            
            done = next_output['done']
            reward = next_output['reward']

            done_value = done.item() if isinstance(done, torch.Tensor) else done
            if not done_value:
                buffer['states'].append(curr_frame)
                buffer['carried'].append(curr_carried)
                buffer['actions'].append(a)
                buffer['next_states'].append(next_frame)
                buffer['next_carried'].append(next_carried)
                buffer['rewards'].append(reward.item() if isinstance(reward, torch.Tensor) else reward)
                buffer['dones'].append(done.item() if isinstance(done, torch.Tensor) else done)

                curr_frame = next_frame
                curr_carried = next_carried
            else:
                break

    data_to_save = {k: np.array(v) for k, v in buffer.items()}
    
    print(f"Collection complete.")
    print(f"States shape: {data_to_save['states'].shape}")
    print(f"Actions shape: {data_to_save['actions'].shape}")
    print(f"Action counts: {action_counts}")
    
    if filename_suffix:
        filename = f"{env_name}_seed{num_seeds}_step{steps_per_seed}_{filename_suffix}.npz"
    elif use_policy:
        policy_name = policy.__class__.__name__
        filename = f"{env_name}_seed{num_seeds}_step{steps_per_seed}_{policy_name}_policy.npz"
    else:
        filename = f"{env_name}_seed{num_seeds}_step{steps_per_seed}_random.npz"
    
    save_path = os.path.join(save_dir, filename)
    
    np.savez_compressed(save_path, **data_to_save)
    print(f"Data saved successfully to: {save_path}")
    
    return save_path


if __name__ == "__main__":
    from policies import MultiTaskPolicy
    
    ENV_ID = 'MiniGrid-Empty-Interact-6x6-o3-v0'
    save_dir = "data"

    # Example 1: Random sampling
    collect_trajectory_data(
        env_name=ENV_ID,
        num_seeds=100,
        steps_per_seed=50,
        save_dir=save_dir,
        fullobs=True
    )
    
    # Example 2: Policy-based sampling
    multi_policy = MultiTaskPolicy()
    collect_trajectory_data(
        env_name=ENV_ID,
        num_seeds=250,
        steps_per_seed=50,
        save_dir=save_dir,
        fullobs=True,
        policy=multi_policy,
        seed_offset=10000
    )

