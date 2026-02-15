"""
Data collection script using expert policies.
This is a backward-compatible wrapper around collect_data.py.
"""

from collect_data import collect_trajectory_data
from policies import MultiTaskPolicy


def collect_data_with_tensor_policy(env_name, num_seeds=50, steps_per_seed=20, save_dir="data", fullobs=True, policy=None):
    """
    Collect (s, a, s', r, done) data using a specific policy for training.
    Similar structure to collect_trajectory_data, but uses policy to select actions instead of random sampling.
    
    Args:
        env_name: Environment name
        num_seeds: Number of seeds to collect
        steps_per_seed: Number of steps per seed
        save_dir: Save directory
        fullobs: Whether to use fully observable wrapper
        policy: Policy object
    """
    if policy is None:
        raise ValueError("Policy is required")
    
    policy_name = policy.__class__.__name__
    filename_suffix = f"{policy_name}_policy_object_matching"
    
    return collect_trajectory_data(
        env_name=env_name,
        num_seeds=num_seeds,
        steps_per_seed=steps_per_seed,
        save_dir=save_dir,
        fullobs=fullobs,
        policy=policy,
        seed_offset=10000,
        filename_suffix=filename_suffix
    )


if __name__ == "__main__":
    env_name = "MiniGrid-Empty-Interact-6x6-o3-v0"
    save_dir = "data"

    multi_policy = MultiTaskPolicy()

    collect_data_with_tensor_policy(
        env_name=env_name, 
        num_seeds=10, 
        steps_per_seed=50, 
        save_dir=save_dir, 
        fullobs=True, 
        policy=multi_policy
    )
