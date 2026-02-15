#!/usr/bin/env python3
"""
Data collection script using random sampling.
This is a backward-compatible wrapper around collect_data.py.
"""

from collect_data import collect_trajectory_data

if __name__ == "__main__":
    ENV_ID = 'MiniGrid-Empty-Interact-6x6-o6-v0'  # o6 - o14 are the different objects in the environment
    
    collect_trajectory_data(
        env_name=ENV_ID,
        num_seeds=100,
        steps_per_seed=50,
        save_dir="data",
        fullobs=True,
        policy=None,
        filename_suffix="train_5000"
    )