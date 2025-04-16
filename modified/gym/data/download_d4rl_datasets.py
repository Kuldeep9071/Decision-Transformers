import gymnasium as gym
import numpy as np
import collections
import pickle
import minari

# Define dataset names
minari_datasets = {
    'halfcheetah': ['mujoco/halfcheetah/medium-v0', 'mujoco/halfcheetah/simple-v0', 'mujoco/halfcheetah/expert-v0'],
    'hopper': ['mujoco/hopper/medium-v0', 'mujoco/hopper/simple-v0', 'mujoco/hopper/expert-v0'],
    'walker2d': ['mujoco/walker2d/medium-v0', 'mujoco/walker2d/simple-v0', 'mujoco/walker2d/expert-v0'],
    'reacher2d': ['mujoco/reacher/medium-v0', 'mujoco/reacher/expert-v0', 'mujoco/reacher/simple-v0']
}

# Placeholder for pickle data saving
all_env_data = {}

def process_and_save_dataset(env_name, dataset_list):
    all_returns = []  # to compute ref_min and ref_max

    # First pass: calculate episode returns across all datasets in this env
    for dataset_id in dataset_list:
        dataset = minari.load_dataset(dataset_id, download=True)

        for episode in dataset.iterate_episodes():
            rewards = episode.rewards
            total_return = np.sum(rewards)
            all_returns.append(total_return)

    ref_min = min(all_returns)
    ref_max = max(all_returns)
    print(f"Reference scores for {env_name} — min: {ref_min:.2f}, max: {ref_max:.2f}")

    # List to hold episode data for all datasets in this environment
    env_paths = []

    # Second pass: save episode data and normalize
    for dataset_id in dataset_list:
        dataset = minari.load_dataset(dataset_id, download=True)
        paths = []

        for episode in dataset.iterate_episodes():
            data_ = collections.defaultdict(list)
            N = len(episode.rewards)

            for i in range(N):
                done_bool = episode.terminations[i] or episode.truncations[i]

                for k in ['observations', 'actions', 'rewards', 'terminations', 'truncations']:
                    data_[k].append(getattr(episode, k)[i])

                if done_bool:
                    episode_data = {k: np.array(v) for k, v in data_.items()}
                    paths.append(episode_data)
                    data_ = collections.defaultdict(list)

        print(f"Collected {len(paths)} episodes from {dataset_id}")
        returns = np.array([np.sum(p['rewards']) for p in paths])
        normalised_returns = (returns - ref_min) / (ref_max - ref_min)
        print(f"Normalised returns: mean = {np.mean(normalised_returns):.4f}, std = {np.std(normalised_returns):.4f}, max = {np.max(normalised_returns):.4f}, min = {np.min(normalised_returns):.4f}")
        num_samples = np.sum([p['rewards'].shape[0] for p in paths])
        print(f'Number of samples collected: {num_samples}')
        print(f'Trajectory returns: mean = {np.mean(returns):.2f}, std = {np.std(returns):.2f}, max = {np.max(returns):.2f}, min = {np.min(returns):.2f}')
        
        # Save processed episodes for this dataset into the environment's list
        env_paths.extend(paths)
    
    # Optional: Save a dictionary that also stores the reference min and max scores
    env_data = {
        'env_name': env_name,
        'ref_min': ref_min,
        'ref_max': ref_max,
        'paths': env_paths
    }
    
    # Save pickle file for this environment
    pickle_file = f"{env_name}_dataset.pkl"
    with open(pickle_file, "wb") as f:
        pickle.dump(env_data, f)
    print(f"Saved pickle file: {pickle_file}")
    
    # Also return data for optional aggregation across environments
    return env_data

# Process each environment and save data
for env_name, dataset_list in minari_datasets.items():
    env_data = process_and_save_dataset(env_name, dataset_list)
    all_env_data[env_name] = env_data

# Optionally, save the aggregated data for all environments
with open("all_env_data.pkl", "wb") as f:
    pickle.dump(all_env_data, f)
print("Saved aggregated pickle file: all_env_data.pkl")
