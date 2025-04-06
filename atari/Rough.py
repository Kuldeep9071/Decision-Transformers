import h5py
import os
import numpy as np
import pandas as pd
from tqdm import tqdm

games = ["breakout", "pong", "qbert", "seaquest"]
base_dir = os.path.join(os.getcwd(), "game_datasets")
output_root = "exported_trajectories"

for game in games:
    print(f"\n📦 Processing: {game}")
    dataset_path = os.path.join(base_dir, game, "expert-v0", "data", "main_data.hdf5")
    
    if not os.path.exists(dataset_path):
        print(f"❌ Dataset not found: {dataset_path}")
        continue
        
    with h5py.File(dataset_path, 'r') as f:
        for ep_idx, episode_key in enumerate(tqdm(f.keys(), desc=f"Episodes in {game}")):
            ep = f[episode_key]
            ep_folder = os.path.join(output_root, game, episode_key)
            os.makedirs(ep_folder, exist_ok=True)

            # Save actions
            if 'actions' in ep:
                np.savetxt(os.path.join(ep_folder, "actions.csv"), ep['actions'][:], delimiter=",", fmt='%d')

            # Save rewards
            if 'rewards' in ep:
                np.savetxt(os.path.join(ep_folder, "rewards.csv"), ep['rewards'][:], delimiter=",", fmt='%f')

            # Save dones
            if 'dones' in ep:
                np.savetxt(os.path.join(ep_folder, "dones.csv"), ep['dones'][:], delimiter=",", fmt='%d')

            # Save observations as .npy files for each timestep
            if 'observations' in ep:
                obs_folder = os.path.join(ep_folder, "observations")
                os.makedirs(obs_folder, exist_ok=True)
                observations = ep['observations'][:]
                for t, obs in enumerate(observations):
                    np.save(os.path.join(obs_folder, f"{t:04d}.npy"), obs)

            # Save infos if present
            if 'infos' in ep:
                infos_folder = os.path.join(ep_folder, "infos")
                os.makedirs(infos_folder, exist_ok=True)
                infos = ep['infos']
                for key in infos.keys():
                    data = infos[key][:]
                    # Save each key in infos as a separate .npy or .csv
                    try:
                        np.savetxt(os.path.join(infos_folder, f"{key}.csv"), data.reshape(data.shape[0], -1), delimiter=",")
                    except:
                        np.save(os.path.join(infos_folder, f"{key}.npy"), data)

