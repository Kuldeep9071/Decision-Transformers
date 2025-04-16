import gymnasium as gym
import numpy as np
import torch
import wandb
import minari
import argparse
import pickle
import random
import sys
import matplotlib.pyplot as plt
import os

from decision_transformer.evaluation.evaluate_episodes import evaluate_episode, evaluate_episode_rtg
from decision_transformer.models.decision_transformer import DecisionTransformer
from decision_transformer.models.mlp_bc import MLPBCModel
from decision_transformer.training.act_trainer import ActTrainer
from decision_transformer.training.seq_trainer import SequenceTrainer
import imageio

REF_MIN_SCORE = {
    'halfcheetah': -79.20,
    'hopper': 395.64,
    'walker2d': 0.20,
}

REF_MAX_SCORE = {
    'halfcheetah': 16584.93,
    'hopper': 4376.33,
    'walker2d': 6972.80,
}

def get_normalised_returns(return_score, min_score, max_score):
    # Normalize the returns to be between 0 and 1
    norm_return = (return_score - min_score) / (max_score - min_score)
    return max(0,norm_return * 100)  # Scale to 100


def discount_cumsum(x, gamma):
    discount_cumsum = np.zeros_like(x)
    discount_cumsum[-1] = x[-1]
    for t in reversed(range(x.shape[0]-1)):
        discount_cumsum[t] = x[t] + gamma * discount_cumsum[t+1]
    return discount_cumsum


def experiment(exp_prefix,variant,):
    device = variant.get('device', 'cuda:0')  # Force GPU
    log_to_wandb = variant.get('log_to_wandb', False)

    env_name, dataset = variant['env'], variant['dataset']
    model_type = variant['model_type']
    group_name = f'{exp_prefix}-{env_name}-{dataset}'
    exp_prefix = f'{group_name}-{random.randint(int(1e5), int(1e6) - 1)}'

    if env_name == 'hopper':
        env = gym.make('Hopper-v5', render_mode='rgb_array')
        max_ep_len = 1000
        env_targets = [3600, 1800]  # evaluation conditioning targets
        scale = 1000.  # normalization for rewards/returns
    elif env_name == 'halfcheetah':
        env = gym.make('HalfCheetah-v5', render_mode='rgb_array')
        max_ep_len = 1000
        env_targets = [13000]
        scale = 1000.
    elif env_name == 'walker2d':
        env = gym.make('Walker2d-v5', render_mode='rgb_array')
        max_ep_len = 1000
        env_targets = [5000, 2500]
        scale = 1000.
    elif env_name == 'reacher2d':
        from decision_transformer.envs.reacher_2d import Reacher2dEnv
        env = Reacher2dEnv()
        max_ep_len = 200
        env_targets = [76, 40]
        scale = 10.
    else:
        raise NotImplementedError

    if model_type == 'bc':
        env_targets = env_targets[:1]  # since BC ignores target, no need for different evaluations

    state_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    # load dataset
    BASE_DIR = os.path.join(os.path.dirname(__file__), 'data')
    if not env_name.startswith("mujoco_"):
        if env == 'reacher2d':
            env_name = 'reacher'
        else:
            env_name = f"mujoco_{env_name}"
    print(f"Loading dataset for {env_name} with dataset {dataset}")
    dataset_path = f"{BASE_DIR}/{env_name}_{dataset}-v0.pkl"
    with open(dataset_path, 'rb') as f:
        trajectories = pickle.load(f)
        
    
    # save all path information into separate lists
    # trajectories is a list of dictionaries
    # Each dictionary contains observations, actions, rewards, terminals and terminations
    
    mode = variant.get('mode', 'normal')
    states, traj_lens, returns = [], [], []
    for path in trajectories:
        if mode == 'delayed':  # delayed: all rewards moved to end of trajectory
            path['rewards'][-1] = path['rewards'].sum()
            path['rewards'][:-1] = 0.
        states.append(path['observations'])
        traj_lens.append(len(path['observations']))
        returns.append(path['rewards'].sum())
    traj_lens, returns = np.array(traj_lens), np.array(returns)
    

    # used for input normalization
    states = np.concatenate(states, axis=0)
    state_mean, state_std = np.mean(states, axis=0), np.std(states, axis=0) + 1e-6

    num_timesteps = sum(traj_lens)

    print('=' * 50)
    print(f'Starting new experiment: {env_name} {dataset}')
    print(f'{len(traj_lens)} trajectories, {num_timesteps} timesteps found')
    print(f'Average return: {np.mean(returns):.2f}, std: {np.std(returns):.2f}')
    print(f'Max return: {np.max(returns):.2f}, min: {np.min(returns):.2f}')
    env_name = env_name.replace('mujoco_', '')
   
    norm_returns = get_normalised_returns(np.mean(returns), REF_MIN_SCORE[env_name], REF_MAX_SCORE[env_name])
    print(f'Normalised return: {norm_returns:.2f}')
    print(f'Normalised max return: {get_normalised_returns(np.max(returns), REF_MIN_SCORE[env_name], REF_MAX_SCORE[env_name]):.2f}')
    print(f'Normalised min return: {get_normalised_returns(np.min(returns), REF_MIN_SCORE[env_name], REF_MAX_SCORE[env_name]):.2f}')
    print(f'Normalised std: {get_normalised_returns(np.std(returns), REF_MIN_SCORE[env_name], REF_MAX_SCORE[env_name]):.2f}')
    print('=' * 50)
    

    K = variant['K']
    batch_size = variant['batch_size']
    num_eval_episodes = variant['num_eval_episodes']
    pct_traj = variant.get('pct_traj', 1.)

    # only train on top pct_traj trajectories (for %BC experiment)
    num_timesteps = max(int(pct_traj*num_timesteps), 1)
    sorted_inds = np.argsort(returns)  # lowest to highest
    num_trajectories = 1
    timesteps = traj_lens[sorted_inds[-1]]
    ind = len(trajectories) - 2
    while ind >= 0 and timesteps + traj_lens[sorted_inds[ind]] <= num_timesteps:
        timesteps += traj_lens[sorted_inds[ind]]
        num_trajectories += 1
        ind -= 1
    sorted_inds = sorted_inds[-num_trajectories:]

    # used to reweight sampling so we sample according to timesteps instead of trajectories
    p_sample = traj_lens[sorted_inds] / sum(traj_lens[sorted_inds])

    def get_batch(batch_size=256, max_len=K):
        batch_inds = np.random.choice(
        np.arange(num_trajectories),
        size=batch_size,
        replace=True,
        p=p_sample
                    )

        s, a, r, d, rtg, timesteps, mask = [], [], [], [], [], [], []

        for i in range(batch_size):
            traj = trajectories[int(sorted_inds[batch_inds[i]])]
            traj_len = traj['rewards'].shape[0]

            si = random.randint(max_len - 1, traj_len - 1)  # ensure enough past context

            # ===== STATE WINDOW =====
            s_window = traj['observations'][si - max_len + 1: si + 1]
            pad_len = max_len - s_window.shape[0]
            if pad_len > 0:
                s_window = np.concatenate([np.zeros((pad_len, state_dim)), s_window], axis=0)
            s_window = (s_window - state_mean) / state_std
            s.append(s_window.reshape(1, max_len, state_dim))

            # ===== ACTION TARGET =====
            a_t = traj['actions'][si].reshape(1, 1, act_dim)
            a.append(a_t)

            # ===== REWARD WINDOW =====
            r_window = traj['rewards'][si - max_len + 1: si + 1].reshape(-1, 1)
            if pad_len > 0:
                r_window = np.concatenate([np.zeros((pad_len, 1)), r_window], axis=0)
            r.append(r_window.reshape(1, max_len, 1))

            # ===== DONE/TERMINAL =====
            done_key = 'dones' if 'dones' in traj else ('terminals' if 'terminals' in traj else 'terminations')
            d_window = traj[done_key][si - max_len + 1: si + 1]
            if pad_len > 0:
                d_window = np.concatenate([np.ones((pad_len,)) * 2, d_window], axis=0)
            d.append(d_window.reshape(1, max_len))

            # ===== TIMESTEPS =====
            ts_window = np.arange(si - max_len + 1, si + 1)
            ts_window[ts_window < 0] = 0
            ts_window[ts_window >= max_ep_len] = max_ep_len - 1
            timesteps.append(ts_window.reshape(1, max_len))

            # ===== RTG =====
            rtg_window = discount_cumsum(traj['rewards'][si:], gamma=1.)[:max_len].reshape(1, -1, 1)
            if rtg_window.shape[1] < max_len:
                # Pad with zeros if the RTG sequence is shorter than max_len
                padding_size = max_len - rtg_window.shape[1]
                rtg_window = np.concatenate([rtg_window, np.zeros((1, padding_size, 1))], axis=1)
            rtg.append(rtg_window / scale)

            # ===== MASK =====
            m = np.concatenate([np.zeros((1, pad_len)), np.ones((1, max_len - pad_len))], axis=1)
            mask.append(m)

        # === STACK BATCHES ===
        s = torch.from_numpy(np.concatenate(s, axis=0)).to(dtype=torch.float32, device=device)
        a = torch.from_numpy(np.concatenate(a, axis=0)).to(dtype=torch.float32, device=device)
        r = torch.from_numpy(np.concatenate(r, axis=0)).to(dtype=torch.float32, device=device)
        d = torch.from_numpy(np.concatenate(d, axis=0)).to(dtype=torch.long, device=device)
        rtg = torch.from_numpy(np.concatenate(rtg, axis=0)).to(dtype=torch.float32, device=device)
        timesteps = torch.from_numpy(np.concatenate(timesteps, axis=0)).to(dtype=torch.long, device=device)
        mask = torch.from_numpy(np.concatenate(mask, axis=0)).to(dtype=torch.float32, device=device)

        with open("debug_output.txt", "w") as f:
            f.write("Sample state: " + str(s[0, -1].cpu().numpy()) + "\n")
            f.write("Sample action: " + str(a[0, 0].cpu().numpy()) + "\n")
            f.write("Sample mask: " + str(mask[0].cpu().numpy()) + "\n")
            f.write("Sample RTG: " + str(rtg[0].cpu().numpy()) + "\n")

        return s, a, r, d, rtg, timesteps, mask

    def eval_episodes(target_rew):
        def fn(model):
            returns, lengths = [], []
            for episode_num in range(num_eval_episodes):
                with torch.no_grad():
                    if model_type == 'dt':
                        ret, length = evaluate_episode_rtg(
                            env,
                            state_dim,
                            act_dim,
                            model,
                            max_ep_len=max_ep_len,
                            scale=scale,
                            target_return=target_rew/scale,
                            mode=mode,
                            state_mean=state_mean,
                            state_std=state_std,
                            device=device,
                            render=True,
                            save_video=True,
                            video_folder=f"videos/target_{target_rew}",
                            video_name=f"hopper_episode{episode_num}"
                        )
                    else:
                        ret, length = evaluate_episode(
                            env,
                            state_dim,
                            act_dim,
                            model,
                            max_ep_len=max_ep_len,
                            target_return=target_rew/scale,
                            mode=mode,
                            state_mean=state_mean,
                            state_std=state_std,
                            device=device,
                        )
                    returns.append(ret)
                    lengths.append(length)
            
            
            return {
                f'target_{target_rew}_return_mean': np.mean(returns),
                f'target_{target_rew}_return_std': np.std(returns)  
            }
        return fn

    if model_type == 'dt':
        model = DecisionTransformer(
            state_dim=state_dim,
            act_dim=act_dim,
            max_length=K,
            max_ep_len=max_ep_len,
            hidden_size=variant['embed_dim'],
            n_layer=variant['n_layer'],
            n_head=variant['n_head'],
            n_inner=4*variant['embed_dim'],
            activation_function=variant['activation_function'],
            n_positions=1024,
            resid_pdrop=variant['dropout'],
            attn_pdrop=variant['dropout'],
        )
    elif model_type == 'bc':
        model = MLPBCModel(
            state_dim=state_dim,
            act_dim=act_dim,
            max_length=K,
            hidden_size=variant['embed_dim'],
            n_layer=variant['n_layer'],
        )
    else:
        raise NotImplementedError

    model = model.to(device)

    torch.nn.utils.clip_grad_norm_(model.parameters(), 0.25)

    warmup_steps = variant['warmup_steps']
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=variant['learning_rate'],
        weight_decay=variant['weight_decay'],
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda steps: min((steps+1)/warmup_steps, 1)
    )

    if model_type == 'dt':
        trainer = SequenceTrainer(
            model=model,
            optimizer=optimizer,
            batch_size=batch_size,
            get_batch=get_batch,
            scheduler=scheduler,
            loss_fn=lambda s_hat, a_hat, r_hat, s, a, r: torch.mean((a_hat - a)**2),
            eval_fns=[eval_episodes(tar) for tar in env_targets],
        )
    elif model_type == 'bc':
        trainer = ActTrainer(
            model=model,
            optimizer=optimizer,
            batch_size=batch_size,
            get_batch=get_batch,
            scheduler=scheduler,
            loss_fn=lambda s_hat, a_hat, r_hat, s, a, r: torch.mean((a_hat - a)**2),
            eval_fns=[eval_episodes(tar) for tar in env_targets],
        )

    if log_to_wandb:
        wandb.init(
            name=exp_prefix,
            group=group_name,
            project='decision-transformer',
            config=variant
        )
        # wandb.watch(model)  # wandb has some bug

    for iter in range(variant['max_iters']):
        outputs = trainer.train_iteration(num_steps=variant['num_steps_per_iter'], iter_num=iter+1, print_logs=True)
        if log_to_wandb:
            wandb.log(outputs)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='hopper')
    parser.add_argument('--dataset', type=str, default='medium')  # medium, medium-replay, medium-expert, expert
    parser.add_argument('--mode', type=str, default='normal')  # normal for standard setting, delayed for sparse
    parser.add_argument('--K', type=int, default=40)
    parser.add_argument('--pct_traj', type=float, default=1.)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--model_type', type=str, default='dt')  # dt for decision transformer, bc for behavior cloning
    parser.add_argument('--embed_dim', type=int, default=256)  # Increased from 128
    parser.add_argument('--n_layer', type=int, default=4)  # Increased from 3
    parser.add_argument('--n_head', type=int, default=4)  # Increased from 1
    parser.add_argument('--activation_function', type=str, default='relu')
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--learning_rate', '-lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', '-wd', type=float, default=1e-4)
    parser.add_argument('--warmup_steps', type=int, default=1000)
    parser.add_argument('--num_eval_episodes', type=int, default=100) # Default = 100
    parser.add_argument('--max_iters', type=int, default=100)
    parser.add_argument('--num_steps_per_iter', type=int, default=2000)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--log_to_wandb', '-w', type=bool, default=False)
    
    args = parser.parse_args()
    experiment('gym-experiment', variant=vars(args))
