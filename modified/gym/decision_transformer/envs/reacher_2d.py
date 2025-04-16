import numpy as np
import gymnasium as gym
import os
import mujoco

class Reacher2dEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 50}
    
    
    def __init__(self, render_mode=None):
        super().__init__()
        self.name = 'reacher2d'
        curr_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(curr_dir, 'assets', 'reacher_2d.xml')
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)
        
        self.fingertip_sid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, 'fingertip')
        self.target_bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, 'target')
        
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(self.model.nu,), dtype=np.float32)
        self.frame_skip = 5
        self.dt = self.model.opt.timestep * self.frame_skip
        obs_dim = len(self._get_obs())
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)
        
        self.render_mode = render_mode
        self.viewer = None
        
        
    def step(self, action):
        action = np.clip(action, -1.0, 1.0)
        self.data.ctrl[:] = action
        for _ in range(self.frame_skip):
            mujoco.mj_step(self.model, self.data, nstep=1)

        tip  = self.data.site_xpos[self.fingertip_sid][:2]
        tar = self.data.xpos[self.target_bid][:2]

        dist = np.sum(np.abs(tip - tar))
        
        reward_dist = 0.0 
        reward_ctrl = 0.0
        reward_bonus = 1.0 if dist < 0.1 else 0.0
        reward = reward_bonus + reward_ctrl + reward_dist
        
        terminated = False
        truncated = False
        obs = self._get_obs()
        
        info = {
            "reward_dist": reward_dist,
            "reward_ctrl": reward_ctrl,
            "reward_bonus": reward_bonus,
            "goal": self.goal,
        }
        return obs, reward, terminated, truncated, info

    def _get_obs(self):
        theta = self.data.qpos[:2].ravel()  # Extract only the first two elements (joint angles)
        qvel = self.data.qvel.ravel()  # Angular velocities
        tar = self.data.xpos[self.target_bid][:2]  # Get target position correctly
        tip = self.data.site_xpos[self.fingertip_sid][:2]  # Fingertip position
        xpos = tip - tar  # Vector from fingertip to targe
        
        return np.concatenate([
        np.cos(theta),
        np.sin(theta),
        tar,  # Target coordinates
        qvel,
        xpos
        ])

    def reset(self, seed = None):
        super().reset(seed = seed)
        
        qpos = self.np_random.uniform(low=-2.0, high=2.0, size=self.model.nq)
        qvel = np.zeros_like(self.data.qvel)
        while True:
            self.goal = self.np_random.uniform(low=-1.5, high=1.5, size=2)
            if 0.5 <= np.linalg.norm(self.goal) <= 1.0:
                break
        
        
        mujoco.mj_resetData(self.model, self.data)
        self.model.body_pos[self.target_bid][:2] = self.goal
        mujoco.mj_forward(self.model, self.data)
        return self._get_obs(), {}
    
    def render(self):
        if self.viewer is None:
            self.viewer = mujoco.MjViewer(self.data)
            self.viewer.cam.distance = self.model.stat.extent * 2.0
            self.viewer.cam.elevation = -20
            self.viewer.cam.azimuth = 90
        
        mujoco.mj_step(self.data)
        self.viewer.render()
        return self.viewer.is_alive()
    
    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None
