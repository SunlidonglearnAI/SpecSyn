"""
T2A (Transformer-to-Action) Wrapper for Terrain Walking Task.
[Single Evolution Mode: Strength (F0) Only]
Optimizes 1 Muscle Parameter:
1. Strength (F0) -> gainprm[2] | Range: [0.8, 1.5]
"""
import numpy as np
import mujoco
import gymnasium as gym
from gymnasium import spaces
import matplotlib.pyplot as plt
import seaborn as sns
import os

from myosuite.envs.myo.myobase.walk_v0 import WalkEnvV0, TerrainEnvV0

class T2AWalkMixin:
    """
    Mixin for Evolution of Muscle Strength (F0) only.
    """
    def _setup_t2a(self, **kwargs):
        # 1. [T2A Config]
        obs_keys = kwargs.get('obs_keys', self.DEFAULT_OBS_KEYS)
        if "design_params" not in obs_keys:
            if isinstance(obs_keys, tuple): obs_keys = list(obs_keys)
            obs_keys.append("design_params")
        kwargs['obs_keys'] = obs_keys

        # 2. [Identify Muscles]
        self.muscle_indices = np.where(self.sim.model.actuator_dyntype == mujoco.mjtDyn.mjDYN_MUSCLE)[0]
        if len(self.muscle_indices) == 0:
            self.muscle_indices = np.arange(self.sim.model.nu)
            
        self.num_muscles = len(self.muscle_indices)
        
        # Robust Name Getter
        self.muscle_names = []
        for i in self.muscle_indices:
            try: name = self.sim.model.actuator_id2name(int(i))
            except: name = f"M{i}"
            self.muscle_names.append(name)

        # 3. [Backup Original Parameters]
        self.original_gainprm = self.sim.model.actuator_gainprm[self.muscle_indices].copy()

        # 4. [Design Phase Config]
        self.design_steps = 1          
        self.design_step_counter = 0   
        
        # === 修改点 A: 维度改为 1 (每块肌肉仅1个参数: Strength) ===
        self.design_dim = self.num_muscles * 1 
        self.control_dim = self.sim.model.nu 
        
        # Init Scales (现在只需存储 F_scale)
        self.current_scales = np.ones((self.num_muscles, 1), dtype=np.float32)
        
        print(f"DEBUG: T2A Walk Setup - Mode: Strength (F0) Only. Dim: {self.design_dim}")
        return kwargs

    def _fix_action_space(self):
        dim = max(self.control_dim, self.design_dim)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(dim,), dtype=np.float32)

    def _apply_design(self, design_act):
        """
        Map Policy Actions to Physical Parameters.
        Action range [-1, 1] -> Strength [0.8, 1.5]
        """
        design_act = np.clip(design_act, -1.0, 1.0)
        
        # === 解析参数 (Num_Muscles,) ===
        # 这里的 design_params 每一行对应一块肌肉的 F0 动作
        design_params = design_act[:self.num_muscles]
        
        # 映射公式: 0->1.0, 1->1.5, -1->0.8
        f_scale = np.where(
            design_params > 0,
            1.0 + 0.5 * design_params,
            1.0 + 0.2 * design_params
        )
        
        # 修改 MuJoCo 模型中的 F0 (gainprm 索引为 2)
        self.sim.model.actuator_gainprm[self.muscle_indices, 2] = self.original_gainprm[:, 2] * f_scale
        
        # 存储用于 Observation
        self.current_scales = f_scale.reshape(-1, 1)

    def reset_t2a(self):
        self.sim.model.actuator_gainprm[self.muscle_indices] = self.original_gainprm.copy()
        self.current_scales = np.ones((self.num_muscles, 1), dtype=np.float32)
        self.design_step_counter = 0

    def get_obs_t2a(self, obs_dict):
        obs_dict["design_params"] = self.current_scales.flatten().copy()
        return obs_dict

    def step_t2a(self, action, parent_step_func):
        current_action_dim = action.shape[0]
        
        if self.design_step_counter < self.design_steps:
            design_act = action[:self.design_dim] if current_action_dim >= self.design_dim else np.zeros(self.design_dim)
            self._apply_design(design_act)
            self.design_step_counter += 1
            return self.get_obs(), 0.0, False, False, {"phase": "design"} 

        else:
            control_act = action[:self.control_dim] if current_action_dim >= self.control_dim else np.zeros(self.control_dim)
            return parent_step_func(control_act)

    def visualize_evolution(self, save_path="walk_strength_evolution.png", show=False):
        if self.current_scales is None: return
        plt.figure(figsize=(6, 12)) 
        sns.heatmap(self.current_scales, 
                    yticklabels=self.muscle_names, 
                    xticklabels=["Strength (F0)"],
                    cmap="YlOrRd", center=1.0, annot=True, fmt=".2f") 
        plt.title("Muscle Strength Evolution")
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        if show: plt.show()
        plt.close()

# 环境类保持不变 (因为它们继承自上面的 Mixin)
class WalkEnvT2A(T2AWalkMixin, WalkEnvV0):
    def _setup(self, **kwargs):
        kwargs = self._setup_t2a(**kwargs)
        super()._setup(**kwargs)
        self._fix_action_space()
    def reset(self, **kwargs):
        self.reset_t2a()
        return super().reset(**kwargs)
    def get_obs_dict(self, sim):
        return self.get_obs_t2a(super().get_obs_dict(sim))
    def step(self, action):
        return self.step_t2a(action, super().step)

class TerrainEnvT2A(T2AWalkMixin, TerrainEnvV0):
    def _setup(self, **kwargs):
        kwargs = self._setup_t2a(**kwargs)
        super()._setup(**kwargs)
        self._fix_action_space()
    def reset(self, **kwargs):
        self.reset_t2a()
        return super().reset(**kwargs)
    def get_obs_dict(self, sim):
        return self.get_obs_t2a(super().get_obs_dict(sim))
    def step(self, action):
        return self.step_t2a(action, super().step)