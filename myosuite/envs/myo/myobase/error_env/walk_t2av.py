"""
T2A (Transformer-to-Action) Wrapper for Terrain Walking Task.
[Velocity-Only Evolution Mode]
Focus: Isolating the effect of Muscle Contraction Velocity (Vmax).
Target Parameter: gainprm[6] (Vmax - Max Contraction Velocity)
Constraint: Strength (F0) and Compliance/Stiffness are FIXED.
Range: [0.5, 1.5] (+/- 50%) - Velocity is safer to vary than length.

Physical Meaning:
- Higher Vmax (>1.0): "Fast Twitch" muscles. Explosive, high power, but maybe less precise.
- Lower Vmax (<1.0): "Slow Twitch" muscles. Slower, smoother, potentially more stable.

Place this file in: myosuite/envs/myo/myobase/walk_t2a.py
"""
import numpy as np
import mujoco
import gymnasium as gym
from gymnasium import spaces
import matplotlib.pyplot as plt
import seaborn as sns
import os

# 导入原始环境
from myosuite.envs.myo.myobase.walk_v0 import WalkEnvV0, TerrainEnvV0

# ===========================================================================
#  T2A Mixin: 仅速度进化逻辑
# ===========================================================================
class T2AWalkMixin:
    """
    Mixin for Velocity-Only Evolution.
    """
    def _setup_t2a(self, **kwargs):
        # 1. [T2A Config]
        obs_keys = kwargs.get('obs_keys', self.DEFAULT_OBS_KEYS)
        if "design_params" not in obs_keys:
            if isinstance(obs_keys, tuple):
                obs_keys = list(obs_keys)
            obs_keys.append("design_params")
        kwargs['obs_keys'] = obs_keys

        # 2. [Identify Muscles]
        self.muscle_indices = np.where(self.sim.model.actuator_dyntype == mujoco.mjtDyn.mjDYN_MUSCLE)[0]
        if len(self.muscle_indices) == 0:
            self.muscle_indices = np.arange(self.sim.model.nu)
            
        self.num_muscles = len(self.muscle_indices)
        
        # === [ROBUST NAME GETTER] ===
        self.muscle_names = []
        for i in self.muscle_indices:
            name = None
            try:
                name = self.sim.model.actuator_id2name(int(i))
            except:
                pass
            if name is None:
                name = f"M{i}"
            self.muscle_names.append(name)

        # 3. [Backup Original Parameters]
        # 备份 Vmax (位于 index 6)
        # 注意：某些模型 Vmax 可能在 bias 中也有体现，但主要控制收缩速度的是 gainprm[6]
        self.original_vmax = self.sim.model.actuator_gainprm[self.muscle_indices, 6].copy()

        # 4. [Design Phase Config]
        self.design_steps = 1          
        self.design_step_counter = 0   
        
        # === 修改点 A: 维度改为 1 ===
        self.design_dim = self.num_muscles * 1 
        self.control_dim = self.sim.model.nu 
        
        # Init Scales
        self.current_scales = np.ones((self.num_muscles, 1), dtype=np.float32)
        
        print(f"DEBUG: T2A Walk Setup - Found {self.num_muscles} muscles. Mode: Velocity (Vmax) Only.")
        return kwargs

    def _fix_action_space(self):
        dim = max(self.control_dim, self.design_dim)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(dim,), dtype=np.float32)

    def _apply_design(self, design_act):
        """
        Map Policy Actions [-1, 1] to Vmax Scaling.
        Range: [0.5, 1.5] (速度参数通常允许较大的变化范围)
        """
        design_act = np.clip(design_act, -1.0, 1.0)
        
        # === 修改点 B: 解析 1 个参数 ===
        design_params = design_act.reshape(self.num_muscles, 1)
        
        # === Velocity Scaling (Vmax) ===
        # Linear Mapping: -1 -> 0.5x (慢肌), 1 -> 1.5x (快肌)
        # Formula: 1.0 + 0.5 * x
        v_scale = 1.0 + 0.5 * design_params[:, 0]
        
        # === 修改点 C: 应用到 Index 6 (Vmax) ===
        # 只修改 gainprm[6]，不碰 F0 (index 2) 或 Length (index 0,1)
        self.sim.model.actuator_gainprm[self.muscle_indices, 6] = self.original_vmax * v_scale
        
        # Store for observation
        self.current_scales = v_scale.reshape(-1, 1)

    def reset_t2a(self):
        # Restore originals
        self.sim.model.actuator_gainprm[self.muscle_indices, 6] = self.original_vmax.copy()
        
        self.current_scales = np.ones((self.num_muscles, 1), dtype=np.float32)
        self.design_step_counter = 0

    def get_obs_t2a(self, obs_dict):
        obs_dict["design_params"] = self.current_scales.flatten().copy()
        return obs_dict

    def step_t2a(self, action, parent_step_func):
        current_action_dim = action.shape[0]
        
        # === Phase 1: Design ===
        if self.design_step_counter < self.design_steps:
            if current_action_dim < self.design_dim:
                padded_design = np.zeros(self.design_dim, dtype=action.dtype)
                padded_design[:current_action_dim] = action
                design_act = padded_design
            else:
                design_act = action[:self.design_dim]

            self._apply_design(design_act)
            self.design_step_counter += 1
            
            obs = self.get_obs()
            info = {"time": self.sim.data.time, "phase": "design"}
            return obs, 0.0, False, False, info 

        # === Phase 2: Control ===
        else:
            if current_action_dim < self.control_dim:
                control_act = np.zeros(self.control_dim, dtype=action.dtype)
                control_act[:current_action_dim] = action
            else:
                control_act = action[:self.control_dim]
                
            return parent_step_func(control_act)

    def visualize_evolution(self, save_path="walk_velocity_evolution_heatmap.png", show=False):
        if self.current_scales is None: return
        data = self.current_scales 
        plt.figure(figsize=(8, 12)) 
        ax = sns.heatmap(data, 
                    yticklabels=self.muscle_names, 
                    xticklabels=["Velocity Scale"],
                    cmap="vlag", center=1.0, annot=False) 
        plt.title(f"Muscle Velocity Evolution", fontsize=16)
        plt.tight_layout()
        save_path = os.path.abspath(save_path)
        plt.savefig(save_path, dpi=300)
        print(f"Heatmap saved to: {save_path}")
        if show: plt.show()
        plt.close()

# ===========================================================================
#  Env Classes
# ===========================================================================
class WalkEnvT2A(T2AWalkMixin, WalkEnvV0):
    def _setup(self, **kwargs):
        kwargs = self._setup_t2a(**kwargs)
        super()._setup(**kwargs)
        self._fix_action_space()
    def reset(self, **kwargs):
        self.reset_t2a()
        return super().reset(**kwargs)
    def get_obs_dict(self, sim):
        obs_dict = super().get_obs_dict(sim)
        return self.get_obs_t2a(obs_dict)
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
        obs_dict = super().get_obs_dict(sim)
        return self.get_obs_t2a(obs_dict)
    def step(self, action):
        return self.step_t2a(action, super().step)