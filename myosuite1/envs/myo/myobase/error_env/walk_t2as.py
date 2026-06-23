"""
T2A (Transformer-to-Action) Wrapper for Terrain Walking Task.
[Stiffness-Only Evolution Mode]
Focus: Isolating the effect of Muscle Passive Stiffness (Compliance).
Target Parameter: biasprm[2] (Passive Force Scale)
Constraint: Strength (F0/gainprm[2]) and Velocity (Vmax/gainprm[6]) are FIXED.
Range: [0.5, 1.5] (+/- 50%)

Physical Meaning:
- Higher Stiffness (>1.0): Muscles generate higher passive restoration forces when stretched. 
  Acts like a "stiffer spring". Good for stability, bad for energy efficiency.
- Lower Stiffness (<1.0): Muscles are more compliant/flexible. 
  Acts like a "loose rubber band". Good for range of motion, requires more active control.

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
#  T2A Mixin: 仅刚度进化逻辑
# ===========================================================================
class T2AWalkMixin:
    """
    Mixin for Stiffness-Only Evolution.
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
        # === 关键修改 ===
        # 只需要备份 biasprm[2]，因为它控制被动刚度
        # 注意：我们不去动 gainprm[2] (Strength)，从而实现刚度与力量的解耦
        self.original_passive_scale = self.sim.model.actuator_biasprm[self.muscle_indices, 2].copy()

        # 4. [Design Phase Config]
        self.design_steps = 1          
        self.design_step_counter = 0   
        
        # Design Dimension = 1 parameter per muscle (Stiffness Only)
        self.design_dim = self.num_muscles * 1 
        self.control_dim = self.sim.model.nu 
        
        # Init Scales
        self.current_scales = np.ones((self.num_muscles, 1), dtype=np.float32)
        
        print(f"DEBUG: T2A Walk Setup - Found {self.num_muscles} muscles. Mode: Stiffness (Passive) Only.")
        return kwargs

    def _fix_action_space(self):
        dim = max(self.control_dim, self.design_dim)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(dim,), dtype=np.float32)

    def _apply_design(self, design_act):
        """
        Map Policy Actions [-1, 1] to Stiffness (Passive Force) Scaling.
        Range: [0.5, 1.5]
        """
        design_act = np.clip(design_act, -1.0, 1.0)
        
        # Reshape to (Num_Muscles, 1)
        design_params = design_act.reshape(self.num_muscles, 1)
        
        # === Stiffness Scaling ===
        # Logic: 
        # Action -1 -> 0.5x (变软/Compliant)
        # Action +1 -> 1.5x (变硬/Stiff)
        k_scale = 1.0 + 0.5 * design_params[:, 0]
        
        # === 应用修改到 biasprm[2] ===
        # 这会改变被动力的大小，但不会改变主动收缩力上限(Strength)
        self.sim.model.actuator_biasprm[self.muscle_indices, 2] = self.original_passive_scale * k_scale
        
        # Store for observation
        self.current_scales = k_scale.reshape(-1, 1)

    def reset_t2a(self):
        # Restore originals
        self.sim.model.actuator_biasprm[self.muscle_indices, 2] = self.original_passive_scale.copy()
        
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

    def visualize_evolution(self, save_path="walk_stiffness_evolution_heatmap.png", show=False):
        if self.current_scales is None: return
        data = self.current_scales 
        plt.figure(figsize=(8, 12)) 
        ax = sns.heatmap(data, 
                    yticklabels=self.muscle_names, 
                    xticklabels=["Stiffness Scale"],
                    cmap="vlag", center=1.0, annot=False) 
        plt.title(f"Muscle Stiffness Evolution", fontsize=16)
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