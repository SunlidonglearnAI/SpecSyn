"""
T2A (Transformer-to-Action) Wrapper for Object Hold Task.
[Option B]: Narrow Search Space for Faster Convergence.

Co-optimizes:
1. Muscle Force (F0) -> Strength (Range: 0.8x - 1.5x)
2. Muscle Length Range (L_opt) -> Speed/Range (Range: 0.9x - 1.1x)
3. Passive Stiffness (Passive Force) -> Stability/Damping (Range: 0.5x - 2.0x)

Place this file in: myosuite/envs/myo/myobase/obj_hold_t2a.py
"""
import numpy as np
import mujoco
import gymnasium as gym
from gymnasium import spaces
import matplotlib.pyplot as plt
import seaborn as sns
import os

# 导入原始环境
from myosuite.envs.myo.myobase.obj_hold_v0 import ObjHoldFixedEnvV0, ObjHoldRandomEnvV0

class T2AMixin:
    """
    T2A Mixin for 3-Parameter Evolution (Force, Length, Stiffness).
    """
    def _setup_t2a(self, **kwargs):
        # 1. [T2A Config] Add 'design_params' to observation
        obs_keys = kwargs.get('obs_keys', self.DEFAULT_OBS_KEYS)
        if "design_params" not in obs_keys:
            obs_keys = list(obs_keys) + ["design_params"]
        kwargs['obs_keys'] = obs_keys

        # 2. [Identify Muscles]
        self.muscle_indices = np.where(self.sim.model.actuator_dyntype == mujoco.mjtDyn.mjDYN_MUSCLE)[0]
        self.num_muscles = len(self.muscle_indices)
        
        # === [ROBUST NAME GETTER] ===
        self.muscle_names = []
        for i in self.muscle_indices:
            name = None
            try:
                model_ptr = self.sim.model
                if hasattr(model_ptr, 'ptr'):
                    model_ptr = model_ptr.ptr
                name = mujoco.mj_id2name(model_ptr, mujoco.mjtObj.mjOBJ_ACTUATOR, int(i))
            except Exception:
                pass
            
            if name is None:
                name = f"Muscle_{i}"
            self.muscle_names.append(name)
        # ============================

        # 3. [Backup Original Parameters]
        self.original_force = self.sim.model.actuator_gainprm[self.muscle_indices, 2].copy()
        self.original_len_range = self.sim.model.actuator_lengthrange[self.muscle_indices].copy()
        self.original_passive = self.sim.model.actuator_biasprm[self.muscle_indices, 2].copy()

        # 4. [Design Phase Config]
        self.design_steps = 1          
        self.design_step_counter = 0   
        
        # Design Dimension = 3 parameters per muscle * Num Muscles
        self.design_dim = self.num_muscles * 3 
        self.control_dim = self.sim.model.nu 
        
        # Init Scales
        self.current_scales = np.ones((self.num_muscles, 3), dtype=np.float32)
        
        return kwargs

    def _fix_action_space(self):
        # Resize action space
        dim = max(self.control_dim, self.design_dim)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(dim,), dtype=np.float32)

    def _apply_design(self, design_act):
        """
        Map Policy Actions [-1, 1] to Physical Parameters.
        [Option B Modification]: Using narrower ranges for faster convergence.
        """
        design_act = np.clip(design_act, -1.0, 1.0)
        
        # Reshape to (Num_Muscles, 3)
        design_params = design_act.reshape(self.num_muscles, 3)
        
        # === 1. Force Scaling (Strength) ===
        # Old: [0.2, 5.0] -> New: [0.8, 1.5]
        # Logic: If action > 0, scale up to 1.5; if < 0, scale down to 0.8
        f_scale = np.where(
            design_params[:, 0] > 0,
            1.0 + 0.5 * design_params[:, 0],  # 0 -> 1.0, 1 -> 1.5
            1.0 + 0.2 * design_params[:, 0]   # 0 -> 1.0, -1 -> 0.8
        )
        new_force = self.original_force * f_scale
        self.sim.model.actuator_gainprm[self.muscle_indices, 2] = new_force

        # === 2. Length Range Scaling (Velocity/L_opt) ===
        # Old: [0.8, 1.2] -> New: [0.9, 1.1] (Very subtle changes)
        l_scale = 1.0 + 0.1 * design_params[:, 1]
        new_len_range = self.original_len_range * l_scale[:, None]
        self.sim.model.actuator_lengthrange[self.muscle_indices] = new_len_range

        # === 3. Passive Stiffness Scaling (Stability) ===
        # Old: [0.2, 5.0] -> New: [0.5, 2.0] (Symmetric Log Scale)
        # 2^-1 = 0.5, 2^1 = 2.0
        p_scale = np.power(2.0, design_params[:, 2])
        new_passive = self.original_passive * p_scale
        self.sim.model.actuator_biasprm[self.muscle_indices, 2] = new_passive
        
        # Store for observation
        self.current_scales = np.stack([f_scale, l_scale, p_scale], axis=1)

    def reset_t2a(self):
        self.sim.model.actuator_gainprm[self.muscle_indices, 2] = self.original_force.copy()
        self.sim.model.actuator_lengthrange[self.muscle_indices] = self.original_len_range.copy()
        self.sim.model.actuator_biasprm[self.muscle_indices, 2] = self.original_passive.copy()
        self.current_scales = np.ones((self.num_muscles, 3), dtype=np.float32)
        self.design_step_counter = 0

    def get_obs_t2a(self, obs_dict):
        obs_dict["design_params"] = self.current_scales.flatten().copy()
        return obs_dict

    def step_t2a(self, action, parent_step_func):
        # === 关键修复: 处理基类初始化时的短向量 ===
        current_action_dim = action.shape[0]
        
        # === Phase 1: Design (Evolution) ===
        if self.design_step_counter < self.design_steps:
            if current_action_dim < self.design_dim:
                # 补零 padding
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

        # === Phase 2: Control (Execution) ===
        else:
            # 对于 Control Phase，我们也需要确保维度匹配
            if current_action_dim < self.control_dim:
                control_act = np.zeros(self.control_dim, dtype=action.dtype)
                control_act[:current_action_dim] = action
            else:
                control_act = action[:self.control_dim]
                
            return parent_step_func(control_act)

    def visualize_evolution(self, save_path="evolution_heatmap.png", show=False):
        if self.current_scales is None: return
        data = self.current_scales 
        plt.figure(figsize=(10, 12)) 
        ax = sns.heatmap(data, 
                    yticklabels=self.muscle_names, 
                    xticklabels=["Max Force (F0)", "Length Range", "Passive Stiffness"],
                    cmap="vlag", center=1.0, annot=True, fmt=".1f", linewidths=.5)
        plt.title(f"T2A Morphology Evolution Result", fontsize=16)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        print(f"Heatmap saved to: {os.path.abspath(save_path)}")
        if show: plt.show()
        plt.close()

# ===========================================================================
#  Environment Classes
# ===========================================================================

class ObjHoldFixedEnvT2A(T2AMixin, ObjHoldFixedEnvV0):
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

class ObjHoldRandomEnvT2A(T2AMixin, ObjHoldRandomEnvV0):
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