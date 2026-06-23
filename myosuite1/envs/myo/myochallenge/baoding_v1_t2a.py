"""
T2A (Transformer-to-Action) Wrapper for Baoding Balls Task.
[Phase 1: Force Only Evolution]
Co-optimizes only Muscle Force (F0) to ensure stability and fast convergence.
Range: [0.6x, 1.8x]

Place this file in: myosuite/envs/myo/myochallenge/baoding_t2a.py
"""
import numpy as np
import mujoco
import gymnasium as gym
from gymnasium import spaces
import matplotlib.pyplot as plt
import seaborn as sns
import os

# 导入原始环境 (假设 baoding_v1.py 在同一目录下)
from myosuite.envs.myo.myochallenge.baoding_v1 import BaodingEnvV1

class T2ABaodingMixin:
    """
    T2A Mixin specifically for Baoding Balls.
    Focus: Force (F0) Evolution Only.
    """
    def _setup_t2a(self, **kwargs):
        # 1. [T2A Config] Add 'design_params' to observation
        obs_keys = kwargs.get('obs_keys', self.DEFAULT_OBS_KEYS)
        if "design_params" not in obs_keys:
            # Handle tuple or list
            if isinstance(obs_keys, tuple):
                obs_keys = list(obs_keys)
            obs_keys.append("design_params")
        kwargs['obs_keys'] = obs_keys

        # 2. [Identify Muscles]
        # Hand model usually has explicit muscle actuators
        self.muscle_indices = np.where(self.sim.model.actuator_dyntype == mujoco.mjtDyn.mjDYN_MUSCLE)[0]
        # Fallback if detection fails (some hand models use specific definitions)
        if len(self.muscle_indices) == 0:
            self.muscle_indices = np.arange(self.sim.model.nu)
            
        self.num_muscles = len(self.muscle_indices)
        
        # === [ROBUST NAME GETTER] ===
        self.muscle_names = []
        for i in self.muscle_indices:
            name = None
            try:
                if hasattr(self.sim.model, 'id2name'):
                    name = self.sim.model.id2name(int(i), 'actuator')
                if name is None:
                    model_ptr = self.sim.model
                    if hasattr(model_ptr, 'ptr'): 
                        model_ptr = model_ptr.ptr
                    name = mujoco.mj_id2name(model_ptr, mujoco.mjtObj.mjOBJ_ACTUATOR, int(i))
            except Exception:
                pass
            if name is None: name = f"M{i}"
            self.muscle_names.append(name)
        # ============================

        # 3. [Backup Original Parameters]
        # Only need F0 for this phase
        self.original_force = self.sim.model.actuator_gainprm[self.muscle_indices, 2].copy()

        # 4. [Design Phase Config]
        self.design_steps = 1          
        self.design_step_counter = 0   
        
        # Design Dimension = 1 parameter per muscle (Force Only)
        self.design_dim = self.num_muscles * 1 
        self.control_dim = self.sim.model.nu 
        
        # Init Scales
        self.current_scales = np.ones((self.num_muscles, 1), dtype=np.float32)
        
        print(f"DEBUG: T2A Baoding Setup - Found {self.num_muscles} muscles. Mode: Force Only.")
        return kwargs

    def _fix_action_space(self):
        # Resize action space to accommodate the larger of Control or Design
        dim = max(self.control_dim, self.design_dim)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(dim,), dtype=np.float32)

    def _apply_design(self, design_act):
        """
        Map Policy Actions [-1, 1] to Muscle Force Scales.
        Range: [0.6, 1.8] (Proven stable range)
        """
        design_act = np.clip(design_act, -1.0, 1.0)
        
        # Reshape to (Num_Muscles, 1)
        design_params = design_act.reshape(self.num_muscles, 1)
        
        # === Force Scaling ===
        # Linear mapping: -1 -> 0.6x, +1 -> 1.8x
        # Formula: y = 0.6x + 1.2
        f_scale = 0.6 * design_params[:, 0] + 1.2
        
        new_force = self.original_force * f_scale
        self.sim.model.actuator_gainprm[self.muscle_indices, 2] = new_force
        # Update bias (bias[1] is usually -F0 for muscles to set range)
        self.sim.model.actuator_biasprm[self.muscle_indices, 1] = -new_force
        
        # Store for observation
        self.current_scales = f_scale.reshape(-1, 1)

    def reset_t2a(self):
        # Restore originals
        self.sim.model.actuator_gainprm[self.muscle_indices, 2] = self.original_force.copy()
        self.sim.model.actuator_biasprm[self.muscle_indices, 1] = -self.original_force.copy()
        
        self.current_scales = np.ones((self.num_muscles, 1), dtype=np.float32)
        self.design_step_counter = 0

    def get_obs_t2a(self, obs_dict):
        # Add scales to observation
        obs_dict["design_params"] = self.current_scales.flatten().copy()
        return obs_dict

    def step_t2a(self, action, parent_step_func):
        current_action_dim = action.shape[0]
        
        # === Phase 1: Design (Evolution) ===
        if self.design_step_counter < self.design_steps:
            # Handle padding if action is too short (e.g. from parent init)
            if current_action_dim < self.design_dim:
                padded_design = np.zeros(self.design_dim, dtype=action.dtype)
                padded_design[:current_action_dim] = action
                design_act = padded_design
            else:
                design_act = action[:self.design_dim]

            self._apply_design(design_act)
            self.design_step_counter += 1
            
            # Return dummy observation for the design step
            obs = self.get_obs()
            # Info dict with phase info
            info = {"time": self.sim.data.time, "phase": "design"}
            # Reward 0, Not Done
            return obs, 0.0, False, False, info 

        # === Phase 2: Control (Execution) ===
        else:
            # Ensure action matches control dimension
            if current_action_dim < self.control_dim:
                control_act = np.zeros(self.control_dim, dtype=action.dtype)
                control_act[:current_action_dim] = action
            else:
                control_act = action[:self.control_dim]
                
            return parent_step_func(control_act)

    def visualize_evolution(self, save_path="baoding_evolution_heatmap.png", show=False):
        if self.current_scales is None: return
        data = self.current_scales 
        
        plt.figure(figsize=(8, 12)) 
        ax = sns.heatmap(data, 
                    yticklabels=self.muscle_names, 
                    xticklabels=["Force Scaling"],
                    cmap="vlag", center=1.0, annot=True, fmt=".2f")
        plt.title(f"Baoding Hand Force Evolution", fontsize=16)
        plt.tight_layout()
        save_path = os.path.abspath(save_path)
        plt.savefig(save_path, dpi=300)
        print(f"Heatmap saved to: {save_path}")
        if show: plt.show()
        plt.close()

# ===========================================================================
#  Baoding T2A Environment Class
# ===========================================================================

class BaodingEnvT2A(T2ABaodingMixin, BaodingEnvV1):
    def _setup(self, **kwargs):
        # 1. Run T2A Setup (modifies kwargs['obs_keys'])
        kwargs = self._setup_t2a(**kwargs)
        
        # 2. Run Parent Setup (initializes Sim and triggers step)
        super()._setup(**kwargs)
        
        # 3. Fix Action Space (Must be done AFTER parent setup)
        self._fix_action_space()

    def reset(self, **kwargs):
        self.reset_t2a()
        return super().reset(**kwargs)

    def get_obs_dict(self, sim):
        # Get parent observation dict
        obs_dict = super().get_obs_dict(sim)
        # Add T2A params
        return self.get_obs_t2a(obs_dict)

    def step(self, action):
        # Route through T2A logic
        return self.step_t2a(action, super().step)