import numpy as np
import mujoco
from gymnasium import spaces
import matplotlib.pyplot as plt
import seaborn as sns
import os
import collections

# 导入原始环境
from myosuite.envs.myo.myochallenge.bimanual_v0 import BimanualEnvV1

class T2ABimanualFullMixin:
    """
    Mixin for Co-evolution of Strength, Velocity, and Stiffness in Bimanual Task.
    """
    def _setup_t2a(self, **kwargs):
        # 1. [观测配置]
        obs_keys = kwargs.get('obs_keys', self.DEFAULT_OBS_KEYS)
        if "design_params" not in obs_keys:
            if isinstance(obs_keys, tuple): obs_keys = list(obs_keys)
            obs_keys.append("design_params")
        kwargs['obs_keys'] = obs_keys

        # 2. [识别肌肉]
        self.muscle_indices = np.where(self.sim.model.actuator_dyntype == mujoco.mjtDyn.mjDYN_MUSCLE)[0]
        self.num_muscles = len(self.muscle_indices)
        
        self.muscle_names = []
        for i in self.muscle_indices:
            try: name = self.sim.model.actuator_id2name(int(i))
            except: name = f"M{i}"
            self.muscle_names.append(name)

        # 3. [备份原始参数]
        # gainprm: [..., gain_type, gain_param, force_peak, ..., vmax, ...]
        # biasprm: [..., bias_type, bias_param, stiffness, ...]
        self.original_gainprm = self.sim.model.actuator_gainprm[self.muscle_indices].copy()
        self.original_biasprm = self.sim.model.actuator_biasprm[self.muscle_indices].copy()

        # 4. [设计阶段配置]
        self.design_steps = 1          
        self.design_step_counter = 0   
        
        # 核心：每块肌肉 3 个进化参数
        self.design_dim = self.num_muscles * 3 
        self.control_dim = self.sim.model.nu 
        
        # 存储当前缩放比例 [Num_Muscles, 3]
        self.current_scales = np.ones((self.num_muscles, 3), dtype=np.float32)
        
        print(f"DEBUG: T2A Bimanual Full Evolution - Muscles: {self.num_muscles}, Total Design Dim: {self.design_dim}")
        return kwargs

    def _fix_action_space(self):
        dim = max(self.control_dim, self.design_dim)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(dim,), dtype=np.float32)

    def _apply_design(self, design_act):
        """
        映射动作到物理参数
        idx 0: Strength (gainprm[2])
        idx 1: Velocity (gainprm[6])
        idx 2: Stiffness (biasprm[2])
        """
        design_act = np.clip(design_act, -1.0, 1.0)
        # 将展平的动作重塑为 [肌肉数, 3]
        design_params = design_act.reshape(self.num_muscles, 3)
        
        # --- 1. Strength (F0) ---
        f_scale = np.where(
            design_params[:, 0] > 0,
            1.0 + 0.5 * design_params[:, 0],  # [1.0, 1.5]
            1.0 + 0.2 * design_params[:, 0]   # [0.8, 1.0]
        )
        self.sim.model.actuator_gainprm[self.muscle_indices, 2] = self.original_gainprm[:, 2] * f_scale

        # --- 2. Velocity (Vmax) ---
        # 映射范围 [0.5, 1.5]
        v_scale = 1.0 + 0.5 * design_params[:, 1]
        self.sim.model.actuator_gainprm[self.muscle_indices, 6] = self.original_gainprm[:, 6] * v_scale

        # --- 3. Stiffness (Kpe) ---
        # 映射范围 [0.8, 1.2]
        k_scale = 1.0 + 0.2 * design_params[:, 2]
        self.sim.model.actuator_biasprm[self.muscle_indices, 2] = self.original_biasprm[:, 2] * k_scale
        
        self.current_scales = np.stack([f_scale, v_scale, k_scale], axis=1)

    def reset_t2a(self):
        self.sim.model.actuator_gainprm[self.muscle_indices] = self.original_gainprm.copy()
        self.sim.model.actuator_biasprm[self.muscle_indices] = self.original_biasprm.copy()
        self.current_scales = np.ones((self.num_muscles, 3), dtype=np.float32)
        self.design_step_counter = 0

    def get_obs_t2a(self, obs_dict):
        obs_dict["design_params"] = self.current_scales.flatten().copy()
        return obs_dict

    def step_t2a(self, action, parent_step_func):
        current_action_dim = action.shape[0]
        
        if self.design_step_counter < self.design_steps:
            # 阶段 1: 进化 (映射 3*N 维度动作)
            design_act = action[:self.design_dim] if current_action_dim >= self.design_dim else np.zeros(self.design_dim)
            self._apply_design(design_act)
            self.design_step_counter += 1
            return self.get_obs(), 0.0, False, False, {"phase": "design"} 
        else:
            # 阶段 2: 控制
            control_act = action[:self.control_dim] if current_action_dim >= self.control_dim else np.zeros(self.control_dim)
            return parent_step_func(control_act)

    def visualize_evolution(self, save_path="bimanual_full_evolution.png", show=False):
        if self.current_scales is None: return
        plt.figure(figsize=(10, 15)) 
        sns.heatmap(self.current_scales, 
                    yticklabels=self.muscle_names, 
                    xticklabels=["Strength", "Velocity", "Stiffness"],
                    cmap="vlag", center=1.0, annot=True, fmt=".2f") 
        plt.title("Bimanual Multi-Parameter Muscle Evolution")
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        if show: plt.show()
        plt.close()

class BimanualEnvT2A(T2ABimanualFullMixin, BimanualEnvV1):
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