import numpy as np
import mujoco
from gymnasium import spaces
import matplotlib.pyplot as plt
import seaborn as sns
import collections

# 导入原始环境
from myosuite.envs.myo.myochallenge.bimanual_v0 import BimanualEnvV1

class T2ABimanualStiffnessMixin:
    """
    Mixin for Evolution of Muscle Passive Stiffness (Kpe) only.
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
        self.num_muscles = len(self.muscle_indices)
        
        self.muscle_names = []
        for i in self.muscle_indices:
            try: name = self.sim.model.actuator_id2name(int(i))
            except: name = f"M{i}"
            self.muscle_names.append(name)

        # 3. [Backup Original Parameters]
        # biasprm 索引 2 对应肌肉的被动比例因子 (Passive force scale/stiffness)
        self.original_biasprm = self.sim.model.actuator_biasprm[self.muscle_indices].copy()

        # 4. [Design Phase Config]
        self.design_steps = 1          
        self.design_step_counter = 0   
        
        # 维度为 1 (每块肌肉仅 1 个参数: Stiffness)
        self.design_dim = self.num_muscles * 1 
        self.control_dim = self.sim.model.nu 
        
        # 初始化缩放比例
        self.current_scales = np.ones((self.num_muscles, 1), dtype=np.float32)
        
        print(f"DEBUG: T2A Bimanual Setup - Mode: Stiffness (Kpe) Only. Dim: {self.design_dim}")
        return kwargs

    def _fix_action_space(self):
        dim = max(self.control_dim, self.design_dim)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(dim,), dtype=np.float32)

    def _apply_design(self, design_act):
        """
        Map Policy Actions to Physical Parameters.
        Action range [-1, 1] -> Stiffness [0.5, 1.5]
        """
        design_act = np.clip(design_act, -1.0, 1.0)
        
        # 解析参数 (Num_Muscles,)
        design_params = design_act[:self.num_muscles]
        
        # 映射公式: -1 -> 0.5 (变松), 0 -> 1.0 (不变), 1 -> 1.5 (变紧)
        k_scale = 1.0 + 0.5 * design_params
        
        # 修改 MuJoCo 模型中的 Kpe (biasprm 索引为 2)
        self.sim.model.actuator_biasprm[self.muscle_indices, 2] = self.original_biasprm[:, 2] * k_scale
        
        # 存储用于 Observation
        self.current_scales = k_scale.reshape(-1, 1)

    def reset_t2a(self):
        self.sim.model.actuator_biasprm[self.muscle_indices] = self.original_biasprm.copy()
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

    def visualize_evolution(self, save_path="bimanual_stiffness_evolution.png", show=False):
        if self.current_scales is None: return
        plt.figure(figsize=(6, 12)) 
        sns.heatmap(self.current_scales, 
                    yticklabels=self.muscle_names, 
                    xticklabels=["Stiffness (Kpe)"],
                    cmap="PiYG", center=1.0, annot=True, fmt=".2f") 
        plt.title("Muscle Stiffness Evolution")
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        if show: plt.show()
        plt.close()

class BimanualEnvT2A(T2ABimanualStiffnessMixin, BimanualEnvV1):
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