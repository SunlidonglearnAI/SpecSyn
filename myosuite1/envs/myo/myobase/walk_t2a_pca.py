import numpy as np
import mujoco
import gymnasium as gym
from gymnasium import spaces
import os
import collections

# 导入原始环境
from myosuite.envs.myo.myobase.walk_v0 import WalkEnvV0, TerrainEnvV0

class T2AWalkMixin:
    """
    消融实验版本：支持三种模式
    mode='strength': 仅进化力量 (Strength)
    mode='velocity': 仅进化速度 (Velocity)
    mode='stiffness': 仅进化刚度 (Stiffness)
    mode='all': 同时进化三个 (原版)
    """
    def _setup_t2a(self, evolution_mode='all', **kwargs):
        self.evolution_mode = evolution_mode
        
        # 1. 识别肌肉并建立映射
        self.muscle_indices = np.where(self.sim.model.actuator_dyntype == mujoco.mjtDyn.mjDYN_MUSCLE)[0]
        self.muscle_groups = collections.OrderedDict()
        for idx in self.muscle_indices:
            try:
                name = self.sim.model.actuator(int(idx)).name
            except AttributeError:
                name = self.sim.model.actuator_id2name(int(idx))
            base_name = name[:-2] if (name.endswith('_l') or name.endswith('_r')) else name
            if base_name not in self.muscle_groups:
                self.muscle_groups[base_name] = []
            self.muscle_groups[base_name].append(idx)
        
        self.muscle_group_names = list(self.muscle_groups.keys())
        self.num_groups = len(self.muscle_group_names)

        # 2. 加载协同矩阵
        filename = "synergy_W_basis5.npy"
        w_path = os.path.join(os.path.dirname(__file__), filename)
        if not os.path.exists(w_path): w_path = filename
        self.W_basis = np.load(w_path)
        self.latent_k = self.W_basis.shape[1] 
        
        # === [核心修改]：根据模式调整设计维度 ===
        if self.evolution_mode == 'all':
            self.design_dim = self.latent_k * 3 
        else:
            self.design_dim = self.latent_k # 消融实验只需进化 5 维
        
        self.control_dim = self.sim.model.nu 
        self.total_action_dim = self.control_dim + self.design_dim

        # 3. 备份参数与初始化
        self.original_gainprm = self.sim.model.actuator_gainprm.copy()
        self.original_biasprm = self.sim.model.actuator_biasprm.copy()
        self.design_steps = 1          
        self.design_step_counter = 0   
        self.current_scales = np.ones((self.sim.model.nu, 3), dtype=np.float32)
        
        print(f"DEBUG: Mode [{self.evolution_mode}] Enabled. Design Dim: {self.design_dim}")
        return kwargs

    def _fix_action_space(self):
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(self.total_action_dim,), dtype=np.float32)

    def _apply_design(self, design_act):
        design_act = np.clip(design_act, -1.0, 1.0)
        
        # 初始化 40 组肌肉的缩放比例（默认为 1.0，即不进化）
        group_f_scales = np.ones(self.num_groups)
        group_v_scales = np.ones(self.num_groups)
        group_k_scales = np.ones(self.num_groups)

        # === [核心修改]：根据模式解析 design_act ===
        if self.evolution_mode == 'all':
            latent_f = design_act[0 : self.latent_k]
            latent_v = design_act[self.latent_k : 2*self.latent_k]
            latent_k_val = design_act[2*self.latent_k : 3*self.latent_k]
            
            group_f = self.W_basis @ latent_f
            group_v = self.W_basis @ latent_v
            group_k = self.W_basis @ latent_k_val
            
            group_f_scales = np.where(group_f > 0, 1.0 + 0.5 * np.tanh(group_f), 1.0 + 0.2 * np.tanh(group_f)) 
            group_v_scales = 1.0 + 0.5 * np.tanh(group_v)
            group_k_scales = 1.0 + 0.2 * np.tanh(group_k)

        elif self.evolution_mode == 'strength':
            group_f = self.W_basis @ design_act
            group_f_scales = np.where(group_f > 0, 1.0 + 0.5 * np.tanh(group_f), 1.0 + 0.2 * np.tanh(group_f))

        elif self.evolution_mode == 'velocity':
            group_v = self.W_basis @ design_act
            group_v_scales = 1.0 + 0.5 * np.tanh(group_v)

        elif self.evolution_mode == 'stiffness':
            group_k = self.W_basis @ design_act
            group_k_scales = 1.0 + 0.2 * np.tanh(group_k)

        # 将参数广播到 80 块肌肉
        full_f_scale = np.ones(self.sim.model.nu)
        full_v_scale = np.ones(self.sim.model.nu)
        full_k_scale = np.ones(self.sim.model.nu)

        for i, base_name in enumerate(self.muscle_group_names):
            indices = self.muscle_groups[base_name]
            for idx in indices:
                full_f_scale[idx] = group_f_scales[i]
                full_v_scale[idx] = group_v_scales[i]
                full_k_scale[idx] = group_k_scales[i]

        m = self.sim.model
        m.actuator_gainprm[:, 2] = self.original_gainprm[:, 2] * full_f_scale
        m.actuator_biasprm[:, 1] = -(self.original_gainprm[:, 2] * full_f_scale)
        m.actuator_gainprm[:, 6] = self.original_gainprm[:, 6] * full_v_scale
        m.actuator_biasprm[:, 2] = self.original_biasprm[:, 2] * full_k_scale
        
        self.current_scales = np.stack([full_f_scale, full_v_scale, full_k_scale], axis=1)

    def get_obs_t2a(self, obs_dict):
        obs_dict["design_params"] = self.current_scales.flatten().copy()
        is_design = 1.0 if self.design_step_counter < self.design_steps else 0.0
        obs_dict["is_design_phase"] = np.array([is_design], dtype=np.float32)
        return obs_dict

    def step_t2a(self, action, parent_step_func):
        if len(action) == self.control_dim:
            padded_action = np.zeros(self.total_action_dim, dtype=action.dtype)
            padded_action[:self.control_dim] = action
            action = padded_action

        control_act = action[:self.control_dim]
        design_act = action[self.control_dim : self.control_dim + self.design_dim]
        
        if self.design_step_counter < self.design_steps:
            self._apply_design(design_act)
            self.design_step_counter += 1
            return self.get_obs(), 0.0, False, False, {"phase": "design"} 
        return parent_step_func(control_act)

    def reset_t2a(self):
        self.sim.model.actuator_gainprm[:] = self.original_gainprm.copy()
        self.sim.model.actuator_biasprm[:] = self.original_biasprm.copy()
        self.current_scales = np.ones((self.sim.model.nu, 3), dtype=np.float32)
        self.design_step_counter = 0

# ===========================================================================
#  子类：定义具体的消融实验环境
# ===========================================================================

# 1. 力量进化环境
class WalkEnvT2AStrength(T2AWalkMixin, WalkEnvV0):
    def _setup(self, **kwargs):
        kwargs = self._setup_t2a(evolution_mode='strength', **kwargs)
        super()._setup(**kwargs)
        self._fix_action_space()
    def reset(self, **kwargs):
        self.reset_t2a()
        return super().reset(**kwargs)
    def step(self, action):
        return self.step_t2a(action, super().step)

# 2. 速度进化环境
class WalkEnvT2AVelocity(T2AWalkMixin, WalkEnvV0):
    def _setup(self, **kwargs):
        kwargs = self._setup_t2a(evolution_mode='velocity', **kwargs)
        super()._setup(**kwargs)
        self._fix_action_space()
    def reset(self, **kwargs):
        self.reset_t2a()
        return super().reset(**kwargs)
    def step(self, action):
        return self.step_t2a(action, super().step)

# 3. 刚度进化环境
class WalkEnvT2AStiffness(T2AWalkMixin, WalkEnvV0):
    def _setup(self, **kwargs):
        kwargs = self._setup_t2a(evolution_mode='stiffness', **kwargs)
        super()._setup(**kwargs)
        self._fix_action_space()
    def reset(self, **kwargs):
        self.reset_t2a()
        return super().reset(**kwargs)
    def step(self, action):
        return self.step_t2a(action, super().step)

# 4. 原版环境（全部进化）
class WalkEnvT2A(T2AWalkMixin, WalkEnvV0):
    def _setup(self, **kwargs):
        kwargs = self._setup_t2a(evolution_mode='all', **kwargs)
        super()._setup(**kwargs)
        self._fix_action_space()
    def reset(self, **kwargs):
        self.reset_t2a()
        return super().reset(**kwargs)
    def step(self, action):
        return self.step_t2a(action, super().step)