import numpy as np
import mujoco
import gymnasium as gym
from gymnasium import spaces
import matplotlib.pyplot as plt
import seaborn as sns
import os
import collections

# 导入原始环境
from myosuite.envs.myo.myobase.walk_v0 import WalkEnvV0, TerrainEnvV0

class T2AWalkMixin:
    """
    Advanced PSE-T2A Mixin (Spectral Evolution Version):
    1. Spectral Synergy Mapping (基于 PCA 的谱空间模态降维)
    2. Bilateral Symmetry Constraint (左右腿一致性约束内置于 W 矩阵)
    3. Action Decoupling (控制与设计动作彻底分离)
    """
    def _setup_t2a(self, **kwargs):
        # 1. 识别肌肉并建立映射 (保留你的对称识别逻辑，用于和 W 矩阵对齐)
        self.muscle_indices = np.where(self.sim.model.actuator_dyntype == mujoco.mjtDyn.mjDYN_MUSCLE)[0]
        if len(self.muscle_indices) == 0:
            self.muscle_indices = np.arange(self.sim.model.nu)
            
        self.muscle_groups = collections.OrderedDict()
        for idx in self.muscle_indices:
            try:
                name = self.sim.model.actuator(int(idx)).name
            except AttributeError:
                name = self.sim.model.actuator_id2name(int(idx))
            
            # 去掉末尾的 _l 或 _r 获取基准名称
            base_name = name[:-2] if (name.endswith('_l') or name.endswith('_r')) else name
            if base_name not in self.muscle_groups:
                self.muscle_groups[base_name] = []
            self.muscle_groups[base_name].append(idx)
        
        self.muscle_group_names = list(self.muscle_groups.keys())
        self.num_groups = len(self.muscle_group_names) # 应该是 40

        # =========================================================
        # [核心新增]: 加载谱空间协同矩阵 W_basis
        # =========================================================
        w_path_local = "synergy_W_basis3.npy"
        w_path_env = os.path.join(os.path.dirname(__file__), "synergy_W_basis3.npy")
        
        if os.path.exists(w_path_local):
            self.W_basis = np.load(w_path_local)
        elif os.path.exists(w_path_env):
            self.W_basis = np.load(w_path_env)
        else:
            raise FileNotFoundError(f"CRITICAL: 找不到 synergy_W_basis3.npy！\n请确保它在 {os.getcwd()} 或 {os.path.dirname(__file__)} 下。")
            
        self.latent_k = self.W_basis.shape[1] # 取模态数量，应该是 5
        print(f"SUCCESS: Spectral Synergy Matrix Loaded. Shape: {self.W_basis.shape} (Groups x Modes)")
        
        # 进化维度大幅降低：模态数(5) * 3 (Str, Vel, Sti) = 15
        self.design_dim = self.latent_k * 3 
        
        # 控制维度：固定的 80 (MyoLeg 执行器总数)
        self.control_dim = self.sim.model.nu 
        self.total_action_dim = self.control_dim + self.design_dim

        # 2. 备份原始参数 (备份全量 80 块肌肉)
        self.original_gainprm = self.sim.model.actuator_gainprm.copy()
        self.original_biasprm = self.sim.model.actuator_biasprm.copy()

        # 3. 状态初始化
        self.design_steps = 1          
        self.design_step_counter = 0   
        self.current_scales = np.ones((self.sim.model.nu, 3), dtype=np.float32)
        
        print(f"DEBUG: Spectral T2A Enabled. Original 120-dim Design Space compressed to {self.design_dim} dims.")
        print(f"DEBUG: Total Action Space: {self.total_action_dim} [80 Control + {self.design_dim} Design]")
        return kwargs

    def _fix_action_space(self):
        """ 彻底解耦动作空间 """
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(self.total_action_dim,), dtype=np.float32)
        print(f"DEBUG: Action Space Fixed!")

    def _apply_design(self, design_act):
        """
        谱空间映射逻辑：
        利用 W 矩阵将低维模态权重映射到高维物理参数上。
        """
        design_act = np.clip(design_act, -1.0, 1.0)
        
        # =========================================================
        # [核心数学运算]
        # =========================================================
        # 1. 切分为力量、速度、刚度的模态权重 (各自 5 维)
        latent_f = design_act[0 : self.latent_k]
        latent_v = design_act[self.latent_k : 2*self.latent_k]
        latent_k = design_act[2*self.latent_k : 3*self.latent_k]

        # 2. 矩阵相乘映射回 40 组肌肉: Shape (40, 5) @ (5,) -> (40,)
        # group_x 包含了 40 组肌肉的变化基准值
        group_f = self.W_basis @ latent_f
        group_v = self.W_basis @ latent_v
        group_k = self.W_basis @ latent_k

        # 3. 将线性映射结果平滑转换到生理倍率区间
        # 力: [0.8, 1.5]
        # 速: [0.5, 1.5]
        # 刚: [0.8, 1.2]
        # 使用 Tanh 保证不会越界
        group_f_scales = np.where(group_f > 0, 
                                  1.0 + 0.5 * np.tanh(group_f), 
                                  1.0 + 0.2 * np.tanh(group_f)) 
        group_v_scales = 1.0 + 0.5 * np.tanh(group_v)
        group_k_scales = 1.0 + 0.2 * np.tanh(group_k)

        # =========================================================
        # 将 40 组参数广播到具体的 80 块肌肉上
        # =========================================================
        full_f_scale = np.ones(self.sim.model.nu)
        full_v_scale = np.ones(self.sim.model.nu)
        full_k_scale = np.ones(self.sim.model.nu)

        for i, base_name in enumerate(self.muscle_group_names):
            indices = self.muscle_groups[base_name]
            for idx in indices:
                full_f_scale[idx] = group_f_scales[i]
                full_v_scale[idx] = group_v_scales[i]
                full_k_scale[idx] = group_k_scales[i]

        # 2. 写入 MuJoCo 模型
        m = self.sim.model
        # Strength & Bias Sync
        m.actuator_gainprm[:, 2] = self.original_gainprm[:, 2] * full_f_scale
        m.actuator_biasprm[:, 1] = -(self.original_gainprm[:, 2] * full_f_scale)
        # Velocity
        m.actuator_gainprm[:, 6] = self.original_gainprm[:, 6] * full_v_scale
        # Stiffness
        m.actuator_biasprm[:, 2] = self.original_biasprm[:, 2] * full_k_scale
        
        # 保存用于状态观测 (Radar Chart 等可视化依然可以调用它)
        self.current_scales = np.stack([full_f_scale, full_v_scale, full_k_scale], axis=1)

    def get_obs_t2a(self, obs_dict):
        # 观察空间保留全量 80 肌肉的状态 (展平为 240 维)
        obs_dict["design_params"] = self.current_scales.flatten().copy()
        is_design = 1.0 if self.design_step_counter < self.design_steps else 0.0
        obs_dict["is_design_phase"] = np.array([is_design], dtype=np.float32)
        return obs_dict

    def step_t2a(self, action, parent_step_func):
        # 自动填充保护机制
        if len(action) == self.control_dim:
            padded_action = np.zeros(self.total_action_dim, dtype=action.dtype)
            padded_action[:self.control_dim] = action
            action = padded_action

        control_act = action[:self.control_dim]
        # 提取后 15 维作为设计动作
        design_act = action[self.control_dim : self.control_dim + self.design_dim]
        
        if self.design_step_counter < self.design_steps:
            self._apply_design(design_act)
            self.design_step_counter += 1
            obs = self.get_obs()
            return obs, 0.0, False, False, {"phase": "design"} 

        else:
            return parent_step_func(control_act)

    def reset_t2a(self):
        self.sim.model.actuator_gainprm[:] = self.original_gainprm.copy()
        self.sim.model.actuator_biasprm[:] = self.original_biasprm.copy()
        self.current_scales = np.ones((self.sim.model.nu, 3), dtype=np.float32)
        self.design_step_counter = 0

# ===========================================================================
#  封装后的环境类
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