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
    Advanced PSE-T2A Mixin:
    1. Bilateral Symmetry Constraint (左右腿一致性进化)
    2. Action Decoupling (控制与设计动作彻底分离)
    3. Multi-Parameter Evolution (力量、速度、刚度)
    """
    def _setup_t2a(self, **kwargs):
        # 1. 识别肌肉并建立对称映射
        self.muscle_indices = np.where(self.sim.model.actuator_dyntype == mujoco.mjtDyn.mjDYN_MUSCLE)[0]
        if len(self.muscle_indices) == 0:
            self.muscle_indices = np.arange(self.sim.model.nu)
            
        # 自动识别左右配对逻辑
        self.muscle_groups = collections.OrderedDict()
        for idx in self.muscle_indices:
            name = self.sim.model.actuator(int(idx)).name
            # 去掉末尾的 _l 或 _r 获取基准名称
            base_name = name[:-2] if (name.endswith('_l') or name.endswith('_r')) else name
            if base_name not in self.muscle_groups:
                self.muscle_groups[base_name] = []
            self.muscle_groups[base_name].append(idx)
        
        # 进化维度：组数 * 3 (Str, Vel, Sti)
        self.muscle_group_names = list(self.muscle_groups.keys())
        self.num_groups = len(self.muscle_group_names)
        self.design_dim = self.num_groups * 3 
        
        # 控制维度：固定的 80 (MyoLeg 执行器总数)
        self.control_dim = self.sim.model.nu 
        self.total_action_dim = self.control_dim + self.design_dim

        # 2. 备份原始参数 (备份全量 80 块肌肉)
        self.original_gainprm = self.sim.model.actuator_gainprm.copy()
        self.original_biasprm = self.sim.model.actuator_biasprm.copy()

        # 3. 状态初始化
        self.design_steps = 1          
        self.design_step_counter = 0   
        # 用于观测的 scales 矩阵 (80, 3)
        self.current_scales = np.ones((self.sim.model.nu, 3), dtype=np.float32)
        
        print(f"DEBUG: Symmetry T2A Enabled. Reduced {len(self.muscle_indices)} muscles to {self.num_groups} symmetrical groups.")
        print(f"DEBUG: Design Dim: {self.design_dim}, Control Dim: {self.control_dim}")
        return kwargs

    def _fix_action_space(self):
        """ 彻底解耦动作空间：前 80 维控步，后 N 维控进化 """
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(self.total_action_dim,), dtype=np.float32)
        print(f"DEBUG: Action Space Fixed! Total Dim: {self.total_action_dim} [80 Control + {self.design_dim} Design]")

    def _apply_design(self, design_act):
        """
        对称映射映射逻辑：
        将 design_act (N_groups * 3) 广播到全量肌肉参数中
        """
        design_act = np.clip(design_act, -1.0, 1.0)
        # 解析为 (Group, 3) 矩阵
        group_params = design_act.reshape(self.num_groups, 3)

        # 临时存储全量 80 维的缩放系数
        full_f_scale = np.ones(self.sim.model.nu)
        full_v_scale = np.ones(self.sim.model.nu)
        full_k_scale = np.ones(self.sim.model.nu)

        # 1. 遍历组，将进化动作映射到每一块肌肉
        for i, base_name in enumerate(self.muscle_group_names):
            indices = self.muscle_groups[base_name]
            p = group_params[i] # [str_act, vel_act, sti_act]
            
            # 计算缩放逻辑
            f_s = 1.0 + 0.5 * p[0] if p[0] > 0 else 1.0 + 0.2 * p[0] # [0.8, 1.5]
            v_s = 1.0 + 0.5 * p[1]                                 # [0.5, 1.5]
            k_s = 1.0 + 0.2 * p[2]                                 # [0.8, 1.2]

            for idx in indices:
                full_f_scale[idx] = f_s
                full_v_scale[idx] = v_s
                full_k_scale[idx] = k_s

        # 2. 写入 MuJoCo 模型
        m = self.sim.model
        # Strength & Bias Sync
        m.actuator_gainprm[:, 2] = self.original_gainprm[:, 2] * full_f_scale
        m.actuator_biasprm[:, 1] = -(self.original_gainprm[:, 2] * full_f_scale)
        # Velocity
        m.actuator_gainprm[:, 6] = self.original_gainprm[:, 6] * full_v_scale
        # Stiffness
        m.actuator_biasprm[:, 2] = self.original_biasprm[:, 2] * full_k_scale
        
        # 保存用于状态观测
        self.current_scales = np.stack([full_f_scale, full_v_scale, full_k_scale], axis=1)

    def get_obs_t2a(self, obs_dict):
        # 即使进化是对称的，观察空间建议保留全量 80 肌肉的状态，方便神经网络感知物理属性
        obs_dict["design_params"] = self.current_scales.flatten().copy()
        is_design = 1.0 if self.design_step_counter < self.design_steps else 0.0
        obs_dict["is_design_phase"] = np.array([is_design], dtype=np.float32)
        return obs_dict

    def step_t2a(self, action, parent_step_func):
        # 1. 自动处理 MyoSuite 探测步 (80维输入)
        if len(action) == self.control_dim:
            padded_action = np.zeros(self.total_action_dim, dtype=action.dtype)
            padded_action[:self.control_dim] = action
            action = padded_action

        # 2. 严格切分动作，绝不重叠
        control_act = action[:self.control_dim]
        design_act = action[self.control_dim : self.control_dim + self.design_dim]
        
        # === Phase 1: Design (仅第一步) ===
        if self.design_step_counter < self.design_steps:
            self._apply_design(design_act)
            self.design_step_counter += 1
            
            obs = self.get_obs()
            return obs, 0.0, False, False, {"phase": "design"} 

        # === Phase 2: Control (正式走路) ===
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