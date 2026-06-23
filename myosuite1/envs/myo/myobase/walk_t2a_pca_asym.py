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
    Asymmetric Spectral Evolution Version:
    1. 取消双边对称约束，每块肌肉独立进化。
    2. 基于 (80 x K) 的非对称谱空间矩阵进行映射。
    """
    def _setup_t2a(self, **kwargs):
        # 1. 识别肌肉（取消对称分组逻辑）
        self.muscle_indices = np.where(self.sim.model.actuator_dyntype == mujoco.mjtDyn.mjDYN_MUSCLE)[0]
        if len(self.muscle_indices) == 0:
            self.muscle_indices = np.arange(self.sim.model.nu)
            
        # --- 修改点：每个索引就是一个独立的组，不再裁剪名称 ---
        self.muscle_groups = collections.OrderedDict()
        for idx in self.muscle_indices:
            try:
                name = self.sim.model.actuator(int(idx)).name
            except AttributeError:
                name = self.sim.model.actuator_id2name(int(idx))
            
            # 直接使用全名作为键，不再去掉 _l 或 _r
            self.muscle_groups[name] = [idx]
        
        self.muscle_names_list = list(self.muscle_groups.keys())
        self.num_muscles = len(self.muscle_names_list) # 应该是 80

        # =========================================================
        # [核心修改]: 加载非对称谱空间协同矩阵 (应为 80 x K)
        # =========================================================
        filename = "synergy_W_basis_asym.npy" # 确保文件名正确
        w_path_local = filename
        w_path_env = os.path.join(os.path.dirname(__file__), filename)
        
        if os.path.exists(w_path_local):
            self.W_basis = np.load(w_path_local)
        elif os.path.exists(w_path_env):
            self.W_basis = np.load(w_path_env)
        else:
            raise FileNotFoundError(f"CRITICAL: 找不到 {filename}！")
            
        self.latent_k = self.W_basis.shape[1] 
        print(f"SUCCESS: Asymmetric Matrix Loaded. Shape: {self.W_basis.shape}")
        
        # 进化维度依然是 K * 3
        self.design_dim = self.latent_k * 3 
        self.control_dim = self.sim.model.nu 
        self.total_action_dim = self.control_dim + self.design_dim

        # 2. 备份原始参数
        self.original_gainprm = self.sim.model.actuator_gainprm.copy()
        self.original_biasprm = self.sim.model.actuator_biasprm.copy()

        # 3. 状态初始化
        self.design_steps = 1          
        self.design_step_counter = 0   
        self.current_scales = np.ones((self.sim.model.nu, 3), dtype=np.float32)
        
        print(f"DEBUG: ASYMMETRIC T2A Enabled. Muscles: {self.num_muscles}, Design Dim: {self.design_dim}")
        return kwargs

    def _fix_action_space(self):
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(self.total_action_dim,), dtype=np.float32)

    def _apply_design(self, design_act):
        """
        非对称映射逻辑：直接将 80 维参数应用到 80 块肌肉
        """
        design_act = np.clip(design_act, -1.0, 1.0)
        
        # 1. 拆分隐变量
        latent_f = design_act[0 : self.latent_k]
        latent_v = design_act[self.latent_k : 2*self.latent_k]
        latent_k = design_act[2*self.latent_k : 3*self.latent_k]

        # 2. 矩阵相乘：得到 80 块肌肉各自的缩放基准值
        # Shape: (80, K) @ (K,) -> (80,)
        muscle_f_raw = self.W_basis @ latent_f
        muscle_v_raw = self.W_basis @ latent_v
        muscle_k_raw = self.W_basis @ latent_k

        # 3. 平滑转换到生理倍率区间 (80 维)
        full_f_scale = np.where(muscle_f_raw > 0, 1.0 + 0.5 * np.tanh(muscle_f_raw), 1.0 + 0.2 * np.tanh(muscle_f_raw))
        full_v_scale = 1.0 + 0.5 * np.tanh(muscle_v_raw)
        full_k_scale = 1.0 + 0.2 * np.tanh(muscle_k_raw)

        # 4. 写入 MuJoCo 模型 (由于没有对称约束，这里直接对应 80 维索引)
        m = self.sim.model
        # 力量
        m.actuator_gainprm[self.muscle_indices, 2] = self.original_gainprm[self.muscle_indices, 2] * full_f_scale
        m.actuator_biasprm[self.muscle_indices, 1] = -(m.actuator_gainprm[self.muscle_indices, 2])
        # 速度阻尼
        m.actuator_gainprm[self.muscle_indices, 6] = self.original_gainprm[self.muscle_indices, 6] * full_v_scale
        # 刚度
        m.actuator_biasprm[self.muscle_indices, 2] = self.original_biasprm[self.muscle_indices, 2] * full_k_scale
        
        # 保存状态
        # 注意：这里需要构造全量 (nu, 3) 矩阵用于状态观察
        self.current_scales = np.ones((m.nu, 3), dtype=np.float32)
        self.current_scales[self.muscle_indices, 0] = full_f_scale
        self.current_scales[self.muscle_indices, 1] = full_v_scale
        self.current_scales[self.muscle_indices, 2] = full_k_scale

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
            obs = self.get_obs()
            return obs, 0.0, False, False, {"phase": "design"} 
        else:
            return parent_step_func(control_act)

    def reset_t2a(self):
        self.sim.model.actuator_gainprm[:] = self.original_gainprm.copy()
        self.sim.model.actuator_biasprm[:] = self.original_biasprm.copy()
        self.current_scales = np.ones((self.sim.model.nu, 3), dtype=np.float32)
        self.design_step_counter = 0

# 环境包装类（与之前一致，但继承了修改后的 Mixin）
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
    # 此处逻辑与 WalkEnvT2A 相同
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