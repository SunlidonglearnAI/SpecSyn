"""
T2A (Transformer-to-Action) Wrapper for Terrain Walking Task.
[Full Evolution Mode: Strength + Velocity + Stiffness]
Co-optimizes 3 Muscle Parameters simultaneously:

1. Strength (F0)    -> gainprm[2]  | Range: [0.8, 1.5]
2. Velocity (Vmax)  -> gainprm[6]  | Range: [0.5, 1.5]
3. Stiffness (Kpe)  -> biasprm[2]  | Range: [0.8, 1.2] (保守范围以防震荡)

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
#  T2A Mixin: 三参数联合进化逻辑
# ===========================================================================
class T2AWalkMixin:
    """
    Mixin for Simultaneous Evolution of Strength, Velocity, and Stiffness.
    """
    def _setup_t2a(self, **kwargs):
        # # 1. [关键修复]：使用副本防止修改类级别的 DEFAULT_OBS_KEYS
        # obs_keys = kwargs.get('obs_keys', self.DEFAULT_OBS_KEYS)
        # if isinstance(obs_keys, (list, tuple)):
        #     obs_keys = list(obs_keys)
        
        # # 必须同时加入这两个键，确保观测完整性
        # if "design_params" not in obs_keys:
        #     obs_keys.append("design_params")
        # if "is_design_phase" not in obs_keys:
        #     obs_keys.append("is_design_phase")
            
        # kwargs['obs_keys'] = obs_keys
        # # 1. [T2A Config]
        # obs_keys = kwargs.get('obs_keys', self.DEFAULT_OBS_KEYS)
        # if "design_params" not in obs_keys:
        #     if isinstance(obs_keys, tuple):
        #         obs_keys = list(obs_keys)
        #     obs_keys.append("design_params")
        # kwargs['obs_keys'] = obs_keys

        # 2. [Identify Muscles]
        self.muscle_indices = np.where(self.sim.model.actuator_dyntype == mujoco.mjtDyn.mjDYN_MUSCLE)[0]
        if len(self.muscle_indices) == 0:
            self.muscle_indices = np.arange(self.sim.model.nu)
            
        self.num_muscles = len(self.muscle_indices)
        self.design_dim = self.num_muscles * 3  # 每块肌肉3个参数
        self.control_dim = self.sim.model.nu    # 控制信号维度 (80)
        self.total_action_dim = self.control_dim + self.design_dim
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
        # 备份所有关键参数的原始值
        self.original_gainprm = self.sim.model.actuator_gainprm[self.muscle_indices].copy()
        self.original_biasprm = self.sim.model.actuator_biasprm[self.muscle_indices].copy()

        # 4. [Design Phase Config]
        self.design_steps = 1          
        self.design_step_counter = 0   
        
        # === 修改点 A: 维度改为 3 (每块肌肉3个参数) ===
        self.design_dim = self.num_muscles * 3 
        self.control_dim = self.sim.model.nu 
        
        # Init Scales
        self.current_scales = np.ones((self.num_muscles, 3), dtype=np.float32)
        
        print(f"DEBUG: T2A Walk Setup - Found {self.num_muscles} muscles. Mode: Full (Str/Vel).")
        return kwargs

    # def _fix_action_space(self):
    #     dim = max(self.control_dim, self.design_dim)
    #     self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(dim,), dtype=np.float32)
    def _fix_action_space(self):
        # 分离动作维度，避免同一个神经元既管设计又管控制
        self.total_action_dim = self.control_dim + self.design_dim
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(self.total_action_dim,), dtype=np.float32)
        print(f"DEBUG: Action Space Fixed! Total Dim: {self.total_action_dim}")

    def _apply_design(self, design_act):
        """
        Map Policy Actions to Physical Parameters.
        idx 0 -> Strength (gainprm[2])
        idx 1 -> Velocity (gainprm[6])
        idx 2 -> Stiffness (biasprm[2])
        """
        design_act = np.clip(design_act, -1.0, 1.0)
        
        # === 解析参数 (Num_Muscles, 3) ===
        design_params = design_act.reshape(self.num_muscles, 3)
        
        # ---------------------------------------------------------
        # 1. Strength (力量) - Index 2 in gainprm
        # Range: [0.8, 1.5]
        # ---------------------------------------------------------
        f_scale = np.where(
            design_params[:, 0] > 0,
            1.0 + 0.5 * design_params[:, 0],  # 0 -> 1.0, 1 -> 1.5
            1.0 + 0.2 * design_params[:, 0]   # 0 -> 1.0, -1 -> 0.8
        )
        # 修改主动力上限 (Strength)
        self.sim.model.actuator_gainprm[self.muscle_indices, 2] = self.original_gainprm[:, 2] * f_scale
        # self.sim.model.actuator_biasprm[self.muscle_indices, 1] = self.original_biasprm[:, 1] * f_scale
        # 同步更新 bias[1] (通常为 -F0) 以维持正确的力学范围
        self.sim.model.actuator_biasprm[self.muscle_indices, 1] = -(self.original_gainprm[:, 2] * f_scale)

        # ---------------------------------------------------------
        # 2. Velocity (速度) - Index 6 in gainprm
        # Range: [0.5, 1.5] (快慢肌差异化)
        # ---------------------------------------------------------
        v_scale = 1.0 + 0.5 * design_params[:, 1]
        self.sim.model.actuator_gainprm[self.muscle_indices, 6] = self.original_gainprm[:, 6] * v_scale

        # ---------------------------------------------------------
        # 3. Stiffness (刚度) - Index 2 in biasprm
        # Range: [0.8, 1.2] (范围稍微收窄以防多参数震荡，你可以改回 0.5-1.5)
        # ---------------------------------------------------------
        k_scale = 1.0 + 0.2 * design_params[:, 2]
        # # 只修改被动力基准，不动主动力
        self.sim.model.actuator_biasprm[self.muscle_indices, 2] = self.original_biasprm[:, 2] * k_scale
        
        # ---------------------------------------------------------
        # Store for observation
        self.current_scales = np.stack([f_scale, v_scale, k_scale], axis=1)

    def reset_t2a(self):
        # Restore originals
        self.sim.model.actuator_gainprm[self.muscle_indices] = self.original_gainprm.copy()
        self.sim.model.actuator_biasprm[self.muscle_indices] = self.original_biasprm.copy()
        
        self.current_scales = np.ones((self.num_muscles, 3), dtype=np.float32)
        self.design_step_counter = 0

    def get_obs_t2a(self, obs_dict):
        # 加入 One-Hot 阶段标识，告诉神经网络当前该干什么,不要同时优化无用参数
        obs_dict["design_params"] = self.current_scales.flatten().copy()
        is_design = 1.0 if self.design_step_counter < self.design_steps else 0.0
        obs_dict["is_design_phase"] = np.array([is_design], dtype=np.float32)
        return obs_dict
        # obs_dict["design_params"] = self.current_scales.flatten().copy()
        # return obs_dict

    # def step_t2a(self, action, parent_step_func):
    #     current_action_dim = action.shape[0]
    #     # === Phase 1: Design ===
    #     if self.design_step_counter < self.design_steps:
    #         if current_action_dim < self.design_dim:
    #             padded_design = np.zeros(self.design_dim, dtype=action.dtype)
    #             padded_design[:current_action_dim] = action
    #             design_act = padded_design
    #         else:
    #             design_act = action[:self.design_dim]

    #         self._apply_design(design_act)
    #         self.design_step_counter += 1
            
    #         obs = self.get_obs()
    #         info = {"time": self.sim.data.time, "phase": "design"}
    #         return obs, 0.0, False, False, info 

    #     # === Phase 2: Control ===
    #     else:
    #         if current_action_dim < self.control_dim:
    #             control_act = np.zeros(self.control_dim, dtype=action.dtype)
    #             control_act[:current_action_dim] = action
    #         else:
    #             control_act = action[:self.control_dim]
                
    #         return parent_step_func(control_act)
    def step_t2a(self, action, parent_step_func):
            # === 【关键修复】：处理 MyoSuite 底层 _setup 时的 "Dummy step" ===
            # MyoSuite 初始化时会硬编码调用 step(np.zeros(80)) 来探测环境
            if len(action) == self.control_dim:
                # 如果收到的是 80 维的探测动作，我们给它补齐到 320 维的全 0 数组
                # 这样设计参数全为 0，肌肉形态在探测步保持原始状态不变
                padded_action = np.zeros(self.total_action_dim, dtype=action.dtype)
                padded_action[:self.control_dim] = action
                action = padded_action
            # =============================================================
            # 确保输入动作的长度是我们期望的 320 (80 + 240)
            # 将 action 切分为两个独立的部分
            control_act = action[:self.control_dim]
            # 明确指定切片的终点，防止 TorchRL 传进来的 action 长度不对
            design_act = action[self.control_dim : self.control_dim + self.design_dim]
            
            # === Phase 1: Design ===
            if self.design_step_counter < self.design_steps:
                # 增加一个防御性报错，如果 TorchRL 还是传错维度，让它报错在源头
                if len(design_act) != self.design_dim:
                    raise ValueError(f"动作维度错误！期望 design_act 长度为 {self.design_dim}，但收到了 {len(design_act)}。请检查 TorchRL 是否正确读取了新的 action_space。")
                    
                self._apply_design(design_act)
                self.design_step_counter += 1
                
                obs = self.get_obs()
                
                info = {"time": self.sim.data.time, "phase": "design"}
                return obs, 0.0, False, False, info 

            # === Phase 2: Control ===
            else:
                return parent_step_func(control_act)
            
    def visualize_evolution(self, save_path="walk_full_evolution_heatmap.png", show=False):
        if self.current_scales is None: return
        data = self.current_scales 
        plt.figure(figsize=(10, 12)) 
        ax = sns.heatmap(data, 
                    yticklabels=self.muscle_names, 
                    xticklabels=["Strength", "Velocity", "Stiffness"],
                    cmap="vlag", center=1.0, annot=False) 
        plt.title(f"Multi-Parameter Evolution", fontsize=16)
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