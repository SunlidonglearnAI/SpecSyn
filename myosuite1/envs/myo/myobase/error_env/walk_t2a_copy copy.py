import numpy as np
import mujoco
from gymnasium import spaces
import matplotlib.pyplot as plt
import seaborn as sns
import os
import collections

# 导入原始环境
from myosuite.envs.myo.myobase.walk_v0 import WalkEnvV0, TerrainEnvV0

class T2AWalkMixin:
    """
    PSE-T2A: Physiological Synergy Evolution - Transform2Act
    将每块肌肉的进化参数压缩为 1 维 (Fiber Type Synergy rho)
    """
    def _setup_t2a(self, **kwargs):
        # 1. 观察空间配置
        obs_keys = kwargs.get('obs_keys', self.DEFAULT_OBS_KEYS)
        if "design_params" not in obs_keys:
            if isinstance(obs_keys, list): obs_keys.append("design_params")
            else: obs_keys = list(obs_keys) + ["design_params"]
        kwargs['obs_keys'] = obs_keys

        # 2. 识别肌肉
        self.muscle_indices = np.where(self.sim.model.actuator_dyntype == mujoco.mjtDyn.mjDYN_MUSCLE)[0]
        self.num_muscles = len(self.muscle_indices)
        
        # 获取肌肉名称
        # self.muscle_names = [self.sim.model.actuator_id2name(int(i)) or f"M{i}" for i in self.muscle_indices]
        self.muscle_names = [
            self.sim.model.actuator(int(i)).name or f"M{i}"
            for i in self.muscle_indices
        ]

        # 3. 备份原始参数
        self.original_gainprm = self.sim.model.actuator_gainprm[self.muscle_indices].copy()
        self.original_biasprm = self.sim.model.actuator_biasprm[self.muscle_indices].copy()

        # 4. T2A 状态
        self.design_steps = 1
        self.design_step_counter = 0
        self.design_dim = self.num_muscles * 1  # 降维：每块肌肉只有 1 个 rho
        self.control_dim = self.sim.model.nu
        
        # 初始化 rho (范围 -1 到 1, 0 代表原始状态)
        self.current_rho = np.zeros((self.num_muscles, 1), dtype=np.float32)
        self.f_scale = np.ones(self.num_muscles) # 用于计算代谢代价
        
        return kwargs

    def _fix_action_space(self):
        dim = max(self.control_dim, self.design_dim)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(dim,), dtype=np.float32)

    def _apply_design(self, design_act):
        """
        生理耦合映射: rho -> (Strength, Velocity, Stiffness)
        """
        rho = np.clip(design_act.reshape(self.num_muscles, 1), -1.0, 1.0)
        self.current_rho = rho

        
        # Sigmoid 映射函数: 将 [-1, 1] 映射到 [0, 1] 的平滑曲线
        # phi = 1 / (1 + exp(-k * rho))
        k_gain = 5.0 
        phi = 1.0 / (1.0 + np.exp(-k_gain * rho.flatten()))
        
        # 基于 phi 在 Slow 极限和 Fast 极限之间插值
        # 力量: 0.8 -> 1.5
        self.f_scale = 0.8 + (1.5 - 0.8) * phi
        # 速度: 0.5 -> 1.5
        v_scale = 0.5 + (1.5 - 0.5) * phi
        # 刚度: 0.8 -> 1.2
        k_scale = 0.8 + (1.2 - 0.8) * phi
        
        # # 生理映射方程 (可作为论文中的 Equation)
        # self.f_scale = 1.15 + 0.35 * rho.flatten()  # [0.8, 1.5]
        # v_scale = 1.0 + 0.5 * rho.flatten()         # [0.5, 1.5]
        # k_scale = 1.0 + 0.2 * rho.flatten()         # [0.8, 1.2]

        # 写入 MuJoCo 模型
        self.sim.model.actuator_gainprm[self.muscle_indices, 2] = self.original_gainprm[:, 2] * self.f_scale
        self.sim.model.actuator_gainprm[self.muscle_indices, 6] = self.original_gainprm[:, 6] * v_scale
        self.sim.model.actuator_biasprm[self.muscle_indices, 2] = self.original_biasprm[:, 2] * k_scale

    def get_reward_dict_t2a(self, obs_dict):
        """
        覆盖奖励函数，加入代谢代价惩罚
        """
        # 调用父类原始奖励逻辑
        rwd_dict = super().get_reward_dict(obs_dict)
        # print(f"DEBUG: obs_dict['act'] shape: {obs_dict['act'].shape}")
        # print(f"DEBUG: muscle_indices: {self.muscle_indices}")
        
        # 计算代谢惩罚: 消耗 = \sum (力量基数 * 激活度)
        if 'act' in obs_dict:
            muscle_act = np.abs(obs_dict['act'].squeeze()[self.muscle_indices])
            # 代谢权重可以在 0.1 到 1.0 之间调整
            metabolic_penalty = np.mean(self.f_scale * muscle_act)
            rwd_dict['metabolic_cost'] = -0.5 * metabolic_penalty
            
            # 重新计算 dense reward
            rwd_dict['dense'] += rwd_dict['metabolic_cost']
            
        return rwd_dict

    def step_t2a(self, action, parent_step_func):
        if self.design_step_counter < self.design_steps:
            # Design Phase
            design_act = action[:self.design_dim] if action.shape[0] >= self.design_dim else np.zeros(self.design_dim)
            self._apply_design(design_act)
            self.design_step_counter += 1
            return self.get_obs(), 0.0, False, False, {"phase": "design"}
        else:
            # Control Phase
            control_act = action[:self.control_dim] if action.shape[0] >= self.control_dim else action
            return parent_step_func(control_act)

    def reset_t2a(self):
        self.design_step_counter = 0
        self.current_rho = np.zeros((self.num_muscles, 1), dtype=np.float32)
        self.f_scale = np.ones(self.num_muscles)
        # 注意：这里不需要恢复模型参数，因为 apply_design 会在 reset 后的第一步重新计算

    def visualize_evolution(self, save_path="pse_t2a_heatmap.png"):
        plt.figure(figsize=(8, 12))
        sns.heatmap(self.current_rho, yticklabels=self.muscle_names, 
                    xticklabels=["Fiber Type (rho)"], cmap="RdBu_r", center=0)
        plt.title("Evolved Muscle Synergies (Slow to Fast-Twitch)")
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close()

# ===========================================================================
#  环境封装 (封装后只需训练这两个类)
# ===========================================================================
class WalkEnvT2A(T2AWalkMixin, WalkEnvV0):
    def _setup(self, **kwargs):
        kwargs = self._setup_t2a(**kwargs)
        super()._setup(**kwargs)
        self._fix_action_space()
    def reset(self, **kwargs):
        self.reset_t2a()
        return super().reset(**kwargs)
    def step(self, action):
        return self.step_t2a(action, super().step)
    def get_reward_dict(self, obs_dict):
        return self.get_reward_dict_t2a(obs_dict)
    def get_obs_dict(self, sim):
        obs_dict = super().get_obs_dict(sim)
        obs_dict["design_params"] = self.current_rho.flatten().copy()
        return obs_dict

class TerrainEnvT2A(T2AWalkMixin, TerrainEnvV0):
    def _setup(self, **kwargs):
        kwargs = self._setup_t2a(**kwargs)
        super()._setup(**kwargs)
        self._fix_action_space()
    def reset(self, **kwargs):
        self.reset_t2a()
        return super().reset(**kwargs)
    def step(self, action):
        return self.step_t2a(action, super().step)
    def get_reward_dict(self, obs_dict):
        return self.get_reward_dict_t2a(obs_dict)
    def get_obs_dict(self, sim):
        obs_dict = super().get_obs_dict(sim)
        obs_dict["design_params"] = self.current_rho.flatten().copy()
        return obs_dict