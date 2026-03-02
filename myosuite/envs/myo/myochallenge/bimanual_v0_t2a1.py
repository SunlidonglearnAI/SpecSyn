import numpy as np
import mujoco
from gymnasium import spaces
import matplotlib.pyplot as plt
import seaborn as sns
import collections

# 导入原始环境
from myosuite.envs.myo.myochallenge.bimanual_v0 import BimanualEnvV1

class T2ABimanualMixin:
    """
    Mixin for Evolution of Muscle Strength (F0) in Bimanual Task.
    """
    def _setup_t2a(self, **kwargs):
        # 1. [T2A 观测配置]
        obs_keys = kwargs.get('obs_keys', self.DEFAULT_OBS_KEYS)
        if "design_params" not in obs_keys:
            if isinstance(obs_keys, tuple): obs_keys = list(obs_keys)
            obs_keys.append("design_params")
        kwargs['obs_keys'] = obs_keys

        # 2. [识别双臂肌肉]
        # 注意：Bimanual 环境包含肌肉(MyoHand)和非肌肉(Prosthesis)执行器
        self.muscle_indices = np.where(self.sim.model.actuator_dyntype == mujoco.mjtDyn.mjDYN_MUSCLE)[0]
        self.num_muscles = len(self.muscle_indices)
        
        # 获取肌肉名称用于可视化
        self.muscle_names = []
        for i in self.muscle_indices:
            try: name = self.sim.model.actuator_id2name(int(i))
            except: name = f"M{i}"
            self.muscle_names.append(name)

        # 3. [备份原始肌肉强度 F0]
        # gainprm 索引 2 对应肌肉的峰值主动力 (Peak active force)
        self.original_gainprm = self.sim.model.actuator_gainprm[self.muscle_indices].copy()

        # 4. [设计阶段配置]
        self.design_steps = 1          
        self.design_step_counter = 0   
        
        # 进化维度：每块肌肉 1 个参数 (Strength)
        self.design_dim = self.num_muscles * 1 
        self.control_dim = self.sim.model.nu 
        
        # 初始化缩放比例 (1.0 代表不改变)
        self.current_scales = np.ones((self.num_muscles, 1), dtype=np.float32)
        
        print(f"DEBUG: T2A Bimanual Setup - Muscles: {self.num_muscles}, Design Dim: {self.design_dim}")
        return kwargs

    def _fix_action_space(self):
        # 确保动作空间涵盖控制和进化两个阶段的维度
        dim = max(self.control_dim, self.design_dim)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(dim,), dtype=np.float32)

    def _apply_design(self, design_act):
        """
        将策略动作映射到物理参数。
        动作 [-1, 1] -> 强度缩放 [0.8, 1.5]
        """
        design_act = np.clip(design_act, -1.0, 1.0)
        
        # 取前 num_muscles 个动作作为强度进化参数
        design_params = design_act[:self.num_muscles]
        
        # 映射逻辑 (与 Walk 保持一致)
        f_scale = np.where(
            design_params > 0,
            1.0 + 0.5 * design_params,  # 0 -> 1.0, 1 -> 1.5
            1.0 + 0.2 * design_params   # -1 -> 0.8, 0 -> 1.0
        )
        
        # 更新 MuJoCo 模型
        self.sim.model.actuator_gainprm[self.muscle_indices, 2] = self.original_gainprm[:, 2] * f_scale
        
        # 存储当前状态
        self.current_scales = f_scale.reshape(-1, 1)

    def reset_t2a(self):
        # 重置物理模型参数
        self.sim.model.actuator_gainprm[self.muscle_indices] = self.original_gainprm.copy()
        self.current_scales = np.ones((self.num_muscles, 1), dtype=np.float32)
        self.design_step_counter = 0

    def get_obs_t2a(self, obs_dict):
        # 将设计参数暴露给观测空间
        obs_dict["design_params"] = self.current_scales.flatten().copy()
        return obs_dict

    def step_t2a(self, action, parent_step_func):
        if self.design_step_counter < self.design_steps:
            # 阶段 1: 肌肉进化
            design_act = action[:self.design_dim] if action.shape[0] >= self.design_dim else np.zeros(self.design_dim)
            self._apply_design(design_act)
            self.design_step_counter += 1
            # 返回初始观测，0奖励，不结束
            return self.get_obs(), 0.0, False, False, {"phase": "design"} 
        else:
            # 阶段 2: 正常操作
            # 确保动作维度正确处理（截取控制部分）
            control_act = action[:self.control_dim] if action.shape[0] >= self.control_dim else np.zeros(self.control_dim)
            return parent_step_func(control_act)

    def visualize_evolution(self, save_path="bimanual_strength_evolution.png", show=False):
        if self.current_scales is None: return
        plt.figure(figsize=(8, 15)) 
        sns.heatmap(self.current_scales, 
                    yticklabels=self.muscle_names, 
                    xticklabels=["Strength (F0)"],
                    cmap="YlOrRd", center=1.0, annot=True, fmt=".2f") 
        plt.title("Bimanual Muscle Strength Evolution")
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        if show: plt.show()
        plt.close()

class BimanualEnvT2A(T2ABimanualMixin, BimanualEnvV1):
    def _setup(self, **kwargs):
        # 先运行 T2A 设置，再运行基类设置
        kwargs = self._setup_t2a(**kwargs)
        super()._setup(**kwargs)
        self._fix_action_space()

    def reset(self, **kwargs):
        self.reset_t2a()
        return super().reset(**kwargs)

    def get_obs_dict(self, sim):
        # 包装原始观测，加入 design_params
        return self.get_obs_t2a(super().get_obs_dict(sim))

    def step(self, action):
        # 包装原始 step，分阶段处理
        return self.step_t2a(action, super().step)