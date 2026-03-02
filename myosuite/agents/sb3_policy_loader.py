import os
from stable_baselines3 import PPO
import gymnasium as gym

# 模型文件的绝对路径 (不带.zip后缀)
MODEL_PATH = "/data/fzh/Workspace/T2A_symmetry/muscle/myosuite/myosuite/agents/outputs/2026-02-03/22-41-00/myoLegRoughTerrainT2A-v0_PPO_model"
ENV_ID = "myoLegRoughTerrainT2A-v0"

# 这是一个包装类，用于将 SB3 模型适配到 examine_env.py 期望的 get_action 接口
class SB3PolicyLoader:
    def __init__(self, env, seed=None):
        # 这里的 env 参数是 examine_env.py 传递进来的环境实例
        # 我们需要在加载模型时，确保 custom_objects={"env": env} 能够处理好。
        # 如果模型训练时使用了 VecNormalize，那么需要确保加载时环境也被正确归一化。

        # PPO.load 期望传入的环境是一个 VecEnv，或者 custom_objects 中包含 env
        # 但 examine_env.py 传入的是一个原始的 gym.Env
        # 因此，我们需要确保 PPO.load 能够处理这种不匹配，或者在加载时提供一个临时的 DummyVecEnv

        # 尝试使用传入的 env 来加载模型
        try:
            self.model = PPO.load(str(MODEL_PATH), custom_objects={"env": env})
            print(f"SB3PolicyLoader: Successfully loaded model from {MODEL_PATH}.zip with custom_objects={{'env': env}}")
        except Exception as e:
            print(f"SB3PolicyLoader: Error loading model {MODEL_PATH}.zip with custom_objects={{'env': env}}: {e}")
            print(f"SB3PolicyLoader: Retrying to load model {MODEL_PATH}.zip without custom_objects...")
            # 尝试不带 custom_objects
            try:
                self.model = PPO.load(str(MODEL_PATH))
                print(f"SB3PolicyLoader: Successfully loaded model from {MODEL_PATH}.zip without custom_objects.")
            except Exception as e_no_custom_obj:
                print(f"SB3PolicyLoader: Critical error loading model {MODEL_PATH}.zip: {e_no_custom_obj}")
                raise e_no_custom_obj

    def get_action(self, obs):
        # examine_policy 传递的 obs 可能是 (obs, info) 元组
        # 检查并提取实际的观测值
        if isinstance(obs, tuple):
            obs = obs[0]
        
        # SB3 模型返回 (action, _states)
        action, _states = self.model.predict(obs, deterministic=True)
        # examine_policy 期望 get_action 返回 (action_exploration, action_evaluation)
        # 这里我们假设 deterministic=True 的 action 就是 evaluation action
        return action, {'evaluation': action}
