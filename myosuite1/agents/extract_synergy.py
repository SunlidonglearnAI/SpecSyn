import os
os.environ['MUJOCO_GL'] = 'egl' 

import gymnasium as gym
import myosuite
import numpy as np
import collections
from sklearn.decomposition import PCA
import sys

def extract_synergy_basis(env_id='myoLegWalk-v0', steps=5000, latent_k=5):
    print(f"=== 启动协同矩阵提取程序 ===", flush=True)
    
    try:
        # 1. 初始化基础环境
        print(f"正在尝试创建环境: {env_id}...", flush=True)
        env = gym.make(env_id)
        print("环境创建成功，正在重置...", flush=True)
        env.reset()
        print("环境重置完成。", flush=True)
        
        # 2. 识别肌肉名称
        muscle_names = []
        for i in range(env.unwrapped.sim.model.nu):
            name = env.unwrapped.sim.model.actuator(i).name
            muscle_names.append(name if name else f"M{i}")
            
        muscle_groups = collections.OrderedDict()
        for idx, name in enumerate(muscle_names):
            base_name = name[:-2] if (name.endswith('_l') or name.endswith('_r')) else name
            if base_name not in muscle_groups:
                muscle_groups[base_name] = []
            muscle_groups[base_name].append(idx)
            
        group_names = list(muscle_groups.keys())
        num_groups = len(group_names)
        print(f"找到 {len(muscle_names)} 块肌肉，映射为 {num_groups} 个对称组。", flush=True)

        # 3. 收集动态数据
        group_length_history = np.zeros((steps, num_groups))
        print(f"开始数据采集 (预计执行 {steps} 步)...", flush=True)
        
        for t in range(steps):
            action = env.action_space.sample()
            env.step(action)
            
            actuator_lengths = env.unwrapped.sim.data.actuator_length
            for i, base_name in enumerate(group_names):
                indices = muscle_groups[base_name]
                group_length_history[t, i] = np.mean(actuator_lengths[indices])
                
            if (t + 1) % 500 == 0:
                print(f"已完成 {t+1}/{steps} 步...", flush=True)
                env.reset()
                
        env.close()

        # 4. PCA
        print("正在进行 PCA 降维...", flush=True)
        normalized_data = (group_length_history - np.mean(group_length_history, axis=0)) / (np.std(group_length_history, axis=0) + 1e-8)
        pca = PCA(n_components=latent_k)
        pca.fit(normalized_data)
        
        W_basis = pca.components_.T
        np.save("synergy_W_basis.npy", W_basis)
        
        explained_var = np.sum(pca.explained_variance_ratio_) * 100
        print(f"【成功】提取完成！解释方差: {explained_var:.2f}%", flush=True)
        print(f"矩阵已保存至: {os.path.abspath('synergy_W_basis.npy')}", flush=True)

    except Exception as e:
        print(f"【运行出错】错误类型: {type(e).__name__}", flush=True)
        print(f"错误信息: {e}", flush=True)
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    print("Python 脚本主程序启动...", flush=True)
    extract_synergy_basis()