import os
os.environ['MUJOCO_GL'] = 'egl' 

import gymnasium as gym
import myosuite
import numpy as np
import collections
from sklearn.decomposition import PCA
import sys

def extract_synergy_basis_asym(env_id='myoLegWalk-v0', steps=5000, latent_k=5):
    print(f"=== [消融实验] 启动非对称协同矩阵提取程序 ===", flush=True)
    
    try:
        # 1. 初始化基础环境
        print(f"正在尝试创建环境: {env_id}...", flush=True)
        env = gym.make(env_id)
        env.reset()
        print("环境重置完成。", flush=True)
        
        # 2. 识别肌肉名称 (取消对称分组)
        muscle_names = []
        model = env.unwrapped.sim.model
        for i in range(model.nu):
            try:
                name = model.actuator(i).name
            except AttributeError:
                name = model.actuator_id2name(i)
            muscle_names.append(name if name else f"M{i}")
            
        # --- 修改点：不再剥离 _l/_r，每一块肌肉独立为一个组 ---
        muscle_groups = collections.OrderedDict()
        for idx, name in enumerate(muscle_names):
            # 直接使用全名，不进行后缀裁剪
            muscle_groups[name] = [idx] 
            
        group_names = list(muscle_groups.keys())
        num_groups = len(group_names)
        print(f"找到 {len(muscle_names)} 块独立肌肉，已取消对称映射 (N={num_groups})。", flush=True)

        # 3. 收集动态数据
        group_length_history = np.zeros((steps, num_groups))
        print(f"开始数据采集 (预计执行 {steps} 步)...", flush=True)
        
        for t in range(steps):
            action = env.action_space.sample() # 此时采样是全随机的，左右腿动作不相关
            env.step(action)
            
            actuator_lengths = env.unwrapped.sim.data.actuator_length
            for i, name in enumerate(group_names):
                idx = muscle_groups[name][0]
                group_length_history[t, i] = actuator_lengths[idx]
                
            if (t + 1) % 500 == 0:
                print(f"已完成 {t+1}/{steps} 步...", flush=True)
                env.reset()
                
        env.close()

        # 4. PCA
        print("正在进行 PCA 降维 (非对称空间)...", flush=True)
        normalized_data = (group_length_history - np.mean(group_length_history, axis=0)) / (np.std(group_length_history, axis=0) + 1e-8)
        pca = PCA(n_components=latent_k)
        pca.fit(normalized_data)
        
        W_basis = pca.components_.T # Shape 将变为 (80, latent_k)
        
        # 保存为不同的文件名，防止覆盖
        save_name = "synergy_W_basis_asym.npy"
        np.save(save_name, W_basis)
        
        explained_var = np.sum(pca.explained_variance_ratio_) * 100
        print(f"--- 提取完成 ---")
        print(f"解释方差: {explained_var:.2f}%")
        print(f"矩阵形状: {W_basis.shape} (应为 80 x {latent_k})")
        print(f"非对称矩阵已保存至: {os.path.abspath(save_name)}")

    except Exception as e:
        print(f"【运行出错】: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    extract_synergy_basis_asym()