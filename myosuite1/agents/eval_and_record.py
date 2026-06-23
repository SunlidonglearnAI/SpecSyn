import os
os.environ['MUJOCO_GL'] = 'egl'
os.environ['PYOPENGL_PLATFORM'] = 'egl'
import argparse
import numpy as np
import torch
import gymnasium as gym
import myosuite
import imageio
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
from stable_baselines3.common.env_util import make_vec_env


# 尝试导入 T2A 相关模块
# try:
#     import myosuite.envs.myo.myobase.walk_t2a1
#     import myosuite.envs.myo.myobase.obj_hold_t2a
#     print("SUCCESS: T2A Environment modules imported.")
# except ImportError:
#     print("WARNING: Could not import T2A modules.")
# except Exception as e:
#     print(f"ERROR: T2A import failed: {e}")

def generate_bio_radar_plot(env, save_path, title_text="T2A Evolution"):
    """
    针对博士论文优化的雷达图函数（修复多余维度显示问题）：
    1. 自动过滤未进化的维度（消除 1.0x 基准圆圈）。
    2. 兼容 (N,) 和 (N, 3) 维度数据。
    3. 强化 1.5x 刻度线为加粗实线。
    """
    print(f"DEBUG: Generating Adaptive Bio-Radar for {title_text}...")
    
    if not hasattr(env, 'muscle_names') or not hasattr(env, 'current_scales'):
        print("WARNING: Environment missing required attributes for plotting.")
        return

    muscle_names = env.muscle_names
    # 原始数据形状可能是 (N,) 或 (N, 3)
    raw_data = np.array(env.current_scales)
    
    # === 核心修复 1: 维度标准化与过滤 ===
    if raw_data.ndim == 1:
        data_to_plot = raw_data.reshape(-1, 1)
    else:
        data_to_plot = raw_data
        
    num_metrics = data_to_plot.shape[1]
    
    # 识别哪些维度发生了进化（不全等于 1.0 的列）
    active_indices = []
    for i in range(num_metrics):
        # 如果这一列的数据不全是 1.0，说明该参数被 T2A 策略修改过
        if not np.allclose(data_to_plot[:, i], 1.0, atol=1e-3):
            active_indices.append(i)
    
    # 如果是 Stiffness 环境但数据全是 1.0 (例如刚开始跑)，至少保留一列避免报错
    if len(active_indices) == 0:
        active_indices = [0] 

    # --- 数据分组逻辑 ---
    muscle_groups_def = {
        "Hip (髋部)": ["hip", "gluteus", "psoas"],
        "Knee (膝部)": ["quad", "hamstring", "rectus_fem"],
        "Ankle (踝部)": ["gastroc", "soleus", "tibialis"],
    }
    
    ordered_names, ordered_values, group_spans = [], [], []
    current_idx, used_indices = 0, set()

    for group_name, patterns in muscle_groups_def.items():
        start = current_idx
        found = False
        for i, m_name in enumerate(muscle_names):
            if i in used_indices: continue
            if any(pat in m_name.lower() for pat in patterns):
                ordered_names.append(m_name)
                ordered_values.append(data_to_plot[i])
                used_indices.add(i); found = True; current_idx += 1
        if found: group_spans.append((start, current_idx, group_name))

    for i, m_name in enumerate(muscle_names):
        if i not in used_indices:
            ordered_names.append(m_name); ordered_values.append(data_to_plot[i]); current_idx += 1
    if current_idx > start: group_spans.append((start, current_idx, "Others"))

    # --- 绘图配置 ---
    final_data = np.array(ordered_values)
    N = len(ordered_names)
    log_data = np.log2(final_data + 1e-6) 
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False) + np.pi / 2
    
    # 视觉参数
    BASE_R = 12.0       
    SCALE_FACTOR = 7.0  
    Y_MIN, Y_MAX = 2.0, 22.0
    
    # 论文级字号调优
    FONT_TITLE = 66
    FONT_GROUP = 52
    FONT_LEGEND = 54    # <--- 这里稍微调小了一点点
    FONT_REF = 38       

    fig = plt.figure(figsize=(22, 22), facecolor='white')
    ax = fig.add_subplot(111, projection='polar')
    ax.set_facecolor('white')
    ax.set_ylim(Y_MIN, Y_MAX)
    ax.grid(False)

    # 1. 绘制背景扇区
    for idx, (start, end, name) in enumerate(group_spans):
        mid_angle = (angles[start] + angles[end-1]) / 2
        ax.text(mid_angle, Y_MAX + 1.5, name, ha='center', va='center', 
                fontsize=FONT_GROUP, fontweight='bold', color='#222222')
        color = '#F5F5F5' if idx % 2 == 0 else '#FFFFFF'
        ax.bar(x=mid_angle, height=Y_MAX-Y_MIN, width=(angles[end-1]-angles[start]+0.1), 
               bottom=Y_MIN, color=color, edgecolor='none', zorder=0)

    # 2. 绘制参考刻度 (1.5x 加粗实线)
    scales = [(0.5, ':', 0.4, "0.5x", 2), (1.0, '--', 0.8, "1.0x (Base)", 3),
              (1.5, '-', 1.0, "1.5x", 6), (2.0, ':', 0.4, "2.0x", 2)]
    
    for val, style, alpha, label, lw in scales:
        r = BASE_R + np.log2(val) * SCALE_FACTOR
        ax.plot(np.linspace(0, 2*np.pi, 100), [r]*100, color='#444444', 
                linestyle=style, linewidth=lw, alpha=alpha, zorder=2)
        ax.text(np.pi/2, r + 0.6, label, color='#333333', ha='center', 
                fontsize=FONT_REF, fontweight='bold', bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))

    # 3. 绘制进化数据 (仅绘制活跃维度)
    all_colors = ['#E31A1C', '#33A02C', '#1F78B4'] # 对应 Force, Velocity, Stiffness
    all_labels = ['Force ($F_0$)', 'Velocity ($V_{max}$)', 'Stiffness ($K$)']
    
    for i in active_indices:
        current_col_log = log_data[:, i]
        radii = BASE_R + current_col_log * SCALE_FACTOR
        current_radii = np.clip(radii, Y_MIN, Y_MAX)
        
        color = all_colors[i % 3]
        label = all_labels[i % 3]
        
        # 连线
        for j in range(N):
            ax.plot([angles[j], angles[j]], [BASE_R, current_radii[j]], 
                    color=color, linewidth=2.5, alpha=0.2, zorder=3)
        # 散点
        ax.scatter(angles, current_radii, c=color, s=600, alpha=0.8, 
                   edgecolors='white', linewidth=2.0, label=label, zorder=4)

    # 4. 图例美化
    ax.set_yticklabels([]); ax.set_xticklabels([])
    leg = ax.legend(loc='lower right', bbox_to_anchor=(1.15, 0.05), 
                    ncol=1, frameon=True, fontsize=FONT_LEGEND, 
                    markerscale=1.4, facecolor='white', edgecolor='#CCCCCC')
    for text in leg.get_texts(): text.set_fontweight('bold')

    plt.title(title_text, fontsize=FONT_TITLE, pad=120, fontweight='bold')
    plt.savefig(save_path, dpi=300, facecolor='white', bbox_inches='tight')
    plt.close(fig) 
    print(f"SUCCESS: Plot updated and saved to {save_path}")

# ==============================================================================
#  Main Evaluation Logic
# ==============================================================================
def evaluate(args):
    print(f"\n{'='*50}")
    print(f"DEBUG: Starting evaluation for {args.env}")
    print(f"DEBUG: Model Directory: {args.model_dir}")
    print(f"{'='*50}\n")
    
    model_filename = f"{args.env}_{args.algo}_model.zip"
    env_stats_filename = f"{args.env}_{args.algo}_env.pkl"
    
    model_path = os.path.join(args.model_dir, model_filename)
    stats_path = os.path.join(args.model_dir, env_stats_filename)

    # 检查模型
    if not os.path.exists(model_path):
        if os.path.exists(os.path.join(args.model_dir, "best_model.zip")):
            model_path = os.path.join(args.model_dir, "best_model.zip")
            print("WARNING: Standard model name not found, using 'best_model.zip'")
        else:
            print(f"\n[CRITICAL ERROR] Model file NOT found!\nExpected path: {model_path}")
            return
    
    # 创建环境
    print(f"DEBUG: Creating environment: {args.env}")
    try:
        env = make_vec_env(args.env, n_envs=1)
        raw_env = env.envs[0].unwrapped
    except Exception as e:
        print(f"CRITICAL ERROR: Failed to create environment. Error: {e}")
        return

    # 加载 VecNormalize
    stats_path_no_ext = os.path.join(args.model_dir, f"{args.env}_{args.algo}_env")
    load_path = None
    if os.path.exists(stats_path): load_path = stats_path
    elif os.path.exists(stats_path_no_ext): load_path = stats_path_no_ext

    if load_path:
        print(f"DEBUG: Loading VecNormalize stats from {load_path}")
        env = VecNormalize.load(load_path, env)
        env.training = False     
        env.norm_reward = False  
    else:
        print(f"WARNING: No VecNormalize stats found. Running without normalization.")

    # 加载模型
    print(f"DEBUG: Loading model from {model_path}...")
    try:
        if args.algo == "PPO":
            model = PPO.load(model_path, env=env, device=args.device)
        elif args.algo == "SAC":
            model = SAC.load(model_path, env=env, device=args.device)
        else:
            print(f"ERROR: Unknown algorithm {args.algo}")
            return
    except Exception as e:
        print(f"CRITICAL ERROR: Failed to load model. Error: {e}")
        return

    # 推理循环
    print("DEBUG: Resetting environment...")
    import random
    first_seed = random.randint(0, 10000)
    env.seed(first_seed)  # 给环境打上随机种子
    obs = env.reset()
    
    frames = []
    total_reward = 0
    video_path = os.path.join(args.model_dir, f"eval_{args.env}.mp4")
    # 这里文件名改成 .png，因为 radar 是静态图
    radar_path = os.path.join(args.model_dir, f"radar_{args.env}.png")

    try:
        print(f"DEBUG: Starting loop for {args.steps} steps...")
        radar_plotted = False # 添加一个旗标，确保只画一次
        
        for i in range(args.steps):
            action, _ = model.predict(obs, deterministic=False)
            obs, reward, done, info = env.step(action)
            total_reward += reward[0]

            # === 调用本地的 Radar 绘图函数 ===
            # 只有当用户指定 --plot_evolution 且在第2步(进化刚完成)时才画图
            # if args.plot_evolution and i == 1:
            #     try:
            #         generate_bio_radar_plot(raw_env, radar_path, args.env)
            #     except Exception as e:
            #         print(f"WARNING: Plotting failed: {e}")
            #         import traceback
            #         traceback.print_exc()
            # === 兼容 PCA 与普通 T2A 的 Radar 绘图逻辑 ===
            if args.plot_evolution and not radar_plotted:
                target_env = raw_env
                # 递归查找真正的环境类
                while hasattr(target_env, 'env') and not hasattr(target_env, 'current_scales'):
                    target_env = target_env.env
                
                # 检查是否拿到了关键数据
                if hasattr(target_env, 'current_scales'):
                    # 【核心修复】：如果环境是 PCA 版，它可能没有 muscle_names，只有 muscle_indices
                    if not hasattr(target_env, 'muscle_names'):
                        print("DEBUG: PCA environment detected. Reconstructing muscle_names...")
                        # 从 MuJoCo 模型中实时提取肌肉名称
                        m_names = []
                        for idx in target_env.muscle_indices:
                            try:
                                name = target_env.sim.model.actuator_id2name(int(idx))
                            except:
                                name = f"M{idx}"
                            m_names.append(name)
                        target_env.muscle_names = m_names # 补齐属性

                    try:
                        generate_bio_radar_plot(target_env, radar_path, args.env)
                        radar_plotted = True
                    except Exception as e:
                        print(f"WARNING: Plotting failed: {e}")
                        radar_plotted = True
                elif i > 15:
                    print("WARNING: Could not find current_scales in environment.")
                    args.plot_evolution = False

            # 渲染图像
            if i % 2 == 0: 
                try:
                    if hasattr(raw_env, 'sim'):
                        frame = raw_env.sim.renderer.render_offscreen(
                            width=args.width,
                            height=args.height,
                            camera_id=args.camera_id
                        )
                        frames.append(frame)
                except Exception as e:
                    if i == 0: print(f"Render Error (ignoring subsequent frames): {e}")

            if done[0]:
                print(f"Episode finished at step {i+1}. Total Reward: {total_reward:.4f}")
                
                # 生成新的随机种子并应用
                # new_seed = random.randint(0, 10000)
                # env.seed(new_seed)
                # obs = env.reset()
                total_reward = 0
                if not args.loop:
                    break
                    
        print(f"DEBUG: Loop finished. Captured {len(frames)} frames.")
                    
    except KeyboardInterrupt:
        print("Interrupted by user.")
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"CRITICAL ERROR: {e}")
    
    env.close()
    # 如果使用了 raw_env.sim.renderer，手动关闭它
    if hasattr(raw_env, 'sim') and hasattr(raw_env.sim, 'renderer'):
        try:
            raw_env.sim.renderer.close() 
            print("DEBUG: Renderer closed safely.")
        except:
            pass

    if frames:
        print(f"DEBUG: Saving video to {video_path}...")
        try:
            imageio.mimsave(video_path, frames, fps=30)
            print("DONE: Video saved successfully!")
        except Exception as e:
            print(f"ERROR saving video: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="myoLegHillyTerrainWalk-v0", help="Environment ID")
    parser.add_argument("--algo", type=str, default="PPO", help="Algorithm (PPO or SAC)")
    parser.add_argument("--model_dir", type=str, default="outputs/walk_ori/hilly/ori1", help="Directory containing the .zip model")
    parser.add_argument("--steps", type=int, default=1000, help="Number of steps to evaluate")
    parser.add_argument("--camera_id", type=int, default=0, help="Camera ID")
    parser.add_argument("--width", type=int, default=640, help="Render width")
    parser.add_argument("--height", type=int, default=480, help="Render height")
    parser.add_argument("--device", type=str, default="cpu", help="Device")
    parser.add_argument("--loop", default=True, action="store_true", help="Loop evaluation")
    parser.add_argument("--plot_evolution", default=True, action="store_true", help="Plot T2A Bio-Radar Chart")
    
    args = parser.parse_args()
    evaluate(args)