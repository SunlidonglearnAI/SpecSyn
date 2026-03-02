# import os
# import argparse
# import numpy as np
# import torch

# import gym
# import myosuite

# import imageio

# from stable_baselines3 import PPO, SAC
# from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
# from stable_baselines3.common.env_util import make_vec_env

# # 尝试导入 T2A
# try:
#     import myosuite.envs.myo.myobase.walk_t2a
#     print("SUCCESS: T2A Environment module imported.")
# except ImportError:
#     print("WARNING: Could not import walk_t2a (Ignore this if using standard env).")
# except Exception as e:
#     print(f"ERROR: T2A import failed with error: {e}")

# def evaluate(args):
    
#     # 1. 创建环境
#     env = make_vec_env(args.env, n_envs=1)
    
#     # 2. 处理 VecNormalize
#     stats_path = os.path.join(args.model_dir, f"{args.env}_{args.algo}_env")
    
#     # SB3 有时会保存为 .pkl，有时没有后缀，我们都试一下
#     if not os.path.exists(stats_path) and os.path.exists(stats_path + ".pkl"):
#         stats_path += ".pkl"

#     if os.path.exists(stats_path):
#         env = VecNormalize.load(stats_path, env)
#         env.training = False
#         env.norm_reward = False
#     else:
#         print(f"WARNING: No VecNormalize stats found at {stats_path}. Running without normalization.")

#     # 3. 加载模型
#     model_path = os.path.join(args.model_dir, f"{args.env}_{args.algo}_model.zip")
    
#     if not os.path.exists(model_path):
#         print(f"CRITICAL ERROR: Model file not found at {model_path}")
#         return

#     if args.algo == "PPO":
#         model = PPO.load(model_path, env=env, device=args.device)
#     elif args.algo == "SAC":
#         model = SAC.load(model_path, env=env, device=args.device)
#     print("DEBUG: Model loaded successfully.")

#     # 4. 推理
#     print("DEBUG: Resetting environment...")
#     obs = env.reset()
#     frames = []
#     total_reward = 0
    
#     print(f"DEBUG: Starting loop for {args.steps} steps...")
    
#     try:
#         for i in range(args.steps):
#             action, _ = model.predict(obs, deterministic=True)
#             obs, reward, done, info = env.step(action)
#             total_reward += reward[0]

#             # 渲染
#             # print(f"DEBUG: Rendering frame {i}...") # 这行太吵，只在第一帧打印
#             if i == 0: print("DEBUG: First step taken. Attempting render...")
            
#             raw_env = env.envs[0].unwrapped
#             frame = raw_env.sim.renderer.render_offscreen(
#                 width=args.width,
#                 height=args.height,
#                 camera_id=args.camera_id
#             )
#             frames.append(frame)

#             if done[0]:
#                 print(f"Episode finished at step {i+1}. Total Reward: {total_reward}")
#                 obs = env.reset()
#                 total_reward = 0
#                 if not args.loop:
#                     break
#         print(f"DEBUG: Loop finished. Captured {len(frames)} frames.")
                    
#     except KeyboardInterrupt:
#         print("Interrupted by user.")
#     except Exception as e:
#         print(f"CRITICAL ERROR during loop: {e}")
    
#     env.close()

#     # 5. 保存
#     if frames:
#         output_path = os.path.join(args.model_dir, f"eval_{args.env}.mp4")
#         print(f"DEBUG: Saving video to {output_path}...")
#         imageio.mimsave(output_path, frames, fps=30)
#         print("DONE: Video saved successfully!")
#     else:
#         print("ERROR: No frames captured.")

# if __name__ == "__main__":
#     print("DEBUG: Entering Main Block")
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--env", type=str, required=True)
#     parser.add_argument("--algo", type=str, default="PPO")
#     parser.add_argument("--model_dir", type=str, default=".")
#     parser.add_argument("--steps", type=int, default=1000)
#     parser.add_argument("--camera_id", type=int, default=-1)
#     parser.add_argument("--width", type=int, default=640)
#     parser.add_argument("--height", type=int, default=480)
#     parser.add_argument("--device", type=str, default="cpu")
#     parser.add_argument("--loop", action="store_true")
    
#     args = parser.parse_args()
#     print(f"DEBUG: Arguments parsed. Env: {args.env}, Model Dir: {args.model_dir}")
#     evaluate(args)

# export MUJOCO_GL=egl
# python eval_and_record.py --env myoLegRoughTerrainT2A1-v0  --model_dir /home/fzh/Workspace/T2A_symmetry/muscle/myosuite/myosuite/agents/outputs/2026-02-10/21-46-20 --algo PPO --steps 1000 --loop --plot_evolution
import os
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
try:
    import myosuite.envs.myo.myobase.walk_t2a1
    import myosuite.envs.myo.myobase.obj_hold_t2a
    print("SUCCESS: T2A Environment modules imported.")
except ImportError:
    print("WARNING: Could not import T2A modules.")
except Exception as e:
    print(f"ERROR: T2A import failed: {e}")

# ==============================================================================
#  Bio-Radar Visualization Function (Local Implementation)
# ==============================================================================
def generate_bio_radar_plot(env, save_path, title_text="T2A Evolution"):
    """
    Reads T2A data from the environment and generates a Bright Bio-Radar Chart.
    """
    print("DEBUG: Generating Bright Bio-Radar Visualization...")
    
    # 1. 获取环境数据
    if not hasattr(env, 'muscle_names') or not hasattr(env, 'current_scales'):
        print("WARNING: Environment does not have T2A attributes (muscle_names/current_scales). Skipping plot.")
        return

    muscle_names = env.muscle_names
    current_scales = env.current_scales # Shape (N, 3) -> [F0, L_opt, Stiffness]

    # 2. 定义解剖学分组 (用于排序和背景色块)
    # 根据 MyoHand 的常用肌肉名定义
    # muscle_groups_def = {
    #     "Wrist": ["ECRL", "ECRB", "ECU", "FCR", "FCU", "PL", "PT", "PQ"],
    #     "Thumb": ["EPL", "EPB", "FPL", "APL", "OP"],
    #     "Flexors": ["FDS", "FDP"], # 包含 FDS2-5, FDP2-5
    #     "Extensors": ["EDC", "EDM", "EIP"], # 包含 EDC2-5
    #     "Intrinsics": ["RI", "LU", "UI", "DI", "PI"] # 手内肌
    # }
    muscle_groups_def = {
        "Hip (髋部)": ["hip", "gluteus", "psoas"],
        "Knee (膝部)": ["quad", "hamstring", "rectus_fem"],
        "Ankle (踝部)": ["gastroc", "soleus", "tibialis"],
    }

    # 3. 数据重组与排序
    ordered_names = []
    ordered_data = []
    group_spans = [] 
    
    current_idx = 0
    used_indices = set()

    # 遍历定义的组，从 env 数据中通过名字匹配找数据
    for group_name, patterns in muscle_groups_def.items():
        start = current_idx
        found_in_group = False
        
        # 查找属于该组的所有肌肉
        # 注意：这里我们遍历环境里的所有肌肉，看它是否匹配当前组的 pattern
        # 这种两层循环是为了保持组的顺序，同时找到所有匹配项
        temp_group_items = []
        
        for i, m_name in enumerate(muscle_names):
            if i in used_indices: continue
            
            # 检查匹配
            for pat in patterns:
                if pat in m_name: # Substring match
                    temp_group_items.append((m_name, current_scales[i]))
                    used_indices.add(i)
                    found_in_group = True
                    break
        
        # 将找到的肌肉加入有序列表
        for name, data in temp_group_items:
            ordered_names.append(name)
            ordered_data.append(data)
            current_idx += 1
            
        if found_in_group:
            group_spans.append((start, current_idx, group_name))

    # 处理未分类的肌肉 (如果有的话)
    start = current_idx
    leftover = False
    for i, m_name in enumerate(muscle_names):
        if i not in used_indices:
            ordered_names.append(m_name)
            ordered_data.append(current_scales[i])
            current_idx += 1
            leftover = True
    if leftover:
        group_spans.append((start, current_idx, "Others"))

    if not ordered_data:
        print("WARNING: No matching muscles found for visualization.")
        return

    data = np.array(ordered_data) # Shape (N, 3)
    N = len(ordered_names)

    # === Log2 转换 ===
    # 将 Scale 转换为对称的 Log Space
    # 1.0 -> 0; 2.0 -> 1; 0.5 -> -1
    # 这样在雷达图上，增强和减弱的视觉距离是相等的
    log_data = np.log2(data + 1e-6) 

    # 4. 绘图设置
    try:
        plt.style.use('seaborn-v0_8-whitegrid')
    except:
        plt.style.use('default') # Fallback

    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    angles += np.pi / 2 
    width = (2 * np.pi / N) * 0.8 

    fig = plt.figure(figsize=(14, 14), facecolor='white')
    ax = fig.add_subplot(111, projection='polar')
    ax.set_facecolor('white')
    ax.grid(False)

    # 配色: F0(红), Length(绿), Stiffness(蓝)
    colors = ['#D70040', '#009E73', '#0072B2'] 
    labels = ['Force ($F_0$)', 'Length ($L_{opt}$)', 'Stiffness']
    
    BASE_R = 6.0 # 基准半径

    # 绘制背景扇区
    bg_colors = ['#F8F8F8', '#FFFFFF']
    for idx, (start, end, name) in enumerate(group_spans):
        start_angle = angles[start] - width/1.5
        end_angle = angles[end-1] + width/1.5
        if start_angle < 0: start_angle += 2*np.pi
        
        # 背景色块
        color = bg_colors[idx % 2]
        ax.bar(x=(start_angle + end_angle)/2, height=15, width=(end_angle - start_angle), 
               bottom=0, color=color, edgecolor='none', zorder=0)
        
        # 组名标签
        mid_angle = (angles[start] + angles[end-1]) / 2
        ax.text(mid_angle, 14.2, name, color='#333333', ha='center', va='center', fontsize=14, fontweight='bold')
        
        # 分隔线
        if idx < len(group_spans) - 1:
            sep_angle = angles[end-1] + (angles[(end)%N] - angles[end-1])/2
            ax.plot([sep_angle, sep_angle], [1.5, 13.5], color='#DDDDDD', linestyle='-', linewidth=1.5, zorder=1)

    # 绘制基准线
    ax.plot(np.linspace(0, 2*np.pi, 100), [BASE_R]*100, color='#555555', linestyle='--', linewidth=1.5, alpha=0.7, zorder=2)
    ax.text(0, BASE_R + 0.5, "Original (1.0x)", color='#333333', ha='center', fontsize=10, fontweight='bold')
    
    # 绘制参考环 (2.0x 和 0.5x)
    # log2(2.0) = 1.0 -> R = Base + 2.5
    # log2(0.5) = -1.0 -> R = Base - 2.5
    ax.plot(np.linspace(0, 2*np.pi, 100), [BASE_R + 2.5]*100, color='#999999', linestyle=':', linewidth=1, alpha=0.5, zorder=2)
    ax.text(np.pi/2, BASE_R + 2.8, "2.0x", color='#777777', fontsize=9, ha='center')
    ax.plot(np.linspace(0, 2*np.pi, 100), [BASE_R - 2.5]*100, color='#999999', linestyle=':', linewidth=1, alpha=0.5, zorder=2)
    ax.text(np.pi/2, BASE_R - 3.2, "0.5x", color='#777777', fontsize=9, ha='center')

    # 5. 绘制数据点
    for i in range(3):
        # 映射半径: R = Base + log_val * scale_factor
        radii = BASE_R + log_data[:, i] * 2.5 
        
        for j in range(N):
            # 连线
            ax.plot([angles[j], angles[j]], [BASE_R, radii[j]], color=colors[i], linewidth=1.5, alpha=0.3, zorder=3)
        
        # 散点
        ax.scatter(angles, radii, c=colors[i], s=70, alpha=0.9, edgecolors='white', linewidth=0.5, label=labels[i], zorder=4)

    # 6. 标签与美化
    ax.set_ylim(0, 15)
    ax.set_yticklabels([])
    ax.set_xticklabels([])

    for j in range(N):
        angle = angles[j]
        name = ordered_names[j]
        # 文字旋转逻辑
        rotation = angle * 180 / np.pi - 90
        if 90 < abs(rotation) < 270: rotation += 180
        
        ax.text(angle, 1.2, name, color='#555555', fontsize=8, ha='center', va='center', rotation=rotation)

    # 2. 将图例位置稍微调低一点，避免撞到新标题
    plt.legend(loc='lower right', bbox_to_anchor=(1.15, 0.0), frameon=False, labelcolor='black', fontsize=12)
    
    # 3. [核心修改] 使用传入的 title_text (环境名) 作为大标题
    plt.title(title_text, color='black', fontsize=20, pad=40, fontweight='bold')
    
    # 4. 把原来的标题作为副标题放在下面
    # plt.figtext(0.5, 0.93, "Morphological Evolution Spectrum", color='#555555', ha='center', fontsize=12)
    # 保存
    save_path = os.path.abspath(save_path)
    plt.savefig(save_path, dpi=300, facecolor='white', bbox_inches='tight')
    print(f"SUCCESS: Bio-Radar Plot saved to: {save_path}")
    plt.close()


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
    obs = env.reset()
    
    frames = []
    total_reward = 0
    video_path = os.path.join(args.model_dir, f"eval_{args.env}.mp4")
    # 这里文件名改成 .png，因为 radar 是静态图
    radar_path = os.path.join(args.model_dir, f"radar_{args.env}.png")

    try:
        print(f"DEBUG: Starting loop for {args.steps} steps...")
        
        for i in range(args.steps):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            total_reward += reward[0]

            # === 调用本地的 Radar 绘图函数 ===
            # 只有当用户指定 --plot_evolution 且在第2步(进化刚完成)时才画图
            if args.plot_evolution and i == 1:
                try:
                    generate_bio_radar_plot(raw_env, radar_path, args.env)
                except Exception as e:
                    print(f"WARNING: Plotting failed: {e}")
                    import traceback
                    traceback.print_exc()

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
                obs = env.reset()
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

    if frames:
        print(f"DEBUG: Saving video to {video_path}...")
        try:
            imageio.mimsave(video_path, frames, fps=30)
            print("DONE: Video saved successfully!")
        except Exception as e:
            print(f"ERROR saving video: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, required=True, help="Environment ID")
    parser.add_argument("--algo", type=str, default="PPO", help="Algorithm (PPO or SAC)")
    parser.add_argument("--model_dir", type=str, default=".", help="Directory containing the .zip model")
    parser.add_argument("--steps", type=int, default=1000, help="Number of steps to evaluate")
    parser.add_argument("--camera_id", type=int, default=-1, help="Camera ID")
    parser.add_argument("--width", type=int, default=640, help="Render width")
    parser.add_argument("--height", type=int, default=480, help="Render height")
    parser.add_argument("--device", type=str, default="cpu", help="Device")
    parser.add_argument("--loop", action="store_true", help="Loop evaluation")
    parser.add_argument("--plot_evolution", action="store_true", help="Plot T2A Bio-Radar Chart")
    
    args = parser.parse_args()
    evaluate(args)