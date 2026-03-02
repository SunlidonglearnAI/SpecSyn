export MUJOCO_GL=egl
python eval_and_record.py \
    --env myoLegRoughTerrainWalk-v0 \
    --model_dir /home/fzh/Workspace/T2A_symmetry/muscle/myosuite/myosuite/agents/outputs/2026-02-04/10-07-39 \
    --algo PPO \
    --steps 2000 \
    --
    
python eval_and_record.py    --env myoLegRoughTerrainWalk-v0    --model_dir /home/fzh/Workspace/T2A_symmetry/muscle/myosuite/myosuite/agents/outputs/Walk    --algo PPO    --steps 1000    --loop