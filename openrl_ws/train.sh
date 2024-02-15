python ./openrl_ws/train.py --headless --num_envs 500 --train_timesteps 30000000\
    --task go1gate \
    --algo ppo \
    --sim_device cuda:0 \
    --rl_device cuda:0 \
    --seed 1 \
    # --use_wandb \
    # --exp_name test