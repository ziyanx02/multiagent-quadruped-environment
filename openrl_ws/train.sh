python ./openrl_ws/train.py --headless --num_envs 1000\
    --task go1gate \
    --algo mat \
    --sim_device cuda:1 \
    --rl_device cuda:0 \
    --use_wandb \
    --exp_name test