
task="go1seesaw"
num_envs=500
num_steps=40000000
random_seed=0
algo="jrpo"
device=1
cfg=./openrl_ws/cfgs/ppo.yaml

python ./openrl_ws/train.py --headless --num_envs $num_envs --train_timesteps $num_steps\
    --task $task \
    --algo $algo \
    --sim_device cuda:$device \
    --rl_device cuda:$device \
    --seed 0 \
    --exp_name test \
    --config $cfg \
    --use_wandb

python ./openrl_ws/train.py --headless --num_envs $num_envs --train_timesteps $num_steps\
    --task $task \
    --algo $algo \
    --sim_device cuda:$device \
    --rl_device cuda:$device \
    --seed 1 \
    --exp_name test \
    --config $cfg \
    --use_wandb

# python ./openrl_ws/train.py --headless --num_envs $num_envs --train_timesteps $num_steps\
#     --task $task \
#     --algo $algo \
#     --sim_device cuda:$device \
#     --rl_device cuda:$device \
#     --seed 2 \
#     --exp_name collect_data \
#     --config $cfg \
#     --use_wandb

# python ./openrl_ws/train.py --headless --num_envs $num_envs --train_timesteps $num_steps\
#     --task $task \
#     --algo $algo \
#     --sim_device cuda:$device \
#     --rl_device cuda:$device \
#     --seed 3 \
#     --exp_name collect_data \
#     --config $cfg \
#     --use_wandb

# python ./openrl_ws/train.py --headless --num_envs $num_envs --train_timesteps $num_steps\
#     --task $task \
#     --algo $algo \
#     --sim_device cuda:$device \
#     --rl_device cuda:$device \
#     --seed 4 \
#     --exp_name collect_data \
#     --config $cfg \
#     --use_wandb
