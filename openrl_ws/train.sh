task="go1sheep"
num_envs=300
num_steps=15000000
random_seed=0
algo="ppo"
sim_device=0
rl_device=0
cfg=./openrl_ws/cfgs/ppo.yaml

python ./openrl_ws/train.py --headless --num_envs $num_envs --train_timesteps $num_steps\
    --task $task \
    --algo $algo \
    --sim_device cuda:$sim_device \
    --rl_device cuda:$rl_device \
    --seed $random_seed \
    --exp_name test \
    --config $cfg \
    --use_wandb

python ./openrl_ws/train.py --headless --num_envs $num_envs --train_timesteps $num_steps\
    --task $task \
    --algo $algo \
    --sim_device cuda:$sim_device \
    --rl_device cuda:$rl_device \
    --seed $random_seed \
    --exp_name test \
    --config $cfg \
    --use_wandb

python ./openrl_ws/train.py --headless --num_envs $num_envs --train_timesteps $num_steps\
    --task $task \
    --algo $algo \
    --sim_device cuda:$sim_device \
    --rl_device cuda:$rl_device \
    --seed $random_seed \
    --exp_name test \
    --config $cfg \
    --use_wandb

task="go1seesaw"
num_envs=500
num_steps=20000000

python ./openrl_ws/train.py --headless --num_envs $num_envs --train_timesteps $num_steps\
    --task $task \
    --algo $algo \
    --sim_device cuda:$sim_device \
    --rl_device cuda:$rl_device \
    --seed $random_seed \
    --exp_name test \
    --config $cfg \
    --use_wandb

task="go1pushbox"

python ./openrl_ws/train.py --headless --num_envs $num_envs --train_timesteps $num_steps\
    --task $task \
    --algo $algo \
    --sim_device cuda:$sim_device \
    --rl_device cuda:$rl_device \
    --seed $random_seed \
    --exp_name test \
    --config $cfg \
    --use_wandb

task="go1gate"

python ./openrl_ws/train.py --headless --num_envs $num_envs --train_timesteps $num_steps\
    --task $task \
    --algo $algo \
    --sim_device cuda:$sim_device \
    --rl_device cuda:$rl_device \
    --seed $random_seed \
    --exp_name test \
    --config $cfg \
    --use_wandb
