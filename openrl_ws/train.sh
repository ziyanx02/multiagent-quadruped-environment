
# task="go1football-defender"
# task="go1gate"
# task="go1seesaw"
# num_envs=500
# num_steps=40000000
# task="go1sheep-easy"
# task="go1sheep-hard"
# num_envs=300
# num_steps=24000000
task="go1pushbox-plane"
task="go1pushbox-gate"
num_steps=15000000
random_seed=0
device=2
# algo="jrpo"
# cfg=./openrl_ws/cfgs/jrpo.yaml
# algo="ppo"
# cfg=./openrl_ws/cfgs/ppo.yaml
# algo="mat"
# cfg=./openrl_ws/cfgs/mat.yaml
# algo="sppo"
# cfg=./openrl_ws/cfgs/ppo.yaml
algo="dppo"
cfg=./openrl_ws/cfgs/dppo.yaml

python ./openrl_ws/train.py --headless --num_envs $num_envs --train_timesteps $num_steps\
    --task $task \
    --algo $algo \
    --sim_device cuda:$device \
    --rl_device cuda:$device \
    --seed 0 \
    --exp_name collect_data \
    --config $cfg \
    --use_wandb

python ./openrl_ws/train.py --headless --num_envs $num_envs --train_timesteps $num_steps\
    --task $task \
    --algo $algo \
    --sim_device cuda:$device \
    --rl_device cuda:$device \
    --seed 1 \
    --exp_name collect_data \
    --config $cfg \
    --use_wandb

python ./openrl_ws/train.py --headless --num_envs $num_envs --train_timesteps $num_steps\
    --task $task \
    --algo $algo \
    --sim_device cuda:$device \
    --rl_device cuda:$device \
    --seed 2 \
    --exp_name collect_data \
    --config $cfg \
    --use_wandb

python ./openrl_ws/train.py --headless --num_envs $num_envs --train_timesteps $num_steps\
    --task $task \
    --algo $algo \
    --sim_device cuda:$device \
    --rl_device cuda:$device \
    --seed 3 \
    --exp_name collect_data \
    --config $cfg \
    --use_wandb

python ./openrl_ws/train.py --headless --num_envs $num_envs --train_timesteps $num_steps\
    --task $task \
    --algo $algo \
    --sim_device cuda:$device \
    --rl_device cuda:$device \
    --seed 4 \
    --exp_name collect_data \
    --config $cfg \
    --use_wandb
