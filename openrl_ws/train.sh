task="go1sheep"
num_envs=100
num_steps=10000000
random_seed=4
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
    # --use_wandb
