export LD_LIBRARY_PATH=/home/ziyan/anaconda3/envs/mqe/lib
# task="go1football-defender"
# task="go1gate"
task="go1seesaw"
# task="go1sheep-easy"
# task="go1sheep-hard"
# task="go1pushbox-plane"
# task="go1pushbox-gate"
# random_seed=0
device=0
num_envs=500
num_steps=40000000

# algo="jrpo"
# cfg=./openrl_ws/cfgs/jrpo.yaml
algo="ppo"
cfg=./openrl_ws/cfgs/ppo.yaml
# algo="mat"
# cfg=./openrl_ws/cfgs/mat.yaml
# algo="sppo"
# cfg=./openrl_ws/cfgs/ppo.yaml
# algo="dppo"
# cfg=./openrl_ws/cfgs/dppo.yaml

python ./openrl_ws/train.py --headless --num_envs $num_envs --train_timesteps $num_steps\
    --task $task \
    --algo $algo \
    --sim_device cuda:$device \
    --rl_device cuda:$device \
    --seed 0 \
    --exp_name test \
    --config $cfg \
    --use_wandb
