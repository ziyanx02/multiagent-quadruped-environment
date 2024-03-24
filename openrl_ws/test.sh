python ./openrl_ws/test.py \
    --task go1sheep-easy \
    --algo ppo \
    --sim_device cuda:0 \
    --rl_device cuda:0 \
    --num_envs 1 --checkpoint /home/ziyanx/python/multiagent-quadruped-environments/checkpoints/go1sheep-easy/module.pt \
    # --record_video