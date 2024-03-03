python ./openrl_ws/test.py \
    --task go1gate \
    --algo mat \
    --sim_device cuda:0 \
    --rl_device cuda:0 \
    --num_envs 1 --checkpoint /home/ziyan/multiagent-quadruped-environments/checkpoints/go1gate/module.pt \
    --record_video \
    --headless