import numpy as np
from openrl_ws.utils import  get_args
import os
from openrl.envs.common import make
from openrl.envs.PettingZoo.registration import register
from openrl_ws.self_play.utils import QuadrupedAECEnv, make_env
from openrl.selfplay.wrappers.random_opponent_wrapper import RandomOpponentWrapper
from openrl.configs.config import create_config_parser
from openrl.selfplay.wrappers.opponent_pool_wrapper import OpponentPoolWrapper

from openrl.modules.common.ppo_net import PPONet as Net
from openrl.runners.common.ppo_agent import PPOAgent as Agent
import torch
from datetime import datetime

def train(env, cfg, args):
    agent = Agent(Net(env, cfg=cfg))
    agent.train(total_time_steps=args.train_timesteps)
    return agent

def test(env, agent):
    obs, info = env.reset()
    done = False
    total_reward = 0
    total_step = 0
    while True:
        action, _ = agent.act(obs)
        obs, r, done, info = env.step(action)
        total_step += 1
        total_reward += np.mean(r)
        print("total test step: ", total_step)
        print("total test reward: ", total_reward)

start_time = datetime.now()
start_time_str = start_time.strftime("%m/%d/%Y-%H:%M:%S")
args = get_args()
cfg_parser = create_config_parser()
cfg = cfg_parser.parse_args(["--config", "./openrl_ws/self_play/selfplay.yaml"])

env = make_env(args, cfg)
agent = train(env, cfg, args)
dir_name = "./checkpoints/" + args.task + "/" + start_time_str
agent.save("dir_name")
test(env, agent)


