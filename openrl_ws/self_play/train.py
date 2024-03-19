import numpy as np
import os
from openrl.envs.common import make
from openrl.envs.PettingZoo.registration import register
from utils import QuadrupedAECEnv, make_env
from openrl.selfplay.wrappers.random_opponent_wrapper import RandomOpponentWrapper
from openrl.configs.config import create_config_parser
from openrl.selfplay.wrappers.opponent_pool_wrapper import OpponentPoolWrapper

from openrl.modules.common.ppo_net import PPONet as Net
from openrl.runners.common.ppo_agent import PPOAgent as Agent
from datetime import datetime
import sys
sys.path.append("..")
from openrl_ws.utils import get_args
from openrl.utils.logger import Logger

def train(env, cfg, args):
    net = Net(env, cfg=cfg)
    agent = Agent(net)
    logger = Logger(
            cfg=net.cfg,
            project_name="MQE",
            scenario_name=args.task,
            wandb_entity="1374203630",
            exp_name=args.exp_name,
            log_path="./log",
            use_wandb=args.use_wandb,
            use_tensorboard=args.use_tensorboard,
        )
    agent.train(total_time_steps=args.train_timesteps,
            logger=logger
            )
    return agent

start_time = datetime.now()
start_time_str = start_time.strftime("%m/%d/%Y-%H:%M:%S")
args = get_args()
cfg_parser = create_config_parser()
cfg = cfg_parser.parse_args(["--config", "./openrl_ws/self_play/selfplay.yaml"])

env = make_env(args, cfg)
agent = train(env, cfg, args)


