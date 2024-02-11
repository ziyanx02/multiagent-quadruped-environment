
from openrl_ws.utils import make_env, get_args, custom_cfg
from openrl.modules.common import PPONet
from openrl.runners.common import PPOAgent
from openrl.utils.logger import Logger

import datetime

# import argparse

def train(args):

    # cfg_parser = create_config_parser()
    # cfg = cfg_parser.parse_args()

    start_time = datetime.now()
    start_time_str = start_time.strftime("%m/%d/%Y-%H:%M:%S")

    env, env_cfg = make_env(args, custom_cfg(args))
    net = PPONet(env, cfg=args, device="cuda")  # Create neural network.
    logger = Logger(
        cfg=net.cfg,
        project_name="MQE",
        scenario_name=args.task,
        wandb_entity="ziyanx02",
        exp_name="test",
        log_path="./log",
        use_wandb=True,
        use_tensorboard=False,
    )
    agent = PPOAgent(net)  # Initialize the agent.
    agent.train(
        total_time_steps=20000000,
        logger=logger)  # Start training and set the total number of steps to 20,000 for the running environment.
    dir_name = "./checkpoints/" + args.task
    agent.save(dir_name)

if __name__ == '__main__':
    args = get_args()
    train(args)
