
from openrl_ws.utils import make_mqe_env
from openrl.modules.common import PPONet
from openrl.runners.common import PPOAgent
from openrl.utils.logger import Logger
from openrl.configs.config import create_config_parser
from legged_gym.utils import get_args

# import argparse

def train(args):

    # cfg_parser = create_config_parser()
    # cfg = cfg_parser.parse_args()

    env = make_mqe_env("go1gate", args=args)
    net = PPONet(env, cfg=None, device="cuda")  # Create neural network.
    logger = Logger(
        cfg=net.cfg,
        project_name="MQE",
        scenario_name="go1gate",
        wandb_entity="ziyanx02",
        exp_name="test",
        log_path="./log",
        use_wandb=False,
        use_tensorboard=False,
    )
    agent = PPOAgent(net)  # Initialize the agent.
    agent.train(
        total_time_steps=2000000,
        logger=logger)  # Start training and set the total number of steps to 20,000 for the running environment.
    agent.save("./result/")

# def get_args():

    # parser = argparse.ArgumentParser()

    # parser.add_argument('filename')           # positional argument
    # parser.add_argument('-c', '--count')      # option that takes a value
    # parser.add_argument('-v', '--verbose',
    #                     action='store_true')  # on/off flag
    
    # return parser

if __name__ == '__main__':
    # parser = get_args()
    # args = parser.parse_args()
    # args.headless = True
    args = get_args()
    train(args)
