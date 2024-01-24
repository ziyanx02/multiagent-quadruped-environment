
from openrl_ws.utils import make_mqe_env
from openrl.envs.common import make
from openrl.modules.common import PPONet as Net
from openrl.runners.common import PPOAgent as Agent

import argparse

def train(args):
    env = make_mqe_env("go1gate")
    net = Net(env)  # Create neural network.
    agent = Agent(net)  # Initialize the agent.
    agent.train(
        total_time_steps=20000)  # Start training and set the total number of steps to 20,000 for the running environment.

def get_args():

    parser = argparse.ArgumentParser()

    # parser.add_argument('filename')           # positional argument
    # parser.add_argument('-c', '--count')      # option that takes a value
    # parser.add_argument('-v', '--verbose',
    #                     action='store_true')  # on/off flag
    
    return parser

if __name__ == '__main__':
    parser = get_args()
    args = parser.parse_args()
    train(args)
