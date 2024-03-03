import numpy as np
from openrl_ws.utils import  get_args
from openrl.envs.common import make
from openrl.envs.PettingZoo.registration import register
from openrl_ws.self_play.utils import QuadrupedAECEnv, make_env
from openrl.selfplay.wrappers.random_opponent_wrapper import RandomOpponentWrapper
from openrl.configs.config import create_config_parser
from legged_gym.envs.utils import make_mqe_env
import torch

args = get_args()
env, env_cfg = make_mqe_env(args.task, args)
env = QuadrupedAECEnv(env)
env.reset()
step = 0
for player_name in env.agent_iter():
    # if step > 20:
    #     break
    observation, reward, termination, truncation, info = env.last()
    action = env.sample_action(player_name)
    # print(torch.tensor(action).cuda().size())

    env.step(action.cpu().numpy())
    step += 1

print("Total steps: {}".format(step))
