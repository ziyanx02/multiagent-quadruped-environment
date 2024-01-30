# environments
from legged_gym.envs.go1.go1 import Go1
from legged_gym.envs.npc.go1_sheep import Go1Sheep

# configs
from legged_gym.envs.configs.go1_plane_config import Go1PlaneCfg
from legged_gym.envs.configs.go1_gate_config import Go1GateCfg
from legged_gym.envs.configs.go1_sheep_config import Go1SheepCfg

# wrappers
from legged_gym.envs.wrappers.go1_gate_wrapper import Go1GateWrapper

from legged_gym.utils import get_args, make_env

ENV_DICT = {
    "go1gate": {
        "class": Go1,
        "config": Go1GateCfg,
        "wrapper": Go1GateWrapper
    },
    "go1sheep": {
        "class": Go1Sheep,
        "config": Go1SheepCfg,
        "wrapper": Go1GateWrapper
    },
}

def make_mqe_env(env_name, args):
    
    env_dict = ENV_DICT[env_name]

    env, env_cfg = make_env(env_dict["class"], env_dict["config"], args)
    env = env_dict["wrapper"](env)

    return env, env_cfg