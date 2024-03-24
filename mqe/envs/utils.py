# environments
from mqe.envs.go1.go1 import Go1
from mqe.envs.npc.go1_sheep import Go1Sheep
from mqe.envs.npc.go1_object import Go1Object
from mqe.envs.npc.go1_football_defender import Go1FootballDefender

# configs
from mqe.envs.configs.go1_plane_config import Go1PlaneCfg
from mqe.envs.configs.go1_gate_config import Go1GateCfg
from mqe.envs.configs.go1_sheep_config import SingleSheepCfg, NineSheepCfg
from mqe.envs.configs.go1_football_defender_config import Go1FootballDefenderCfg
from mqe.envs.configs.go1_seesaw_config import Go1SeesawCfg
from mqe.envs.configs.go1_pushbox_config import Go1PushboxCfg

# wrappers
from mqe.envs.wrappers.empty_wrapper import EmptyWrapper
from mqe.envs.wrappers.go1_gate_wrapper import Go1GateWrapper
from mqe.envs.wrappers.go1_pushbox_wrapper import Go1PushboxWrapper
from mqe.envs.wrappers.go1_sheep_wrapper import Go1SheepWrapper
from mqe.envs.wrappers.go1_seesaw_wrapper import Go1SeesawWrapper
from mqe.envs.wrappers.go1_football_defender_wrapper import Go1FootballDefenderWrapper

from mqe.utils import get_args, make_env

ENV_DICT = {
    "go1plane": {
        "class": Go1,
        "config": Go1PlaneCfg,
        "wrapper": EmptyWrapper
    },
    "go1gate": {
        "class": Go1,
        "config": Go1GateCfg,
        "wrapper": Go1GateWrapper
    },
    "go1sheep-easy": {
        "class": Go1Sheep,
        "config": SingleSheepCfg,
        "wrapper": Go1SheepWrapper
    },
    "go1sheep-hard": {
        "class": Go1Sheep,
        "config": NineSheepCfg,
        "wrapper": Go1SheepWrapper
    },
    "go1football-defender": {
        "class": Go1FootballDefender,
        "config": Go1FootballDefenderCfg,
        "wrapper": Go1FootballDefenderWrapper
    },
    "go1seesaw": {
        "class": Go1Object,
        "config": Go1SeesawCfg,
        "wrapper": Go1SeesawWrapper
    },
    "go1pushbox": {
        "class": Go1Object,
        "config": Go1PushboxCfg,
        "wrapper": Go1PushboxWrapper
    },
}

def make_mqe_env(env_name, args=None, custom_cfg=None):
    
    env_dict = ENV_DICT[env_name]

    if callable(custom_cfg):
        env_dict["config"] = custom_cfg(env_dict["config"])

    env, env_cfg = make_env(env_dict["class"], env_dict["config"], args)
    env = env_dict["wrapper"](env)

    return env, env_cfg