# environments
from legged_gym.envs.go1.go1 import Go1
from legged_gym.envs.npc.go1_sheep import Go1Sheep
from legged_gym.envs.npc.go1_object import Go1Object
from legged_gym.envs.npc.go1_football_defender import Go1FootballDefender
from legged_gym.envs.npc.go1_football import Go1Football

# configs
from legged_gym.envs.configs.go1_plane_config import Go1PlaneCfg
from legged_gym.envs.configs.go1_gate_config import Go1GateCfg
from legged_gym.envs.configs.go1_sheep_config import SingleSheepCfg, NineSheepCfg
from legged_gym.envs.configs.go1_football_defender_config import Go1FootballDefenderCfg
from legged_gym.envs.configs.go1_seesaw_config import Go1SeesawCfg
from legged_gym.envs.configs.go1_pushbox_config import Go1PushboxPlaneCfg, Go1PushboxGateCfg
from legged_gym.envs.configs.go1_bridge_config import Go1BridgeCfg
from legged_gym.envs.configs.go1_rotation_config import Go1RotationCfg
from legged_gym.envs.configs.go1_wrestling_config import Go1WrestlingCfg
from legged_gym.envs.configs.go1_tug_config import Go1TugCfg
from legged_gym.envs.configs.go1_football_against_config import Go1FootballAgainstCfg
from legged_gym.envs.configs.go1_football_config import Go1FootballCfg

# wrappers
from legged_gym.envs.wrappers.empty_wrapper import EmptyWrapper
from legged_gym.envs.wrappers.go1_gate_wrapper import Go1GateWrapper
from legged_gym.envs.wrappers.go1_pushbox_wrapper import Go1PushboxPlaneWrapper, Go1PushboxGateWrapper
from legged_gym.envs.wrappers.go1_sheep_wrapper import Go1SheepWrapper
from legged_gym.envs.wrappers.go1_seesaw_wrapper import Go1SeesawWrapper
from legged_gym.envs.wrappers.go1_football_defender_wrapper import Go1FootballDefenderWrapper
from legged_gym.envs.wrappers.go1_football_against_wrapper import Go1FootballAgainstWrapper
from legged_gym.envs.wrappers.go1_football_wrapper import Go1FootballWrapper
from legged_gym.envs.wrappers.go1_tug_wrapper import Go1TugWrapper
from legged_gym.envs.wrappers.go1_bridge_wrapper import Go1BridgeWrapper
from legged_gym.envs.wrappers.go1_wrestling_wrapper import Go1WrestlingWrapper
from legged_gym.envs.wrappers.go1_rotation_wrapper import Go1RotationWrapper

from legged_gym.utils import get_args, make_env

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
    "go1pushbox-plane": {
        "class": Go1Object,
        "config": Go1PushboxPlaneCfg,
        "wrapper": Go1PushboxPlaneWrapper
    },
    "go1pushbox-gate": {
        "class": Go1Object,
        "config": Go1PushboxGateCfg,
        "wrapper": Go1PushboxGateWrapper
    },
    "go1football-against": {
        "class": Go1Object,
        "config": Go1FootballAgainstCfg,
        "wrapper": Go1FootballAgainstWrapper
    },
    "go1football": {
        "class": Go1Football,
        "config": Go1FootballCfg,
        "wrapper": Go1FootballWrapper
    },
    "go1rotation": {
        "class": Go1Object,
        "config": Go1RotationCfg,
        "wrapper": Go1RotationWrapper
    },
    "go1bridge": {
        "class": Go1Object,
        "config": Go1BridgeCfg,
        "wrapper": Go1BridgeWrapper
    },
    "go1wrestling": {
        "class": Go1Object,
        "config": Go1WrestlingCfg,
        "wrapper": Go1WrestlingWrapper
    },
    "go1tug": {
        "class": Go1Object,
        "config": Go1TugCfg,
        "wrapper": Go1TugWrapper
    },
}

def make_mqe_env(env_name, args=None, custom_cfg=None):
    
    env_dict = ENV_DICT[env_name]

    if callable(custom_cfg):
        env_dict["config"] = custom_cfg(env_dict["config"])

    env, env_cfg = make_env(env_dict["class"], env_dict["config"], args)
    env = env_dict["wrapper"](env)

    return env, env_cfg