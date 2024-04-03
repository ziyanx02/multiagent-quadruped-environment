
from openrl_ws.utils import make_env, get_args, MATWrapper
from mqe.envs.utils import custom_cfg
from openrl.utils.logger import Logger

from datetime import datetime

# import argparse

def train(args):

    # cfg_parser = create_config_parser()
    # cfg = cfg_parser.parse_args()

    start_time = datetime.now()
    start_time_str = start_time.strftime("%m/%d/%Y-%H:%M:%S")

    if args.algo == "sppo" or args.algo == "dppo":
        single_agent = True
    else:
        single_agent = False
    
    env, env_cfg = make_env(args, custom_cfg(args), single_agent)
    
    if args.algo == "ppo":
        args.config = "./openrl_ws/cfgs/ppo.yaml"
    
    elif args.algo == "jrpo":
        args.config = "./openrl_ws/cfgs/jrpo.yaml"

    elif args.algo == "mat":
        args.config = "./openrl_ws/cfgs/mat.yaml"

        # from openrl.envs.wrappers.mat_wrapper import MATWrapper
        from openrl.modules.common import MATNet
        from openrl.runners.common import MATAgent
        env = MATWrapper(env)
        net = MATNet(env, cfg=args, device=args.rl_device)
        agent = MATAgent(net, use_wandb=args.use_wandb)

    elif args.algo == "sppo" or args.algo == "dppo":
        pass

    else:
        raise NotImplementedError
    
    if "po" in args.algo:
        from openrl.modules.common import PPONet
        from openrl.runners.common import PPOAgent
        net = PPONet(env, cfg=args, device=args.rl_device)
        agent = PPOAgent(net)
        logger = Logger(
            cfg=net.cfg,
            project_name="MQE",
            scenario_name=args.task,
            wandb_entity="ziyanx02",
            exp_name=args.exp_name,
            log_path="./log",
            use_wandb=args.use_wandb,
            use_tensorboard=args.use_tensorboard,
        )
        agent.train(
            total_time_steps=args.train_timesteps,
            logger=logger
        )
    else:
        agent.train(total_time_steps=args.train_timesteps)
    # dir_name = "./checkpoints/" + args.task + "/" + start_time_str
    dir_name = "./checkpoints/" + args.task
    agent.save(dir_name)

if __name__ == '__main__':
    args = get_args()
    train(args)
