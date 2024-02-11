import isaacgym
from openrl_ws.utils import make_env, get_args

from openrl.envs.common import make
from openrl.modules.common import PPONet
from openrl.runners.common import PPOAgent
from utils import make_env

args = get_args()
args.num_envs = 5
args.headless = False
env = make_env(args)
net = PPONet(env, device="cuda")  # Create neural network.
agent = PPOAgent(net)  # Initialize the agent.
agent.load("/home/ziyanx/python/multiagent-quadruped-environments/checkpoints/go1gate_10_0_-0.004_-0.25_-0.01/module.pt")

agent.set_env(env)  # The agent requires an interactive environment.
obs = env.reset()  # Initialize the environment to obtain initial observations and environmental information.
while True:
    action, _ = agent.act(obs)  # The agent predicts the next action based on environmental observations.
    # The environment takes one step according to the action, obtains the next observation, reward, whether it ends and environmental information.
    obs, r, done, info = env.step(action)