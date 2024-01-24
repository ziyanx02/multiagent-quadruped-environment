from openrl.envs.common import make
from openrl.modules.common import PPONet as Net
from openrl.runners.common import PPOAgent as Agent

agent = Agent(Net(make("CartPole-v1", env_num=9)))  # Initialize trainer.
agent.train(total_time_steps=20000)
# Create an environment for test, set the parallelism of the environment to 9, and set the rendering mode to group_human.
env = make("CartPole-v1", env_num=9, render_mode="group_human")
agent.set_env(env)  # The agent requires an interactive environment.
obs, info = env.reset()  # Initialize the environment to obtain initial observations and environmental information.
while True:
    action, _ = agent.act(obs)  # The agent predicts the next action based on environmental observations.
    # The environment takes one step according to the action, obtains the next observation, reward, whether it ends and environmental information.
    obs, r, done, info = env.step(action)
    if any(done): break
env.close()  # Close test environment