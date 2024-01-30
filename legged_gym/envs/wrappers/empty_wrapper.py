import gym

class EmptyWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)