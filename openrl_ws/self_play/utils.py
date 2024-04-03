#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright 2023 The OpenRL Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

""""""
import functools
from copy import deepcopy
from typing import Optional

import numpy as np
from openrl_ws.utils import make_env, get_args
from legged_gym.envs.utils import make_mqe_env
from gymnasium import spaces
from pettingzoo import AECEnv
from pettingzoo.utils import agent_selector
from typing import List, Optional
from typing import Any, Dict, Optional, Union

from openrl.selfplay.opponents.utils import get_opponent_from_info
from openrl.selfplay.selfplay_api.selfplay_client import SelfPlayClient

import torch

import copy
import numbers
from abc import ABC, abstractmethod
from typing import Optional, Union

import numpy as np
from gymnasium import spaces
from gymnasium.core import ActType, ObsType, WrapperActType, WrapperObsType
from gymnasium.utils import seeding

from openrl.envs.wrappers.base_wrapper import BaseWrapper


class BaseMultiPlayerWrapper(BaseWrapper, ABC):
    """
    Base class for multi-player wrappers.
    """

    _np_random: Optional[np.random.Generator] = None
    self_player: Optional[str] = None

    def close(self):
        self.env.close()

    @property
    def np_random(self) -> np.random.Generator:
        """Returns the environment's internal :attr:`_np_random` that if not set will initialise with a random seed.

        Returns:
            Instances of `np.random.Generator`
        """
        if self._np_random is None:
            self._np_random, _ = seeding.np_random()
        return self._np_random

    @property
    def action_space(
        self,
    ) -> Union[spaces.Space[ActType], spaces.Space[WrapperActType]]:
        """Return the :attr:`Env` :attr:`action_space` unless overwritten then the wrapper :attr:`action_space` is used."""
        if self._action_space is None:
            if self.self_player is None:
                self.env.reset()
                self.self_player = self.np_random.choice(self.env.agents)
            action_sapce = self.env.action_space(self.self_player)
            if isinstance(action_sapce, list):
                return action_sapce[0]
            else:
                return action_sapce
        return self._action_space

    @property
    def observation_space(
        self,
    ) -> Union[spaces.Space[ObsType], spaces.Space[WrapperObsType]]:
        """Return the :attr:`Env` :attr:`observation_space` unless overwritten then the wrapper :attr:`observation_space` is used."""
        if self._observation_space is None:
            if self.self_player is None:
                self.env.reset()
                self.self_player = self.np_random.choice(self.env.agents)

            return self.env.observation_spaces[self.self_player]

        return self._observation_space

    @abstractmethod
    def get_opponent_action(
        self, player_name: str, observation, reward, termination, truncation, info
    ):
        raise NotImplementedError

    def reset(self, *, seed: Optional[int] = None, **kwargs):
        while True:
            self.env.reset(seed=seed, **kwargs)
            self.self_player = self.env.agents[0]
            self.opponent_players = self.env.agents.copy()
            self.opponent_players.remove(self.self_player)
            while True:
                for player_name in self.env.agent_iter():
                    observation, reward, termination, truncation, info = self.env.last()
                    if termination or truncation:
                        assert False, "This should not happen"

                    if self.self_player == player_name:
                        return copy.copy(observation), info

                    action = self.get_opponent_action(
                        player_name, observation, reward, termination, truncation, info
                    )

                    self.env.step(action)

    def on_episode_end(
        self, player_name, observation, reward, termination, truncation, info
    ):
        pass

    def step(self, action):
        observation, reward, termination, truncation, info = self._step(action)
        need_convert_termination = isinstance(termination, bool)
        if need_convert_termination:
            termination = [termination for _ in range(self.agent_num)]
            truncation = [truncation for _ in range(self.agent_num)]

        need_convert_reward = isinstance(reward, numbers.Real)
        if need_convert_reward:
            reward = [[reward] for _ in range(self.agent_num)]

        return observation, reward, termination, truncation, info

    def _step(self, action):
        self.env.step(action)

        while True:
            for player_name in self.env.agent_iter():
                observation, reward, termination, truncation, info = self.env.last()

                if termination:
                    self.on_episode_end(
                        player_name, observation, reward, termination, truncation, info
                    )

                if self.self_player == player_name:
                    return (
                        copy.copy(observation),
                        reward,
                        termination,
                        truncation,
                        info,
                    )
                if termination or truncation:
                    return (
                        copy.copy(self.env.observe(self.self_player)),
                        (
                            self.env.rewards[self.self_player]
                            if self.self_player in self.env.rewards
                            else 0
                        ),
                        termination,
                        truncation,
                        (
                            self.env.infos[self.self_player]
                            if self.self_player in self.env.rewards
                            else {}
                        ),
                    )

                else:
                    action = self.get_opponent_action(
                        player_name, observation, reward, termination, truncation, info
                    )
                    self.env.step(action)


class QuadrupedAECEnv(AECEnv):
    metadata = {"render.modes": ["human"], "name": "mqe"}

    @property
    def agent_num(self):
        return self.player_each_side

    def __init__(self, env, render_mode: Optional[str] = None, id: str = None):
        self.env = env
        self.env_name = self.env.env_name
        self.num_envs = self.env.num_envs
        agent_num = len(self.possible_agents)
        player_each_side = (self.env.num_agents) // 2
        self.player_each_side = player_each_side
        self.parallel_env_num = 1
        self.agent_name_to_slice = dict(
            zip(
                self.possible_agents,
                [
                    slice(i, (i + 1))
                    for i in range(agent_num)
                ]
            )
        )
        self.agent_name_mapping = dict(
            zip(self.possible_agents, list(range(len(self.possible_agents))))
        )

        self.action_spaces = {
            agent: [self.env.action_space for _ in range(player_each_side)]
            for agent in self.possible_agents
        }

        self.observation_spaces = {
            agent: self.env.observation_space
            for agent in self.possible_agents
        }

        self.agents = self.possible_agents

        self.observations = {agent: None for agent in self.agents}
        self.raw_obs, self.raw_reward, self.raw_done, self.raw_info = (
            None,
            None,
            None,
            None,
        )

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        return deepcopy(self.observation_spaces[agent])

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        return deepcopy(self.action_spaces[agent])

    def observe(self, agent):
        return torch.squeeze(self.raw_obs[self.agent_name_to_slice[agent]], 0).cpu().numpy()

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ):
        # if seed is not None:
        #     self.env.seed(seed)
        self.agents = self.possible_agents[:]
        self.rewards = {agent: 0 for agent in self.agents}
        self._cumulative_rewards = {agent: 0 for agent in self.agents}
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}
        self.state = {agent: None for agent in self.agents}
        self.observations = {agent: None for agent in self.agents}

        raw_obs = self.env.reset()
        self.raw_obs = torch.stack([raw_obs[:, :self.player_each_side, :], raw_obs[:, self.player_each_side:, :]], dim=0)
        self._agent_selector = agent_selector(self.agents)
        self.agent_selection = self._agent_selector.next()

    def step(self, action):
        agent = self.agent_selection
        self._cumulative_rewards[agent] = 0
        
        action = torch.from_numpy(action* 0.5).cuda().float().clip(-1, 1) if type(action) is np.ndarray else action
        self.state[self.agent_selection] = action
        if self._agent_selector.is_last():
            joint_action = torch.zeros(self.env.num_envs, self.env.num_agents, 3, dtype=torch.float, device=self.env.device, requires_grad=False)
            for agent in self.agents:
                agent_num = int(agent.split("_")[1])
                joint_action[:, agent_num * self.player_each_side : (agent_num + 1) * self.player_each_side, :] = self.state[agent]
                
            raw_obs, raw_reward, raw_done, self.raw_info = self.env.step(
                joint_action
            )
            self.raw_obs = torch.stack([raw_obs[:, :self.player_each_side, :], raw_obs[:, self.player_each_side:, :]], dim=0)
            self.raw_reward = torch.stack([raw_reward[:, :self.player_each_side], raw_reward[:, self.player_each_side:]], dim=0)
            self.raw_done = raw_done.cpu().unsqueeze(-1).repeat(1, self.player_each_side).numpy().astype(bool)

            self.rewards = {
                agent: torch.squeeze(self.raw_reward[self.agent_name_to_slice[agent]], 0).cpu().numpy()
                for agent in self.agents
            }

            if np.any(self.raw_done):
                for key in self.terminations:
                    self.terminations[key] = True
        else:
            self._clear_rewards()

            # selects the next agent.
        self.agent_selection = self._agent_selector.next()
        self._accumulate_rewards()

    def render(self):
        img = self.env.render()
        return img

    def close(self):
        self.env.close()

    @property
    def possible_agents(self):
        return ["player_" + str(i) for i in range(2)]

    @property
    def num_agents(self):
        return len(self.possible_agents)

    def sample_action(self, player_name):
        self.player_action_space = self.action_space(player_name)
        if isinstance(self.player_action_space, list):
            action = []
            for space in self.player_action_space:
                action.append(space.sample())

        else:
            action = self.player_action_space.sample()
        return torch.tensor(action, dtype=torch.float32, device="cuda").repeat(self.env.num_envs, self.player_each_side, 1)



class OpponentPoolWrapper(BaseMultiPlayerWrapper):
    def __init__(self, env, cfg, reward_class=None) -> None:
        super().__init__(env, cfg, reward_class)

        host = cfg.selfplay_api.host
        port = cfg.selfplay_api.port
        self.api_client = SelfPlayClient(f"http://{host}:{port}/selfplay/")
        self.opponent = None
        self.opponent_player = None
        self.lazy_load_opponent = cfg.lazy_load_opponent
        self.player_ids = None
        self.parallel_env_num = self.env.num_envs
        

    def reset(self, *, seed: Optional[int] = None, **kwargs):
        
        results = super().reset(seed=seed, **kwargs)
        self.opponent = self.get_opponent(self.opponent_players)

        if self.opponent is not None:
            self.opponent.reset()

        return results[0]
    
    def step(self, action, extra_data: Optional[Dict[str, Any]] = None):
        
        observation, reward, termination, truncation, info =super().step(action)
        infos = []
        for i in range(self.env.raw_done.shape[0]):
            infos.append({})
        
        return observation, reward.reshape(self.parallel_env_num, self.player_each_side, 1), self.env.raw_done, infos
    
    def get_opponent(self, opponent_players: List[str]):
        opponent_info = self.api_client.get_opponent(opponent_players)

        if opponent_info is not None:
            # currentkly, we only support 1 opponent, that means we only support games with two players
            opponent_info = opponent_info[0]
            opponent_player = opponent_players[0]
            opponent, is_new_opponent = get_opponent_from_info(
                opponent_info,
                current_opponent=self.opponent,
                lazy_load_opponent=self.lazy_load_opponent,
            )
            if opponent is None:
                return self.opponent
            if is_new_opponent or (self.opponent_player != opponent_player):
                opponent.set_env(self.env, opponent_player)
                self.opponent_player = opponent_player

            return opponent
        else:
            return self.opponent

    def get_opponent_action(
        self, player_name, observation, reward, termination, truncation, info
    ):
        if self.opponent is None:
            self.opponent = self.get_opponent(self.opponent_players)
            if self.opponent is not None:
                self.opponent.reset()

        if self.opponent is None:
            action = self.env.sample_action(player_name)
        else:
            action = self.opponent.act(
                        player_name, observation, reward, termination, truncation, None
                    )
            action = torch.from_numpy(action* 0.5).cuda().float().clip(-1, 1).reshape(self.parallel_env_num, self.env.player_each_side, -1)
        return action
    

    def batch_rewards(self, buffer):
        step_count = self.env.env.reward_buffer["step count"]
        reward_dict = {}
        if self.self_player == "player_0":
            
            for k in self.env.env.reward_buffer.keys():
                if k == "step count":
                    continue
                reward_dict[k] = self.env.env.reward_buffer[k] / (self.num_envs * step_count)
                self.env.env.reward_buffer[k] = 0
        else:
            
            for k in self.env.env.reward_buffer_1.keys():
                if k == "step count":
                    continue
                reward_dict[k] = self.env.env.reward_buffer_1[k] / (self.num_envs * step_count)
                self.env.env.reward_buffer_1[k] = 0
        self.env.env.reward_buffer["step count"] = 0
        return reward_dict


def make_env(args, cfg, custom_cfg=None):
    env, env_cfg = make_mqe_env(args.task, args, custom_cfg=custom_cfg)
    env = QuadrupedAECEnv(env)
    return OpponentPoolWrapper(env, cfg)
