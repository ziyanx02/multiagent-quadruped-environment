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
import torch
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
from openrl.selfplay.wrappers.base_multiplayer_wrapper import BaseMultiPlayerWrapper


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
                    slice(i * player_each_side, (i + 1) * player_each_side)
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
        self.raw_obs = torch.stack([raw_obs[:, :self.env.num_agents//2, :], raw_obs[:, self.env.num_agents//2:, :]], dim=0)
        self._agent_selector = agent_selector(self.agents)
        self.agent_selection = self._agent_selector.next()

    def step(self, action):
        agent = self.agent_selection
        self._cumulative_rewards[agent] = 0
        self.state[self.agent_selection] = action
        if self._agent_selector.is_last():
            joint_action = torch.zeros(self.env.num_envs, self.env.num_agents, 3, dtype=torch.float, device=self.env.device, requires_grad=False)
            for agent in self.agents:
                agent_num = int(agent.split("_")[1])
                joint_action[:, agent_num:agent_num+1, :] = torch.from_numpy(self.state[agent]).cuda().clip(-1, 1)
               
            raw_obs, raw_reward, raw_done, self.raw_info = self.env.step(
                joint_action
            )
            self.raw_obs = torch.stack([raw_obs[:, :self.env.num_agents//2, :], raw_obs[:, self.env.num_agents//2:, :]], dim=0)
            self.raw_reward = torch.stack([raw_reward[:, :self.env.num_agents//2], raw_reward[:, self.env.num_agents//2:]], dim=0)
            self.raw_done = raw_done.cpu().unsqueeze(-1).repeat(1, self.agent_num).numpy().astype(bool)

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
        return torch.tensor(action, dtype=torch.float32, device="cuda").repeat(self.env.num_envs, 1, 1)



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

        return results
    
    def step(self, action, extra_data: Optional[Dict[str, Any]] = None):
        observation, reward, termination, truncation, info =super().step(action)
        infos = []
        for i in range(self.env.raw_done.shape[0]):
            infos.append({})
        
        return observation, reward, self.env.raw_done, infos
    
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
        return action
    

    def batch_rewards(self, buffer):
        step_count = self.env.env.reward_buffer["step count"]
        reward_dict = {}
        for k in self.env.env.reward_buffer.keys():
            if k == "step count":
                continue
            reward_dict[k] = self.env.env.reward_buffer[k] / (self.num_envs * step_count)
            self.env.env.reward_buffer[k] = 0
        self.env.env.reward_buffer["step count"] = 0
        return reward_dict
    # def on_episode_end(
    #     self, player_name, observation, reward, termination, truncation, info
    # ):
    #     assert "winners" in info, "winners must be in info"
    #     assert "losers" in info, "losers must be in info"
    #     assert len(info["winners"]) >= 1, "winners must be at least 1"

    #     winner_ids = []
    #     loser_ids = []

    #     for player in info["winners"]:
    #         if player == self.self_player:
    #             winner_id = "training_agent"
    #         else:
    #             winner_id = self.opponent.opponent_id
    #         winner_ids.append(winner_id)

    #     for player in info["losers"]:
    #         if player == self.self_player:
    #             loser_id = "training_agent"
    #         else:
    #             loser_id = self.opponent.opponent_id
    #         loser_ids.append(loser_id)
    #     assert set(winner_ids).isdisjoint(set(loser_ids)), (
    #         "winners and losers must be disjoint, but get winners: {}, losers: {}"
    #         .format(winner_ids, loser_ids)
    #     )
    #     battle_info = {"winner_ids": winner_ids, "loser_ids": loser_ids}
    #     self.api_client.add_battle_result(battle_info)


def make_env(args, cfg, custom_cfg=None):
    env, env_cfg = make_mqe_env(args.task, args, custom_cfg=custom_cfg)
    env = QuadrupedAECEnv(env)
    return OpponentPoolWrapper(env, cfg)
# def make_env(args, cfg, custom_cfg=None):
#     env, env_cfg = make_mqe_env(args.task, args, custom_cfg=custom_cfg)
    
#     return QuadrupedAECEnv(env)
