# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

from legged_gym import LEGGED_GYM_ROOT_DIR, LEGGED_GYM_ENVS_DIR
from legged_gym.envs.base.legged_robot import LeggedRobot
from legged_gym.envs.field.legged_robot_field import LeggedRobotField
from legged_gym.envs.base.legged_robot_noisy import LeggedRobotNoisy
from legged_gym.envs.field.go1_field_config import Go1FieldCfg, Go1FieldCfgPPO
from legged_gym.envs.go1.go1_field_distill_config import Go1FieldDistillCfg, Go1FieldDistillCfgPPO


import os

from legged_gym.utils.task_registry import task_registry

task_registry.register( "go1_field", LeggedRobotNoisy, Go1FieldCfg(), Go1FieldCfgPPO())
task_registry.register( "go1_distill", LeggedRobotNoisy, Go1FieldDistillCfg(), Go1FieldDistillCfgPPO())

from .go1.go1_dualrun_config import Go1DualrunCfg, Go1DualrunCfgPPO
task_registry.register( "go1_dualrun", LeggedRobotField, Go1DualrunCfg(), Go1DualrunCfgPPO() )
from .go1.go1_config import Go1Cfg, Go1CfgPPO
from .go1.go1 import Go1
task_registry.register( "go1", Go1, Go1Cfg(), Go1CfgPPO() )
from .configs.go1_dualrun_test_config import Go1DualrunTestCfg
task_registry.register("go1_test_dualrun", Go1, Go1DualrunTestCfg(), Go1DualrunCfgPPO())
