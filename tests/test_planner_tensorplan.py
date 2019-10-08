# This file is part of tf-plan.

# tf-plan is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# tf-plan is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with tf-plan. If not, see <http://www.gnu.org/licenses/>.

# pylint: disable=missing-docstring,redefined-outer-name


import numpy as np
import pytest

import rddlgym

from tfplan.planners import Tensorplan
from tfplan.train.policy import OpenLoopPolicy


HORIZON = 20
BATCH_SIZE = 32
EPOCHS = 10


@pytest.fixture(scope="module", params=["Navigation-v1"])
def planner(request):
    rddl = request.param
    config = {
        "batch_size": BATCH_SIZE,
        "epochs": EPOCHS,
        "horizon": HORIZON,
        "optimization": {"optimizer": "RMSProp", "learning_rate": 1e-3},
    }
    planner = Tensorplan(rddl, config)
    planner.build()
    return planner


def test_build_policy_ops(planner):
    assert planner.policy is not None
    assert isinstance(planner.policy, OpenLoopPolicy)
    assert planner.policy.parallel_plans


def test_build_initial_state_ops(planner):
    assert planner.initial_state is not None


def test_build_trajectory_ops(planner):
    assert planner.trajectory is not None
    assert planner.final_state is not None
    assert planner.total_reward is not None


def test_build_loss_ops(planner):
    assert planner.avg_total_reward is not None
    assert planner.loss is not None


def test_build_optimization_ops(planner):
    assert planner.optimizer is not None
    assert planner.train_op is not None


def test_build_solution_ops(planner):
    assert planner.best_plan_idx is not None
    assert planner.best_plan is not None


def test_run(planner):
    action = planner.compiler.default_action()

    plan_ = planner.run()
    assert plan_ is not None

    assert len(plan_) == len(action)
    for plan_action, action_tensor in zip(plan_, action):
        assert isinstance(plan_action, np.ndarray)
        assert plan_action.shape == (HORIZON, *action_tensor.shape[1:])


def test_call(planner):
    # pylint: disable=invalid-name,protected-access
    state = None
    for timestep in range(HORIZON):
        action = planner(state, timestep)

        assert len(action) == len(planner._plan)
        for (_, a1), a2 in zip(action.items(), planner._plan):
            assert np.allclose(a1, a2[timestep])


def test_runner(planner):
    # pylint: disable=protected-access
    rddl = planner.rddl
    env = rddlgym.make(rddl, mode=rddlgym.GYM)
    env.set_horizon(HORIZON)
    runner = rddlgym.Runner(env, planner)
    trajectory = runner.run()
    assert len(trajectory) == env._horizon
