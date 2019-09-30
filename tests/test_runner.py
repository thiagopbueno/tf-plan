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

# pylint: disable=missing-docstring,protected-access,redefined-outer-name


from collections import OrderedDict
import numpy as np
import pytest

import rddlgym

from tfplan.planners import DEFAULT_CONFIG, StraightLinePlanner, Tensorplan
from tfplan.test.runner import Runner


HORIZON = 5
EPOCHS = 3


@pytest.fixture(scope="module", params=["Navigation-v1"])
def tensorplan(request):
    rddl = request.param

    env = rddlgym.make(rddl, mode=rddlgym.GYM)
    env._horizon = HORIZON

    config = {**DEFAULT_CONFIG, "epochs": EPOCHS, "horizon": HORIZON}
    planner = Tensorplan(rddl, config)
    planner.build()

    runner = Runner(env, planner, debug=False)
    yield runner
    runner.close()


@pytest.fixture(scope="module", params=["Navigation-v2"])
def straightline(request):
    rddl = request.param

    env = rddlgym.make(rddl, mode=rddlgym.GYM)
    env._horizon = HORIZON

    config = {**DEFAULT_CONFIG, "epochs": EPOCHS, "horizon": HORIZON}
    planner = StraightLinePlanner(rddl, config)
    planner.build()

    runner = Runner(env, planner, debug=False)
    yield runner
    runner.close()


def test_run_tensorplan(tensorplan):
    _test_run(tensorplan)


def test_run_straightline(straightline):
    _test_run(straightline)


def _test_run(runner):
    total_reward, trajectory = runner.run()
    assert len(trajectory) == runner.env._horizon

    for idx, transition in enumerate(trajectory):
        assert transition.step == idx
        assert isinstance(transition.state, OrderedDict)
        assert isinstance(transition.action, OrderedDict)
        assert isinstance(transition.reward, np.float32)
        assert isinstance(transition.next_state, OrderedDict)
        assert isinstance(transition.next_state, OrderedDict)
        assert isinstance(transition.info, OrderedDict)
        assert isinstance(transition.done, bool)

    assert all(not transition.done for transition in trajectory[:-1])
    assert trajectory[-1].done

    assert np.isclose(
        total_reward, sum(map(lambda transition: transition.reward, trajectory))
    )
