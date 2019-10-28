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


import pytest
import numpy as np
import tensorflow as tf

import rddlgym
from rddl2tf.compilers import ReparameterizationCompiler

from tfplan.planners.stochastic import utils
from tfplan.planners.stochastic.simulation import SimulationCell, Simulator
from tfplan.train.policy import OpenLoopPolicy


BATCH_SIZE = 32
HORIZON = 20


@pytest.fixture(scope="module", params=["Navigation-v3"])
def cell(request):
    rddl = request.param
    model = rddlgym.make(rddl, mode=rddlgym.AST)

    compiler = ReparameterizationCompiler(model, batch_size=BATCH_SIZE)
    compiler.init()

    policy = OpenLoopPolicy(compiler, HORIZON, parallel_plans=True)
    policy.build("planning")

    with compiler.graph.as_default():
        reparameterization_map = compiler.get_cpfs_reparameterization()
        cell_samples = utils.get_noise_samples(
            reparameterization_map, BATCH_SIZE, horizon=1
        )
        cell_noise, encoding = utils.encode_noise_samples_as_inputs(cell_samples)

    cell = SimulationCell(compiler, policy, config={"encoding": encoding})
    cell.cell_noise = cell_noise
    yield cell


def test_state_size(cell):
    rddl = cell.compiler.rddl
    state_size = rddl.state_size
    cell_state_size = cell.state_size
    assert len(cell_state_size) == len(state_size)
    for size, cell_size in zip(state_size, cell_state_size):
        if size == tuple():
            assert cell_size == (1,)
        else:
            assert cell_size == size


def test_action_size(cell):
    rddl = cell.compiler.rddl
    action_size = rddl.action_size
    cell_action_size = cell.action_size
    assert len(cell_action_size) == len(action_size)
    for size, cell_size in zip(action_size, cell_action_size):
        if size == tuple():
            assert cell_size == (1,)
        else:
            assert cell_size == size


def test_interm_size(cell):
    rddl = cell.compiler.rddl
    interm_size = rddl.interm_size
    cell_interm_size = cell.interm_size
    assert len(cell_interm_size) == len(interm_size)
    for size, cell_size in zip(interm_size, cell_interm_size):
        if size == tuple():
            assert cell_size == (1,)
        else:
            assert cell_size == size


def test_output_size(cell):
    output_size = cell.output_size
    assert output_size == (cell.state_size, cell.action_size, cell.interm_size, 1)


def test_call(cell):
    state = cell.compiler.initial_state()
    action = cell.compiler.default_action()

    with cell.compiler.graph.as_default():
        timesteps = tf.zeros((BATCH_SIZE, 1), dtype=tf.float32)
        inputs = tf.concat([timesteps, cell.cell_noise[:, 0, ...]], axis=1)
        output, _ = cell(inputs, state)

    assert len(output) == 4
    assert output[0] is not None
    assert output[1] is not None
    assert output[2] is not None
    assert output[3] is not None

    assert len(output[0]) == len(state)
    for output_state_tensor, state_tensor in zip(output[0], state):
        assert isinstance(output_state_tensor[0], tf.Tensor)
        assert output_state_tensor[0].dtype == state_tensor.dtype
        assert output_state_tensor[0].shape == state_tensor.shape

    assert len(output[1]) == len(action)
    for output_action_tensor, action_tensor in zip(output[1], action):
        assert isinstance(output_action_tensor[0], tf.Tensor)
        assert output_action_tensor[0].dtype == action_tensor.dtype
        assert output_action_tensor[0].shape == action_tensor.shape

    interm_size = cell.compiler.rddl.interm_size
    assert len(output[2]) == len(interm_size)
    for output_interm_tensor, interm_fluent_size in zip(output[2], interm_size):
        assert isinstance(output_interm_tensor[0], tf.Tensor)
        assert output_interm_tensor[0].shape == (BATCH_SIZE, *interm_fluent_size)

    assert isinstance(output[3], tf.Tensor)
    assert output[3].dtype == tf.float32
    assert output[3].shape == (BATCH_SIZE, 1)


@pytest.fixture(scope="module", params=["Navigation-v3"])
def simulator(request):
    rddl = request.param
    model = rddlgym.make(rddl, mode=rddlgym.AST)

    compiler = ReparameterizationCompiler(model, batch_size=BATCH_SIZE)
    compiler.init()

    policy = OpenLoopPolicy(compiler, HORIZON, parallel_plans=True)
    policy.build("planning")

    simulator = Simulator(compiler, policy, config=None)
    simulator.build()
    yield simulator


def test_build(simulator):
    reparameterization_map = simulator.cell
    assert reparameterization_map is not None

    samples = simulator.samples
    assert samples is not None

    noise = simulator.noise
    assert noise is not None

    cell = simulator.cell
    assert cell is not None
    assert isinstance(cell, SimulationCell)


def test_trajectory(simulator):
    state = simulator.cell.compiler.initial_state()
    action = simulator.cell.compiler.default_action()

    trajectory, final_state, total_reward = simulator.trajectory(state)

    assert len(trajectory) == 4
    assert trajectory.states is not None
    assert trajectory.actions is not None
    assert trajectory.interms is not None
    assert trajectory.rewards is not None

    assert len(trajectory.states) == len(state)
    for output_state_tensor, state_tensor in zip(trajectory.states, state):
        assert isinstance(output_state_tensor, tf.Tensor)
        assert output_state_tensor.dtype == state_tensor.dtype
        assert output_state_tensor.shape == (
            BATCH_SIZE,
            HORIZON,
            *state_tensor.shape[1:],
        )

    assert len(trajectory.actions) == len(action)
    for output_action_tensor, action_tensor in zip(trajectory.actions, action):
        assert isinstance(output_action_tensor, tf.Tensor)
        assert output_action_tensor.dtype == action_tensor.dtype
        assert output_action_tensor.shape == (
            BATCH_SIZE,
            HORIZON,
            *action_tensor.shape[1:],
        )

    interm_size = simulator.cell.compiler.rddl.interm_size
    assert len(trajectory.interms) == len(interm_size)
    for output_interm_tensor, interm_fluent_size in zip(
        trajectory.interms, interm_size
    ):
        assert isinstance(output_interm_tensor, tf.Tensor)
        assert output_interm_tensor.shape == (BATCH_SIZE, HORIZON, *interm_fluent_size)

    assert isinstance(trajectory.rewards, tf.Tensor)
    assert trajectory.rewards.dtype == tf.float32
    assert trajectory.rewards.shape == (BATCH_SIZE, HORIZON, 1)

    assert len(final_state) == len(state)
    for output_state_tensor, state_tensor in zip(final_state, state):
        assert isinstance(output_state_tensor, tf.Tensor)
        assert output_state_tensor.dtype == state_tensor.dtype
        assert output_state_tensor.shape == (BATCH_SIZE, *state_tensor.shape[1:])

    assert isinstance(total_reward, tf.Tensor)
    assert total_reward.dtype == tf.float32
    assert total_reward.shape == (BATCH_SIZE,)


def test_run(simulator):
    state = simulator.cell.compiler.initial_state()
    trajectory, _, _ = simulator.trajectory(state)
    trajectory_ = simulator.run(trajectory)
    assert trajectory_ is not None


def test_timesteps():
    timesteps = Simulator.timesteps(BATCH_SIZE, HORIZON)
    assert isinstance(timesteps, tf.Tensor)
    assert timesteps.dtype == tf.float32
    assert timesteps.shape == (BATCH_SIZE, HORIZON, 1)
    with tf.Session() as sess:
        timesteps_ = sess.run(timesteps)
        for batch in range(BATCH_SIZE):
            assert np.allclose(timesteps_[batch], timesteps_[0])
            steps = timesteps_[batch]
            for time in range(1, HORIZON):
                assert steps[time] == steps[time - 1] + 1
