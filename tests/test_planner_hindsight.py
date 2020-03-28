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


from collections import OrderedDict
import pytest
import numpy as np
import tensorflow as tf

import rddlgym
from tfplan.planners import DEFAULT_CONFIG, HindsightPlanner
from tfplan.planners.stochastic import utils


BATCH_SIZE = 16
HORIZON = 20
EPOCHS = 5


@pytest.fixture(scope="module", params=["Navigation-v2"])
def planner(request):
    rddl = request.param
    config = {
        **DEFAULT_CONFIG,
        **{"batch_size": BATCH_SIZE, "horizon": HORIZON, "epochs": EPOCHS},
        "verbose": False,
    }
    planner = HindsightPlanner(rddl, config)
    planner.build()
    return planner


def test_build_base_policy_ops(planner):
    base_policy = planner.base_policy
    assert not base_policy.parallel_plans
    assert base_policy.horizon == 1
    assert hasattr(base_policy, "_policy_variables")


def test_build_scenario_policy_ops(planner):
    scenario_policy = planner.scenario_policy
    compiler = planner.compiler
    assert scenario_policy.parallel_plans
    assert scenario_policy.horizon == compiler.rddl.instance.horizon - 1
    assert hasattr(scenario_policy, "_policy_variables")


def test_build_initial_state_ops(planner):
    initial_state = planner.initial_state
    compiler = planner.compiler
    batch_size = compiler.batch_size
    assert isinstance(initial_state, tuple)
    assert len(initial_state) == len(compiler.initial_state_fluents)
    for tensor, fluent in zip(initial_state, compiler.initial_state_fluents):
        assert tensor.dtype == fluent[1].dtype
        assert tensor.shape == (batch_size, *fluent[1].shape.fluent_shape)


def test_build_scenario_start_states_ops(planner):
    initial_state = planner.initial_state
    next_state = planner.next_state
    assert isinstance(next_state, tuple)
    assert len(next_state) == len(initial_state)
    for initial_state_tensor, next_state_tensor in zip(initial_state, next_state):
        assert initial_state_tensor.shape == next_state_tensor.shape
        assert initial_state_tensor.dtype == next_state_tensor.dtype


def test_build_sequence_length_ops(planner):
    assert planner.steps_to_go is not None
    assert planner.steps_to_go.dtype == tf.int32
    assert planner.steps_to_go.shape == ()
    assert planner.sequence_length is not None
    assert planner.sequence_length.dtype == tf.int32
    assert planner.sequence_length.shape == (planner.compiler.batch_size,)


def test_build_trajectory_ops(planner):
    trajectory = planner.trajectory
    actions = trajectory.actions
    rewards = trajectory.rewards
    assert rewards.shape == (BATCH_SIZE, HORIZON - 1, 1)

    action_fluents = planner.compiler.default_action_fluents
    for action, action_fluent in zip(actions, action_fluents):
        size = action_fluent[1].shape.as_list()
        assert action.shape.as_list() == [BATCH_SIZE, HORIZON - 1, *size]

    with planner.graph.as_default():
        last_reward = tf.reduce_mean(planner.trajectory.rewards[:, -1, 0])
        base_policy_vars = tf.trainable_variables(scope="base_policy")
        base_policy_grads = tf.gradients(last_reward, base_policy_vars)

        base_policy_grads_ = _session_run(planner, base_policy_grads)
        for grad_ in base_policy_grads_:
            assert grad_ is not None
            assert not np.allclose(grad_, np.zeros_like(grad_))

        reward = tf.reduce_mean(planner.reward)
        base_policy_grads = tf.gradients(reward, base_policy_vars)
        assert all(grad is None for grad in base_policy_grads)


def test_loss_ops(planner):
    reward = planner.reward
    scenario_total_reward = planner.scenario_total_reward

    assert reward.shape == (BATCH_SIZE,)
    assert scenario_total_reward.shape == (BATCH_SIZE,)


def test_optimization_ops(planner):

    with planner.graph.as_default():

        grads_and_vars = planner.grads_and_vars
        assert isinstance(grads_and_vars, list)
        assert len(grads_and_vars) == len(tf.trainable_variables())

        for variable in tf.trainable_variables(scope="base_policy"):
            assert variable.shape[:2] == [1, 1]

        for variable in tf.trainable_variables(scope="scenario_policy"):
            assert variable.shape[:2] == [BATCH_SIZE, HORIZON - 1]

    grads_and_vars_ = _session_run(planner, grads_and_vars)
    for grad_, _ in grads_and_vars_:
        assert grad_ is not None
        assert not np.allclose(grad_, np.zeros_like(grad_))


def test_call(planner):
    env = rddlgym.make(planner.rddl, mode=rddlgym.GYM)
    state, timestep = env.reset()
    action = planner(state, timestep)
    assert isinstance(action, OrderedDict)


def test_get_batch_initial_state(planner):
    # pylint: disable=protected-access
    env = rddlgym.make(planner.rddl, mode=rddlgym.GYM)

    with planner.compiler.graph.as_default():
        state = env.observation_space.sample()
        batch_state = planner._get_batch_initial_state(state)
        assert len(state) == len(batch_state)

        for fluent, batch_fluent in zip(state.values(), batch_state):
            assert fluent.dtype == batch_fluent.dtype
            assert fluent.shape == batch_fluent.shape[1:]
            assert batch_fluent.shape[0] == planner.compiler.batch_size


def test_get_action(planner):
    # pylint: disable=protected-access
    env = rddlgym.make(planner.rddl, mode=rddlgym.GYM)

    with tf.Session(graph=planner.compiler.graph) as sess:
        sess.run(tf.global_variables_initializer())
        feed_dict = _get_feed_dict(sess, planner, env)

        actions_ = planner._get_action(planner.action, feed_dict)

        action_fluents = planner.compiler.default_action_fluents
        assert isinstance(actions_, OrderedDict)
        assert len(actions_) == len(action_fluents)
        for action_, action_fluent in zip(actions_.values(), action_fluents):
            assert tf.dtypes.as_dtype(action_.dtype) == action_fluent[1].dtype
            assert list(action_.shape) == list(action_fluent[1].shape.fluent_shape)


def test_runner(planner):
    # pylint: disable=protected-access
    rddl = planner.rddl
    env = rddlgym.make(rddl, mode=rddlgym.GYM)
    env._horizon = 3
    runner = rddlgym.Runner(env, planner)
    trajectory = runner.run()
    assert len(trajectory) == env._horizon


def _session_run(planner, fetches):
    env = rddlgym.make(planner.rddl, mode=rddlgym.GYM)

    with tf.Session(graph=planner.compiler.graph) as sess:
        sess.run(tf.global_variables_initializer())
        feed_dict = _get_feed_dict(sess, planner, env)
        return sess.run(fetches, feed_dict=feed_dict)


def _get_feed_dict(sess, planner, env):
    # pylint: disable=protected-access
    state = env.observation_space.sample()
    batch_state = planner._get_batch_initial_state(state)

    next_state_noise = utils.evaluate_noise_samples_as_inputs(
        sess, planner.cell_samples
    )
    scenario_noise = utils.evaluate_noise_samples_as_inputs(
        sess, planner.simulator.samples
    )

    feed_dict = {
        planner.initial_state: batch_state,
        planner.cell_noise: next_state_noise,
        planner.simulator.noise: scenario_noise,
        planner.steps_to_go: HORIZON - 1,
    }
    return feed_dict
