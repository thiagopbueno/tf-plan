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

# pylint: disable=missing-docstring,redefined-outer-name,protected-access


import numpy as np
import pytest
import tensorflow as tf

import rddlgym
from rddl2tf.compilers import DefaultCompiler

from tfplan.train.policy import OpenLoopPolicy


HORIZON = 20
BATCH_SIZE = 32


@pytest.fixture(scope="module", params=["Navigation-v1"])
def compiler(request):
    rddl = request.param
    model = rddlgym.make(rddl, mode=rddlgym.AST)
    compiler = DefaultCompiler(model, batch_size=BATCH_SIZE)
    compiler.init()
    return compiler


@pytest.fixture(scope="module")
def parallel_plans(compiler):
    policy = OpenLoopPolicy(compiler, HORIZON, parallel_plans=True)
    policy.build("parallel_plans")
    return policy


@pytest.fixture(scope="module")
def non_parallel_plans(compiler):
    policy = OpenLoopPolicy(compiler, HORIZON, parallel_plans=False)
    policy.build("non_parallel_plans")
    return policy


def test_build_policy_variables(parallel_plans, non_parallel_plans):
    _test_build_policy_variables(parallel_plans, batch_size=BATCH_SIZE)
    _test_build_policy_variables(non_parallel_plans, batch_size=1)


def _test_build_policy_variables(policy, batch_size):
    compiler = policy._compiler
    action_size = compiler.rddl.action_size
    policy_variables = policy._policy_variables

    assert isinstance(policy_variables, tuple)
    assert len(policy_variables) == len(action_size)

    for policy_var, size in zip(policy_variables, action_size):
        assert isinstance(policy_var, tf.Variable)
        assert policy_var.shape == (batch_size, HORIZON, *size)


def test_call_parallel_plans(parallel_plans):
    policy_variables = parallel_plans._policy_variables

    compiler = parallel_plans._compiler
    state = compiler.initial_state()
    with compiler.graph.as_default():
        timestep = tf.constant(0, dtype=tf.int32, shape=(BATCH_SIZE, 1))

    action = parallel_plans(state, timestep)

    assert len(action) == len(policy_variables)
    for action_tensor, policy_var in zip(action, policy_variables):
        assert isinstance(action_tensor, tf.Tensor)
        assert action_tensor.shape == (BATCH_SIZE, *policy_var.shape[2:])

    with tf.Session(graph=compiler.graph) as sess:
        sess.run(tf.global_variables_initializer())
        actions_ = sess.run(action)

        for action_ in actions_:
            for i in range(BATCH_SIZE):
                for j in range(BATCH_SIZE):
                    if i != j:
                        assert not np.allclose(action_[i], action_[j])


def test_call_non_parallel_plans(non_parallel_plans):
    policy_variables = non_parallel_plans._policy_variables

    compiler = non_parallel_plans._compiler
    state = compiler.initial_state()
    with compiler.graph.as_default():
        timestep = tf.constant(0, dtype=tf.int32, shape=(BATCH_SIZE, 1))

    action = non_parallel_plans(state, timestep)

    assert len(action) == len(policy_variables)
    for action_tensor, policy_var in zip(action, policy_variables):
        assert isinstance(action_tensor, tf.Tensor)
        assert action_tensor.shape == (BATCH_SIZE, *policy_var.shape[2:])

    with tf.Session(graph=compiler.graph) as sess:
        sess.run(tf.global_variables_initializer())
        actions_ = sess.run(action)

        for action_ in actions_:
            for i in range(BATCH_SIZE):
                for j in range(BATCH_SIZE):
                    assert np.allclose(action_[i], action_[j])
