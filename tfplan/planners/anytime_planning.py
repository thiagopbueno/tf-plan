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


from tfrddlsim.simulation.transition_simulator import ActionSimulationCell

import numpy as np
import tensorflow as tf


class OnlinePlanning(object):
    '''OnlinePlanning implements the plan-execute-monitor cycle.

    Args:
        compiler:
        planner:
    '''

    def __init__(self, compiler, planner):
        self._compiler = compiler
        self._planner = planner

    @property
    def graph(self):
        return self._compiler.graph

    def build(self):
        with self.graph.as_default():
            self._build_execution_graph()

    def run(self, horizon, show_progress=True):
        actions = []
        policy_vars = []
        for size, dtype in zip(self._compiler.action_size, self._compiler.action_dtype):
            shape = [horizon] + list(size)
            actions.append(np.zeros(shape, dtype=np.float32))
            policy_vars.append(np.zeros(shape, dtype=np.float32))

        with tf.Session(graph=self.graph) as sess:
            initial_state = self._compiler.compile_initial_state(batch_size=1)
            initial_state = sess.run(initial_state)

        state = initial_state
        for step in range(horizon):

            # plan
            action, policy_var = self._planner(state, step)
            for i, (fluent, var) in enumerate(zip(action, policy_var)):
                actions[i][step] = fluent
                policy_vars[i][step] = var

            # execute
            with tf.Session(graph=self.graph) as sess:
                feed_dict = { self.state: state, self.action: action }
                next_state, reward = sess.run([self.next_state, self.reward], feed_dict=feed_dict)

            # monitor
            state = next_state

        return actions, policy_vars

    def _build_execution_graph(self):
        self._transition = ActionSimulationCell(self._compiler)
        self.state = self._build_state_inputs()
        self.action = self._build_action_inputs()
        self.reward, self.next_state = self._transition(self.action, self.state)

    def _build_state_inputs(self):
        fluents = self._compiler.state_fluent_ordering
        sizes = self._compiler.state_size
        dtypes = self._compiler.state_dtype
        state_inputs = []
        for fluent, size, dtype in zip(fluents, sizes, dtypes):
            shape = [1, *size]
            state_inputs.append(tf.placeholder(dtype, shape=shape))
        return tuple(state_inputs)

    def _build_action_inputs(self):
        fluents = self._compiler.action_fluent_ordering
        sizes = self._compiler.action_size
        dtypes = self._compiler.action_dtype
        action_inputs = []
        for fluent, size, dtype in zip(fluents, sizes, dtypes):
            shape = [1, *size]
            action_inputs.append(tf.placeholder(dtype, shape=shape))
        return tuple(action_inputs)
