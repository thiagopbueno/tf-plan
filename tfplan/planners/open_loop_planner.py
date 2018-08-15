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


from tfplan.train.policy import OpenLoopPolicy
from tfplan.train.optimizer import ActionOptimizer

import numpy as np
import tensorflow as tf


class OfflineOpenLoopPlanner(object):

    def __init__(self, compiler, batch_size, horizon):
        self._compiler = compiler
        self._policy = OpenLoopPolicy(self._compiler, batch_size, horizon)
        self._optimizer = ActionOptimizer(self._compiler, self._policy)

    @property
    def horizon(self):
        return self._policy.horizon

    @property
    def batch_size(self):
        return self._policy.batch_size

    def build(self, learning_rate=0.01):
        self._policy.build('planning')
        self._optimizer.build(learning_rate)

    def run(self, epochs, show_progress=True):
        solution, policy_vars = self._optimizer.run(epochs, show_progress=show_progress)
        return solution, policy_vars


class OnlineOpenLoopPlanner(object):

    def __init__(self, compiler, batch_size, horizon):
        self._compiler = compiler
        self._policy = OpenLoopPolicy(self._compiler, batch_size, horizon)
        self._optimizer = ActionOptimizer(self._compiler, self._policy)

    @property
    def horizon(self):
        return self._policy.horizon

    @property
    def batch_size(self):
        return self._policy.batch_size

    def build(self, learning_rate=0.01, epochs=100, show_progress=True):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.show_progress = show_progress
        self._policy.build('planning')

    def __call__(self, state, t):
        with self._compiler.graph.as_default():
            with tf.name_scope('timestep{}'.format(t)):
                self._optimizer.build(self.learning_rate, self.horizon - t)

        initial_state = tuple(np.stack([fluent] * self.batch_size) for fluent in state[0])
        actions, _ = self._optimizer.run(self.epochs, initial_state, self.show_progress)
        action = tuple(np.expand_dims(fluent[0], axis=0) for fluent in actions)
        return action
