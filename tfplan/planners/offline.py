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

from rddl2tf.compiler import Compiler

from tfplan.train.policy import OpenLoopPolicy
from tfplan.train.optimizer import ActionOptimizer

import numpy as np
import tensorflow as tf

from typing import Sequence, Tuple

ActionArray = Sequence[np.array]
PolicyVarsArray = Sequence[np.array]


class OfflineOpenLoopPlanner(object):
    '''OfflineOpenLoopPlanner implements a gradient-based planner that optimizes
    a sequence of actions in an offline setting.

    Note:
        For details please refer to NIPS 2017 paper:
        "Scalable Planning with Tensorflow for Hybrid Nonlinear Domains".

    Args:
        compiler (:obj:`rddl2tf.compiler.Compiler`): A RDDL2TensorFlow compiler.
        batch_size (int): The size of the batch used in policy simulation.
        horizon (int): The number of timesteps.
    '''

    def __init__(self, compiler: Compiler, batch_size: int, horizon: int) -> None:
        self._compiler = compiler
        self.batch_size = batch_size
        self.horizon = horizon

    def build(self, learning_rate: int) -> None:
        '''Builds the offline open loop planning ops.'''
        self._build_policy_graph()
        self._build_optimizer_graph(learning_rate)

    def _build_policy_graph(self) -> None:
        '''Builds the open loop policy ops.'''
        self._policy = OpenLoopPolicy(self._compiler, self.batch_size, self.horizon)
        self._policy.build('planning')

    def _build_optimizer_graph(self, learning_rate: int) -> None:
        '''Builds the action optimizer ops.'''
        self._optimizer = ActionOptimizer(self._compiler, self._policy)
        self._optimizer.build(learning_rate, self.batch_size, self.horizon)

    def run(self,
            epochs: int,
            show_progress: bool = True) -> Tuple[ActionArray, PolicyVarsArray]:
        '''Runs action optimizer for the given number of training `epochs`.

        Args:
            epochs (int): The number of training epochs.
            show_progress (bool): The boolean flag for showing current progress.

        Returns:
            Tuple[ActionArray, PolicyVarsArray]: The sequence of actions and
            policy variables optimized after training.
        '''
        actions, policy_vars = self._optimizer.run(epochs, show_progress=show_progress)
        return actions, policy_vars
