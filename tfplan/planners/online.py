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


from tfrddlsim.rddl2tf.compiler import Compiler

from tfplan.train.policy import OpenLoopPolicy
from tfplan.train.optimizer import ActionOptimizer

import numpy as np
import tensorflow as tf

from typing import Sequence, Tuple

StateTensor = Sequence[tf.Tensor]
Action = Sequence[np.array]
PolicyVars = Sequence[np.array]


class OnlineOpenLoopPlanner(object):
    '''OnlineOpenLoopPlanner implements a gradient-based planner that optimizes
    a sequence of actions in an online setting (i.e., interleaving planning and
    execution).

    Args:
        compiler (:obj:`tfrddlsim.rddl2tf.compiler.Compiler`): A RDDL2TensorFlow compiler.
        batch_size (int): The size of the batch used in policy simulation.
        horizon (int): The number of timesteps.
        parallel_plans (bool): The boolean flag for optimizing parallel sequence of actions.
    '''

    def __init__(self,
            compiler: Compiler,
            batch_size: int,
            horizon: int,
            parallel_plans: bool = True) -> None:
        self._compiler = compiler
        self.batch_size = batch_size
        self.horizon = horizon
        self.parallel_plans = parallel_plans

    def build(self,
            learning_rate: int,
            epochs: int,
            show_progress: bool = True) -> None:
        '''Builds the online open loop planning ops.'''
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.show_progress = show_progress
        self._build_policy_graph()
        self._build_optimizer_graph()

    def _build_policy_graph(self) -> None:
        '''Builds the open loop policy ops.'''
        self._policy = OpenLoopPolicy(self._compiler, self.batch_size, self.horizon, self.parallel_plans)
        self._policy.build('planning')

    def _build_optimizer_graph(self) -> None:
        '''Builds the action optimizer ops.'''
        self._optimizer = ActionOptimizer(self._compiler, self._policy)

    def __call__(self, state: StateTensor, t: int) -> Tuple[Action, PolicyVars]:
        '''Returns action to be executed in current `state` at timestep `t`.

        Args:
            state (StateTensor): The current state.
            t (int): The current timestep.

        Returns:
            Tuple[ActionArray, PolicyVarsArray]: The action and
            policy variables optimized for the current timestep.
        '''

        # initialize action optimizer
        with self._compiler.graph.as_default():
            with tf.name_scope('timestep{}'.format(t)):
                self._optimizer.build(self.learning_rate, self.batch_size, self.horizon - t, parallel_plans=False)

        # optimize next action
        initial_state = tuple(np.stack([fluent[0]] * self.batch_size) for fluent in state)
        actions, policy_vars = self._optimizer.run(self.epochs, initial_state, self.show_progress)

        # outputs
        action = tuple(np.expand_dims(fluent[0], axis=0) for fluent in actions)
        policy_vars = tuple(np.expand_dims(var[(self.horizon-1) - t], axis=0) for var in policy_vars)

        return action, policy_vars