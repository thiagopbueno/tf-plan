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

from tfrddlsim.compiler import Compiler
from tfrddlsim.simulator import Simulator

import numpy as np
import tensorflow as tf

from typing import List


class ActionOptimizer(object):
    '''ActionOptimizer implements a wrapper around RMSProp for optimizing actions.

    It uses tfrddlsim package to generate trajectories for the RDDL MDP, and
    optimizes the variables of an open loop policy in order to maximize the
    total (undiscounted) reward received from start state.

    Note:
        For details please refer to NIPS 2017 paper:
        "Scalable Planning with Tensorflow for Hybrid Nonlinear Domains".

    Args:
        compiler (:obj:`tfrddlsim.compiler.Compiler`): A RDDL2TensorFlow compiler.
        policy (:obj:`tfplan.train.policy.OpenLoopPolicy`): A sequence of actions
        implemented as an open loop policy.
    '''

    def __init__(self, compiler: Compiler, policy: OpenLoopPolicy) -> None:
        self._compiler = compiler
        self._policy = policy

    @property
    def graph(self) -> tf.Graph:
        '''Returns the compiler's graph.'''
        return self._compiler.graph

    @property
    def batch_size(self):
        '''Returns the policy's batch size.'''
        return self._policy._batch_size

    def build(self, horizon: int, learning_rate: float = 0.001) -> None:
        '''Builds all graph operations necessary for optimizing actions.

        Args:
            horizon (int): The number of timesteps.
            learning_rate (int): The learning rate passed to the underlying optimizer.
        '''
        with self.graph.as_default():
            self._build_trajectory_graph(horizon)
            self._build_loss_graph()
            self._build_optimization_graph(learning_rate)

    def run(self, epochs: int, show_progress: bool = True) -> List[np.float32]:
        '''Runs the optimization ops for a given number of training `epochs`.

        Args:
            epochs (int): The number of training epochs.
            show_progress (bool): The boolean flag for showing intermediate results.

        Returns:
            List[np.float32]: The loss function over time.
        '''
        with tf.Session(graph=self.graph) as sess:
            sess.run(tf.global_variables_initializer())
            losses = []
            for step in range(epochs):
                _, loss = sess.run([self._train_op, self.loss])
                losses.append(loss)
                if show_progress:
                    print('Epoch {0:5}: loss = {1}\r'.format(step, loss), end='')
            return losses

    def _build_trajectory_graph(self, horizon: int) -> None:
        '''Builds the (state, action, interm, reward) trajectory ops.'''
        simulator = Simulator(self._compiler, self._policy, self.batch_size)
        trajectories = simulator.trajectory(horizon)
        self.states = trajectories[1]
        self.actions = trajectories[2]
        self.rewards = trajectories[4]

    def _build_loss_graph(self) -> None:
        '''Builds the loss ops.'''
        self.total_reward = tf.reduce_sum(self.rewards, axis=1)
        self.avg_total_reward = tf.reduce_mean(self.total_reward)
        self.loss = -self.avg_total_reward

    def _build_optimization_graph(self, learning_rate: float) -> None:
        '''Builds the training ops.'''
        self._optimizer = tf.train.RMSPropOptimizer(learning_rate)
        self._train_op = self._optimizer.minimize(self.loss)
