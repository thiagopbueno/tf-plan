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

from rddl2tf.compiler import Compiler
from tfrddlsim.simulation.policy_simulator import PolicySimulator

import sys
import numpy as np
import tensorflow as tf

from typing import List, Optional, Sequence


class ActionOptimizer(object):
    '''ActionOptimizer implements a wrapper around RMSProp for optimizing actions.

    It uses tfrddlsim package to generate trajectories for the MDP, and
    optimizes the variables of an open loop policy in order to maximize the
    total (undiscounted) reward received from start state.

    Note:
        For details please refer to NIPS 2017 paper:
        "Scalable Planning with Tensorflow for Hybrid Nonlinear Domains".

    Args:
        compiler (:obj:`rddl2tf.compiler.Compiler`): A RDDL2TensorFlow compiler.
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

    def build(self,
            learning_rate: float,
            batch_size: int,
            horizon: int,
            parallel_plans: Optional[bool] = True) -> None:
        '''Builds all graph operations necessary for optimizing actions.

        Args:
            learning_rate (int): The learning rate passed to the underlying optimizer.
            batch_size (int): The simulation batch size.
            horizon (int): The number of timesteps in a trajectory.
        '''
        with self.graph.as_default():
            with tf.name_scope('action_optimizer'):
                self._build_trajectory_graph(horizon, batch_size)
                self._build_loss_graph()
                self._build_optimization_graph(learning_rate)
                self._build_solution_graph(parallel_plans)

    def run(self,
            epochs: int,
            initial_state: Optional[Sequence[np.array]] = None,
            show_progress: bool = True) -> List[np.array]:
        '''Runs the optimization ops for a given number of training `epochs`.

        Args:
            epochs (int): The number of training epochs.
            show_progress (bool): The boolean flag for showing intermediate results.

        Returns:
            List[np.array], List[np.array]: A tuple with the optimized actions and policy variables.
        '''
        with tf.Session(graph=self.graph) as sess:
            sess.run(tf.global_variables_initializer())

            feed_dict = None
            if initial_state is not None:
                feed_dict = { self.initial_state: initial_state }

            solution = None
            policy_variables = None
            best_reward = float(-sys.maxsize)

            for step in range(epochs):
                _, loss, total_reward = sess.run(
                    [self._train_op, self.loss, self._best_total_reward],
                    feed_dict=feed_dict)

                if total_reward > best_reward:
                    best_reward = total_reward
                    solution, policy_variables = sess.run(
                        [self._best_solution, self._best_variables],
                        feed_dict=feed_dict)

                if show_progress:
                    print('Epoch {0:5}: loss = {1:3.6f}\r'.format(step, loss), end='')

            return solution, policy_variables

    def _build_trajectory_graph(self, horizon: int, batch_size: int) -> None:
        '''Builds the (state, action, interm, reward) trajectory ops.'''
        simulator = PolicySimulator(self._compiler, self._policy, batch_size)
        trajectories = simulator.trajectory(horizon)
        self.initial_state = trajectories[0]
        self.states = trajectories[1]
        self.actions = trajectories[2]
        self.rewards = trajectories[4]

    def _build_loss_graph(self) -> None:
        '''Builds the loss ops.'''
        self.total_reward = tf.squeeze(tf.reduce_sum(self.rewards, axis=1))
        self.avg_total_reward = tf.reduce_mean(self.total_reward)
        self.loss = tf.square(self.avg_total_reward)

    def _build_optimization_graph(self, learning_rate: float) -> None:
        '''Builds the training ops.'''
        self._optimizer = tf.train.RMSPropOptimizer(learning_rate)
        self._train_op = self._optimizer.minimize(self.loss)

    def _build_solution_graph(self, parallel_plans: bool) -> None:
        '''Builds ops for getting best solution and corresponding policy variables.'''
        self._best_idx = tf.argmax(self.total_reward, output_type=tf.int32)
        self._best_total_reward = self.total_reward[self._best_idx]
        self._best_solution = tuple(fluent[self._best_idx] for fluent in self.actions)
        self._best_variables = self._policy[self._best_idx] if parallel_plans else self._policy[0]
