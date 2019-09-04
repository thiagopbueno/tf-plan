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


import abc
import numpy as np
from typing import Any, Dict, Sequence

from rddl2tf import Compiler

from tfplan.planners.planner import Planner
from tfplan.train.policy import OpenLoopPolicy
from tfplan.planners.stochastic.simulation import Simulator
from tfplan.train.optimizers import ActionOptimizer


Action = Sequence[np.ndarray]
State = Sequence[np.ndarray]
StateTensor = Sequence[tf.Tensor]


DEFAULT_CONFIG = {
    'epochs': 200
}


class StraightLinePlanner(Planner):

    def __init__(self, compiler: Compiler, config: Dict[str, Any]) -> None:
        super(Planner, self).__init__(compiler, { **DEFAULT_CONFIG, **config })

    def build(self, horizon: int) -> None:
        with self.graph.as_default():
            self._build_policy_ops(horizon)
            self._build_initial_state_ops()
            self._build_trajectory_ops()
            self._build_loss_ops()
            self._build_optimization_ops()

    def _build_policy_ops(self, horizon: int):
        self.policy = OpenLoopPolicy(self.compiler, horizon, parallel_plans=False)
        self.policy.build('planning')

    def _build_initial_state_ops(self):
        self.initial_state = tuple(tf.placeholder(t.dtype, t.shape) for t in self.compiler.initial_state())

    def _build_trajectory_ops(self):
        self.simulator = Simulator(self.compiler, self.policy, config=None)
        self.simulator.build()
        self.trajectory, self.final_state, self.total_reward = self.simulator.trajectory(self.initial_state)

    def _build_loss_ops(self):
        with tf.name_scope('loss'):
            self.avg_total_reward = tf.reduce_mean(self.total_reward)
            self.loss = tf.square(self.avg_total_reward)

    def _build_optimization_ops(self):
        self.optimizer = ActionOptimizer(self.compiler, self.config['optimization'])
        self.train_op = self.optimizer.minimize(self.loss)

    def __call__(self, state: State, t: int) -> Action:

        with tf.Session(graph=self.graph) as sess:

            # init
            sess.run(tf.global_variables_initializer())

            # sample scenarios
            samples = utils.evaluate_noise_samples_as_inputs(sess, self.simulator.samples)

            # fix current state and scenarios
            feed_dict = {
                self.initial_state: state,
                self.simulator.noise: samples
            }

            # optimize policy variables
            for step in range(self.config['epochs']):
                _, loss = sess.run([self._train_op, self.loss], feed_dict=feed_dict)

            # select action
            actions = sess.run(self.trajectory.actions, feed_dict=feed_dict)
            action = tuple(fluent[0] for fluent in actions)

        return action
