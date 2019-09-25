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

# pylint: disable=missing-docstring


from collections import OrderedDict
import numpy as np
import tensorflow as tf

from rddl2tf.compilers import ReparameterizationCompiler

from tfplan.planners.planner import Planner
from tfplan.train.policy import OpenLoopPolicy
from tfplan.planners.stochastic.simulation import Simulator
from tfplan.planners.stochastic import utils
from tfplan.train.optimizer import ActionOptimizer


class StraightLinePlanner(Planner):
    """StraightLinePlanner class implements the Planner interface
    for the online gradient-based planner that chooses the next action
    based on the lower bound objective function.

    Args:
        model (pyrddl.rddl.RDDL): A RDDL model.
        config (Dict[str, Any]): The planner config dict.
    """

    # pylint: disable=too-many-instance-attributes

    def __init__(self, model, config):
        compiler = ReparameterizationCompiler(model, batch_size=config["batch_size"])
        super(StraightLinePlanner, self).__init__(compiler, config)

        self.policy = None

        self.initial_state = None

        self.steps_to_go = None
        self.sequence_length = None

        self.simulator = None
        self.trajectory = None
        self.final_state = None
        self.total_reward = None

        self.avg_total_reward = None
        self.loss = None

        self.optimizer = None
        self.train_op = None

    def build(self,):
        with self.graph.as_default():
            self._build_policy_ops()
            self._build_initial_state_ops()
            self._build_sequence_length_ops()
            self._build_trajectory_ops()
            self._build_loss_ops()
            self._build_optimization_ops()

    def _build_policy_ops(self):
        horizon = self.config["horizon"]
        self.policy = OpenLoopPolicy(self.compiler, horizon, parallel_plans=False)
        self.policy.build("planning")

    def _build_initial_state_ops(self):
        self.initial_state = tuple(
            tf.placeholder(t.dtype, t.shape) for t in self.compiler.initial_state()
        )

    def _build_sequence_length_ops(self):
        self.steps_to_go = tf.placeholder(tf.int32, shape=())
        self.sequence_length = tf.tile(
            tf.reshape(self.steps_to_go, [1]), [self.compiler.batch_size]
        )

    def _build_trajectory_ops(self):
        self.simulator = Simulator(self.compiler, self.policy, config=None)
        self.simulator.build()
        self.trajectory, self.final_state, self.total_reward = self.simulator.trajectory(
            self.initial_state, self.sequence_length
        )

    def _build_loss_ops(self):
        with tf.name_scope("loss"):
            self.avg_total_reward = tf.reduce_mean(self.total_reward)
            self.loss = tf.square(self.avg_total_reward)

    def _build_optimization_ops(self):
        self.optimizer = ActionOptimizer(self.config["optimization"])
        self.optimizer.build()
        self.train_op = self.optimizer.minimize(self.loss)

    def __call__(self, state, timestep):

        with tf.Session(graph=self.graph) as sess:
            sess.run(tf.global_variables_initializer())

            feed_dict = {
                self.initial_state: self._get_batch_initial_state(state),
                self.simulator.noise: self._get_noise_samples(sess),
                self.steps_to_go: self.config["horizon"] - timestep,
            }

            for _ in range(self.config["epochs"]):
                _, _ = sess.run([self.train_op, self.loss], feed_dict=feed_dict)

            action = self._get_action(sess, feed_dict)

        return action

    def _get_batch_initial_state(self, state):
        batch_size = self.compiler.batch_size
        return tuple(
            map(
                lambda fluent: np.tile(
                    fluent, (batch_size, *([1] * len(fluent.shape)))
                ),
                state.values(),
            )
        )

    def _get_noise_samples(self, sess):
        samples = utils.evaluate_noise_samples_as_inputs(sess, self.simulator.samples)
        return samples

    def _get_action(self, sess, feed_dict):
        action_fluent_ordering = self.compiler.rddl.domain.action_fluent_ordering
        actions = sess.run(self.trajectory.actions, feed_dict=feed_dict)
        action = OrderedDict(
            {
                name: fluent[0][0]
                for name, fluent in zip(action_fluent_ordering, actions)
            }
        )
        return action
