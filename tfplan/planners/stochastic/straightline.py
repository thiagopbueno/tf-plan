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

import tensorflow as tf

from rddl2tf.compilers import ReparameterizationCompiler

from tfplan.planners.stochastic import StochasticPlanner
from tfplan.train.policy import OpenLoopPolicy
from tfplan.planners.stochastic.simulation import Simulator
from tfplan.planners.stochastic import utils


class StraightLinePlanner(StochasticPlanner):
    """StraightLinePlanner class implements the online gradient-based
    planner that chooses the next action based on the lower bound of
    the Value function of the start state.

    Args:
        rddl (str): A RDDL domain/instance filepath or rddlgym id.
        config (Dict[str, Any]): The planner config dict.
    """

    # pylint: disable=too-many-instance-attributes

    def __init__(self, rddl, config):
        super().__init__(rddl, ReparameterizationCompiler, config)

        self.policy = None

        self.simulator = None
        self.trajectory = None
        self.final_state = None
        self.total_reward = None

        self.avg_total_reward = None
        self.loss = None

        self.writer = None
        self.summaries = None

    def _build_policy_ops(self):
        horizon = self.config["horizon"]
        self.policy = OpenLoopPolicy(self.compiler, horizon, parallel_plans=False)
        self.policy.build("planning")

    def _build_trajectory_ops(self):
        with tf.name_scope("scenarios"):
            self.simulator = Simulator(self.compiler, self.policy, config=None)
            self.simulator.build()
            (
                self.trajectory,
                self.final_state,
                self.total_reward,
            ) = self.simulator.trajectory(self.initial_state, self.sequence_length)

    def _build_loss_ops(self):
        with tf.name_scope("loss"):
            self.avg_total_reward = tf.reduce_mean(self.total_reward)
            self.loss = tf.square(self.avg_total_reward)

    def _build_summary_ops(self):
        if self.config["verbose"]:

            with tf.name_scope("summary"):
                _ = tf.compat.v1.summary.FileWriter(self.config["logdir"], self.graph)
                tf.compat.v1.summary.scalar("avg_total_reward", self.avg_total_reward)
                tf.compat.v1.summary.scalar("loss", self.loss)

                tf.compat.v1.summary.histogram("total_reward", self.total_reward)
                tf.compat.v1.summary.histogram("scenario_noise", self.simulator.noise)

                for grad, variable in self.grads_and_vars:
                    var_name = variable.name
                    tf.compat.v1.summary.histogram(f"{var_name}_grad", grad)
                    tf.compat.v1.summary.histogram(var_name, variable)

                self.summaries = tf.compat.v1.summary.merge_all()

    def __call__(self, state, timestep):
        scenario_noise = utils.evaluate_noise_samples_as_inputs(
            self._sess, self.simulator.samples
        )

        feed_dict = {
            self.initial_state: self._get_batch_initial_state(state),
            self.simulator.noise: scenario_noise,
            self.steps_to_go: self.config["horizon"] - timestep,
        }

        self.run(timestep, feed_dict)

        action = self._get_action(self.trajectory.actions, feed_dict)
        return action
