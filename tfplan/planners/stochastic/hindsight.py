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
from tfplan.planners.stochastic import utils
from tfplan.planners.stochastic.simulation import SimulationCell, Simulator
from tfplan.train.policy import OpenLoopPolicy


class HindsightPlanner(StochasticPlanner):
    """HindsightPlanner class implements an online gradient-based
    planner that chooses the next action based on the upper bound of
    the Value function of the current state.

    Args:
        rddl (str): A RDDL domain/instance filepath or rddlgym id.
        config (Dict[str, Any]): The planner config dict.
    """

    # pylint: disable=too-many-instance-attributes

    def __init__(self, rddl, config):
        super().__init__(rddl, ReparameterizationCompiler, config)

        self.base_policy = None
        self.scenario_policy = None

        self.next_state = None

        self.cell = None
        self.cell_noise = None
        self.cell_samples = None
        self.action = None
        self.reward = None

        self.simulator = None
        self.trajectory = None
        self.final_state = None
        self.scenario_total_reward = None
        self.total_reward = None

        self.avg_total_reward = None
        self.loss = None

        self.writer = None
        self.summaries = None

    def _build_policy_ops(self):
        self._build_base_policy_ops()
        self._build_scenario_policy_ops()

    def _build_base_policy_ops(self):
        horizon = 1
        self.base_policy = OpenLoopPolicy(self.compiler, horizon, parallel_plans=False)
        self.base_policy.build("base_policy")

    def _build_scenario_policy_ops(self):
        horizon = self.config["horizon"] - 1
        self.scenario_policy = OpenLoopPolicy(
            self.compiler, horizon, parallel_plans=True
        )
        self.scenario_policy.build("scenario_policy")

    def _build_trajectory_ops(self):
        self._build_scenario_start_states_ops()
        self._build_scenario_trajectory_ops()

    def _build_scenario_start_states_ops(self):
        with tf.name_scope("current_action"):

            with tf.name_scope("reparameterization"):
                reparameterization_map = self.compiler.get_cpfs_reparameterization()
                self.cell_samples = utils.get_noise_samples(
                    reparameterization_map, self.batch_size, horizon=1
                )
                self.cell_noise, encoding = utils.encode_noise_samples_as_inputs(
                    self.cell_samples
                )

            self.cell = SimulationCell(
                self.compiler, self.base_policy, config={"encoding": encoding}
            )

            timesteps = tf.zeros((self.batch_size, 1), dtype=tf.float32)

            inputs = tf.concat([timesteps, self.cell_noise[:, 0, ...]], axis=1)
            output, self.next_state = self.cell(inputs, self.initial_state)

            self.action = output[1]
            self.reward = tf.squeeze(output[3])

    def _build_scenario_trajectory_ops(self):
        with tf.name_scope("scenarios"):
            self.simulator = Simulator(self.compiler, self.scenario_policy, config=None)
            self.simulator.build()
            (
                self.trajectory,
                self.final_state,
                self.scenario_total_reward,
            ) = self.simulator.trajectory(self.next_state, self.sequence_length)

    def _build_loss_ops(self):
        with tf.name_scope("loss"):
            self.total_reward = self.reward + self.scenario_total_reward
            self.avg_total_reward = tf.reduce_mean(self.total_reward)
            self.loss = tf.square(self.avg_total_reward)

    def _build_summary_ops(self):
        if self.config["verbose"]:

            with tf.name_scope("summary"):
                _ = tf.compat.v1.summary.FileWriter(self.config["logdir"], self.graph)
                tf.compat.v1.summary.scalar("avg_total_reward", self.avg_total_reward)
                tf.compat.v1.summary.scalar("loss", self.loss)

                tf.compat.v1.summary.histogram("reward", self.reward)
                tf.compat.v1.summary.histogram(
                    "scenario_total_reward", self.scenario_total_reward
                )
                tf.compat.v1.summary.histogram("total_reward", self.total_reward)
                tf.compat.v1.summary.histogram("next_state_noise", self.cell_noise)
                tf.compat.v1.summary.histogram("scenario_noise", self.simulator.noise)

                for grad, variable in self.grads_and_vars:
                    var_name = variable.name
                    tf.compat.v1.summary.histogram(f"{var_name}_grad", grad)
                    tf.compat.v1.summary.histogram(var_name, variable)

                self.summaries = tf.compat.v1.summary.merge_all()

    def __call__(self, state, timestep):
        next_state_noise = utils.evaluate_noise_samples_as_inputs(
            self._sess, self.cell_samples
        )
        scenario_noise = utils.evaluate_noise_samples_as_inputs(
            self._sess, self.simulator.samples
        )

        feed_dict = {
            self.initial_state: self._get_batch_initial_state(state),
            self.cell_noise: next_state_noise,
            self.simulator.noise: scenario_noise,
            self.steps_to_go: self.config["horizon"] - timestep - 1,
        }

        self.run(timestep, feed_dict)

        action = self._get_action(self.action, feed_dict)
        return action
