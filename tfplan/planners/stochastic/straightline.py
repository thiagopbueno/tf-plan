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


# from collections import OrderedDict
import os

# import numpy as np
import tensorflow as tf
from tqdm import trange

from rddl2tf.compilers import ReparameterizationCompiler

from tfplan.planners.stochastic import StochasticPlanner
from tfplan.train.policy import OpenLoopPolicy
from tfplan.planners.stochastic.simulation import Simulator
from tfplan.planners.stochastic import utils

# from tfplan.train.optimizer import ActionOptimizer


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

    @property
    def logdir(self):
        return self.config.get("logdir") or f"/tmp/tfplan/straigthline/{self.rddl}"

    def build(self,):
        with self.graph.as_default():
            self._build_policy_ops()
            self._build_initial_state_ops()
            self._build_sequence_length_ops()
            self._build_trajectory_ops()
            self._build_loss_ops()
            self._build_optimization_ops(self.loss)
            self._build_summary_ops()
            self._build_init_ops()

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
        with tf.name_scope("summary"):
            _ = tf.compat.v1.summary.FileWriter(self.logdir, self.graph)
            tf.compat.v1.summary.scalar("avg_total_reward", self.avg_total_reward)
            tf.compat.v1.summary.scalar("loss", self.loss)

            if self.config["verbose"]:
                tf.compat.v1.summary.histogram("total_reward", self.total_reward)
                tf.compat.v1.summary.histogram("scenario_noise", self.simulator.noise)

                for grad, variable in self.grads_and_vars:
                    var_name = variable.name
                    tf.compat.v1.summary.histogram(f"{var_name}_grad", grad)
                    tf.compat.v1.summary.histogram(var_name, variable)

            self.summaries = tf.compat.v1.summary.merge_all()

    def __call__(self, state, timestep):
        # pylint: disable=too-many-locals

        logdir = os.path.join(self.logdir, f"timestep={timestep}")
        self.writer = tf.compat.v1.summary.FileWriter(logdir)

        self._sess.run(self.init_op)

        run_id = self.config.get("run_id", 0)
        pid = os.getpid()
        position = run_id % self.config.get("num_workers", 1)
        epochs = self.config["epochs"]
        desc = f"(pid={pid}) Run #{run_id:<3d} / step={timestep:<3d}"

        feed_dict = {
            self.initial_state: self._get_batch_initial_state(state),
            self.simulator.noise: self._get_noise_samples(self._sess),
            self.steps_to_go: self.config["horizon"] - timestep,
        }

        with trange(
            epochs, unit="epoch", desc=desc, position=position, leave=False
        ) as t:

            for step in t:
                _, loss_, avg_total_reward_, summary_ = self._sess.run(
                    [self.train_op, self.loss, self.avg_total_reward, self.summaries],
                    feed_dict=feed_dict,
                )

                self.writer.add_summary(summary_, step)

                t.set_postfix(
                    loss=f"{loss_:10.4f}", avg_total_reward=f"{avg_total_reward_:10.4f}"
                )

        self.writer.close()

        action = self._get_action(self.trajectory.actions, feed_dict)
        return action

    def _get_noise_samples(self, sess):
        samples = utils.evaluate_noise_samples_as_inputs(sess, self.simulator.samples)
        return samples
