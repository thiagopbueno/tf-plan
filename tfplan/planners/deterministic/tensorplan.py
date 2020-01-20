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
import os

import tensorflow as tf
from tqdm import trange

from rddl2tf.compilers import DefaultCompiler

from tfplan.planners.planner import Planner
from tfplan.planners.deterministic.simulation import Simulator
from tfplan.train.policy import OpenLoopPolicy
from tfplan.train.optimizer import ActionOptimizer


class Tensorplan(Planner):
    """Tensorplan class implements the Planner interface
    for the offline gradient-based planner (i.e., tensorplan).

    Args:
        model (pyrddl.rddl.RDDL): A RDDL model.
        config (Dict[str, Any]): The planner config dict.
    """

    # pylint: disable=too-many-instance-attributes

    def __init__(self, rddl, config):
        super(Tensorplan, self).__init__(rddl, DefaultCompiler, config)

        self.policy = None
        self.initial_state = None

        self.simulator = None
        self.trajectory = None
        self.final_state = None
        self.total_reward = None

        self.avg_total_reward = None
        self.loss = None

        self.optimizer = None
        self.train_op = None

        self.best_plan_idx = None
        self.best_plan = None

        self._plan = None

        self.writer = None
        self.summaries = None

    @property
    def logdir(self):
        return self.config.get("logdir") or f"/tmp/tfplan/tensorplan/{self.rddl}"

    def build(self):
        """Builds planner ops."""
        with self.graph.as_default():
            self._build_policy_ops()
            self._build_initial_state_ops()
            self._build_trajectory_ops()
            self._build_loss_ops()
            self._build_optimization_ops()
            self._build_solution_ops()
            self._build_summary_ops()
            self._build_init_ops()

    def _build_init_ops(self):
        self.init_op = tf.global_variables_initializer()

    def _build_policy_ops(self):
        horizon = self.config["horizon"]
        self.policy = OpenLoopPolicy(self.compiler, horizon, parallel_plans=True)
        self.policy.build("tensorplan")

    def _build_initial_state_ops(self):
        self.initial_state = self.compiler.initial_state()

    def _build_trajectory_ops(self):
        self.simulator = Simulator(self.compiler, self.policy)
        self.simulator.build()
        self.trajectory, self.final_state, self.total_reward = self.simulator.trajectory(
            self.initial_state
        )

    def _build_loss_ops(self):
        with tf.name_scope("loss"):
            self.avg_total_reward = tf.reduce_mean(self.total_reward)
            self.loss = tf.square(self.avg_total_reward)

    def _build_optimization_ops(self):
        self.optimizer = ActionOptimizer(self.config["optimization"])
        self.optimizer.build()
        self.train_op = self.optimizer.minimize(self.loss)

    def _build_solution_ops(self):
        self.best_plan_idx = tf.argmax(self.total_reward, axis=0)
        self.best_plan = tuple(
            action[self.best_plan_idx] for action in self.trajectory.actions
        )

    def _build_summary_ops(self):
        tf.compat.v1.summary.histogram("total_reward", self.total_reward)
        tf.compat.v1.summary.scalar("avg_total_reward", self.avg_total_reward)
        tf.compat.v1.summary.scalar("loss", self.loss)
        self.summaries = tf.compat.v1.summary.merge_all()

    def run(self):
        """Run the planner for the given number of epochs.

        Returns:
            plan (Sequence(np.ndarray): The best solution plan.
        """
        self.writer = tf.compat.v1.summary.FileWriter(self.logdir, self.graph)

        self._sess.run(self.init_op)

        run_id = self.config.get("run_id", 0)
        pid = os.getpid()
        position = run_id % self.config.get("num_workers", 1)
        epochs = self.config["epochs"]
        desc = f"(pid={pid}) Run #{run_id:<3d}"

        with trange(
            epochs, desc=desc, unit="epoch", position=position, leave=False
        ) as t:

            for step in t:
                _, loss_, avg_total_reward_, summary_ = self._sess.run(
                    [self.train_op, self.loss, self.avg_total_reward, self.summaries]
                )

                self.writer.add_summary(summary_, step)

                t.set_postfix(
                    loss=f"{loss_:10.4f}", avg_total_reward=f"{avg_total_reward_:10.4f}"
                )

        self.writer.close()

        plan_ = self._sess.run(self.best_plan)
        return plan_

    def __call__(self, state, timestep):
        """Returns the action for the given `timestep`."""
        # find plan
        if self._plan is None:
            self._plan = self.run()

        # select action for given timestep
        action_fluent_ordering = self.compiler.rddl.domain.action_fluent_ordering
        action = OrderedDict(
            {
                name: action[timestep]
                for name, action in zip(action_fluent_ordering, self._plan)
            }
        )
        return action
