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

import abc
import collections
import os

import numpy as np
import pandas as pd
import tensorflow as tf
from tqdm import trange

from rddl2tf.compilers import ReparameterizationCompiler

from tfplan.planners.planner import Planner
from tfplan.train.optimizer import ActionOptimizer


class StochasticPlanner(Planner):
    """StochasticPlanner abstract class implements basic methods for
    online stochastic gradient-based planners.

    Args:
        rddl (str): A RDDL domain/instance filepath or rddlgym id.
        compiler_cls (rddl2tf.Compiler): The RDDL-to-TensorFlow compiler class.
        config (Dict[str, Any]): The planner config dict.
    """

    __metaclass__ = abc.ABCMeta

    def __init__(self, rddl, compiler_cls, config):
        super().__init__(rddl, ReparameterizationCompiler, config)

        self.initial_state = None

        self.steps_to_go = None
        self.sequence_length = None

        self.optimizer = None
        self.grads_and_vars = None

        self.avg_total_reward = None
        self.loss = None

        self.init_op = None
        self.train_op = None

        self.summaries = None

        self.stats = {"loss": pd.DataFrame()}

    def build(self,):
        with self.graph.as_default():
            self._build_policy_ops()
            self._build_initial_state_ops()
            self._build_sequence_length_ops()
            self._build_trajectory_ops()
            self._build_loss_ops()
            self._build_optimization_ops()
            self._build_summary_ops()
            self._build_init_ops()

    @abc.abstractmethod
    def __call__(self, state, timestep):
        raise NotImplementedError

    @abc.abstractmethod
    def _build_policy_ops(self):
        raise NotImplementedError

    @abc.abstractmethod
    def _build_trajectory_ops(self):
        raise NotImplementedError

    @abc.abstractmethod
    def _build_loss_ops(self):
        raise NotImplementedError

    @abc.abstractmethod
    def _build_summary_ops(self):
        raise NotImplementedError

    def _build_init_ops(self):
        self.init_op = tf.global_variables_initializer()

    def _build_initial_state_ops(self):
        with tf.name_scope("initial_state"):
            self.initial_state = tuple(
                tf.placeholder(t.dtype, t.shape) for t in self.compiler.initial_state()
            )

    def _build_sequence_length_ops(self):
        with tf.name_scope("sequence_length"):
            self.steps_to_go = tf.placeholder(tf.int32, shape=())
            self.sequence_length = tf.tile(
                tf.reshape(self.steps_to_go, [1]), [self.batch_size]
            )

    def _build_optimization_ops(self):
        with tf.name_scope("optimization"):
            self.optimizer = ActionOptimizer(self.config["optimization"])
            self.optimizer.build()
            self.grads_and_vars = self.optimizer.compute_gradients(self.loss)
            self.train_op = self.optimizer.apply_gradients(self.grads_and_vars)

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

    def _get_action(self, actions, feed_dict):
        action_fluent_ordering = self.compiler.rddl.domain.action_fluent_ordering
        actions = self._sess.run(actions, feed_dict=feed_dict)
        action = collections.OrderedDict(
            {
                name: fluent[0][0]
                for name, fluent in zip(action_fluent_ordering, actions)
            }
        )
        return action

    def run(self, timestep, feed_dict):
        self._sess.run(self.init_op)

        if self.summaries:
            logdir = os.path.join(self.config.get("logdir"), f"timestep={timestep}")
            writer = tf.compat.v1.summary.FileWriter(logdir)

        run_id = self.config.get("run_id", 0)
        pid = os.getpid()
        position = run_id % self.config.get("num_workers", 1)
        epochs = self.config["epochs"]
        desc = f"(pid={pid}) Run #{run_id:<3d} / step={timestep:<3d}"

        with trange(
            epochs, desc=desc, unit="epoch", position=position, leave=False
        ) as t:

            losses = []

            loss_ = self._sess.run(self.loss, feed_dict=feed_dict)
            losses.append(loss_)

            for step in t:
                self._sess.run(self.train_op, feed_dict=feed_dict)

                loss_, avg_total_reward_ = self._sess.run(
                    [self.loss, self.avg_total_reward], feed_dict=feed_dict
                )

                losses.append(loss_)

                if self.summaries:
                    summary_ = self._sess.run(self.summaries, feed_dict=feed_dict)
                    writer.add_summary(summary_, step)

                t.set_postfix(
                    loss=f"{loss_:10.4f}", avg_total_reward=f"{avg_total_reward_:10.4f}"
                )

            self.stats["loss"][timestep] = pd.Series(losses)

        if self.summaries:
            writer.close()

    def save_stats(self):
        for key, value in self.stats.items():
            filepath = os.path.join(self.config["logdir"], f"{key}.csv")
            value.to_csv(filepath, index=False)
