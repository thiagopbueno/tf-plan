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

import tensorflow as tf

import rddlgym


DEFAULT_CONFIG = {
    "batch_size": 128,
    "epochs": 200,
    "optimization": {"optimizer": "RMSProp", "learning_rate": 0.001},
}


class Planner(metaclass=abc.ABCMeta):
    """Planner abstract base class.

    Args:
        rddl: (pyrddl.rddl.RDDL): A RDDL
        compiler_cls (rddl2tf.Compiler): The RDDL-to-TensorFlow compiler class.
        config (Dict[str, Any]): The planner config dict.
    """

    def __init__(self, rddl, compiler_cls, config):
        self.rddl = rddl
        self.model = rddlgym.make(rddl, mode=rddlgym.AST)
        self.compiler = compiler_cls(self.model, batch_size=config["batch_size"])
        self.config = config

        self.compiler.init()

        config = tf.ConfigProto(
            inter_op_parallelism_threads=1,
            intra_op_parallelism_threads=1,
            log_device_placement=False,
        )
        self._sess = tf.Session(graph=self.graph, config=config)

    @property
    def graph(self):
        """Returns the compiler's graph."""
        return self.compiler.graph

    @property
    def batch_size(self):
        """Returns the compiler's batch size."""
        return self.compiler.batch_size

    @abc.abstractmethod
    def build(self):
        """Builds the planner."""
        raise NotImplementedError

    @abc.abstractmethod
    def __call__(self, state, timestep):
        raise NotImplementedError

    def close(self):
        self._sess.close()
