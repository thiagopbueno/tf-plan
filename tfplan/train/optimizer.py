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


OPTIMIZERS = {
    "Adadelta": tf.train.AdadeltaOptimizer,
    "Adagrad": tf.train.AdagradOptimizer,
    "Adam": tf.train.AdamOptimizer,
    "GradientDescent": tf.train.GradientDescentOptimizer,
    "ProximalGradientDescent": tf.train.ProximalGradientDescentOptimizer,
    "ProximalAdagrad": tf.train.ProximalAdagradOptimizer,
    "RMSProp": tf.train.RMSPropOptimizer,
}


DEFAULT_CONFIG = {"optimizer": "RMSProp", "learning_rate": 0.001}


class ActionOptimizer:
    """ActionOptimizer wraps a TensorFlow optimizer.

    Args:
        config (Dict[str, Any]): The optimizer config dict.
    """

    def __init__(self, config):
        self.config = config

        self.optimizer = None

    def build(self):
        """Builds the underlying optimizer."""
        tf_optimizer = OPTIMIZERS[self.config["optimizer"]]
        learning_rate = self.config["learning_rate"]
        self.optimizer = tf_optimizer(learning_rate)

    def minimize(self, loss, name):
        """Returns the train op corresponding to the loss minimization."""
        return self.optimizer.minimize(loss, name)
