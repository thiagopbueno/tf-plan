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


import numpy as np
import tensorflow as tf
from typing import Any, Dict, Optional

from rddl2tf import Compiler


optimizers = {
    'Adadelta': tf.train.AdadeltaOptimizer,
    'Adagrad': tf.train.AdagradOptimizer,
    'Adam': tf.train.AdamOptimizer,
    'GradientDescent': tf.train.GradientDescentOptimizer,
    'ProximalGradientDescent': tf.train.ProximalGradientDescentOptimizer,
    'ProximalAdagrad': tf.train.ProximalAdagradOptimizer,
    'RMSProp': tf.train.RMSPropOptimizer
}


DEFAULT_CONFIG = {
    'optimizer': 'RMSProp',
    'learning_rate': 0.001,
}

class ActionOptimizer(object):

    def __init__(self, compiler: Compiler, config: Dict[str, Any]) -> None:
        self.compiler = compiler
        self.config = { **DEFAULT_CONFIG, **config }

    def build(self):
        tf_optimizer = optimizers[config['optimizer']]
        learning_rate = config['learning_rate']
        self.optimizer = tf_optimizer(learning_rate)

    def minimize(self, loss: tf.Tensor, name: Optional[str] = None) -> tf.Tensor:
        return self.optimizer.minimize(loss, name)
