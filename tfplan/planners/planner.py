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


Action = Sequence[np.array]
StateTensor = Sequence[tf.Tensor]


class Planner(metaclass=abc.ABCMeta):

    def __init__(self, compiler: Compiler, config: Dict[str, Any]) -> None:
        self.compiler = compiler
        self.config = config

    @property
    def graph(self):
        return self.compiler.graph

    @property
    def batch_size(self):
        return self.compiler.batch_size

    @abc.abstractmethod
    def build(self, horizon: int) -> None:
        raise NotImplementedError

    @abc.abstractmethod
    def __call__(self, state: StateTensor, t: int) -> Action:
        raise NotImplementedError
