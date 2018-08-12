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


from tfrddlsim.simulator import Simulator

import tensorflow as tf


class ActionEvaluator(object):

    def __init__(self, compiler, policy):
        self._compiler = compiler
        self._policy = policy

    @property
    def graph(self):
        return self._compiler.graph

    @property
    def batch_size(self):
        return self._policy._batch_size

    @property
    def horizon(self):
        return self._policy._horizon

    def run(self):
        self._simulator = Simulator(self._compiler, self._policy, self.batch_size)
        return self._simulator.run(self.horizon)
