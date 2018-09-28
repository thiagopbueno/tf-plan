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


from tfplan.train.policy import OpenLoopPolicy

from rddl2tf.compiler import Compiler
from tfrddlsim.simulation.policy_simulator import PolicySimulator

import numpy as np
import tensorflow as tf

from typing import Sequence, Tuple
NonFluentsArray = Sequence[np.array]
StateArray = Sequence[np.array]
StatesArray = Sequence[np.array]
ActionsArray = Sequence[np.array]
IntermsArray = Sequence[np.array]
SimulationOutput = Tuple[NonFluentsArray, StateArray, StatesArray, ActionsArray, IntermsArray, np.array]


class ActionEvaluator(object):
    '''ActionEvaluator is a wraper for tfrddlsim.Simulator for evaluating
    an open loop policy.

    Args:
        compiler (:obj:`rddl2tf.compiler.Compiler`): A RDDL2TensorFlow compiler.
        policy (:obj:`tfplan.train.policy.OpenLoopPolicy`): A sequence of actions
        implemented as an open loop policy.
    '''

    def __init__(self, compiler: Compiler, policy: OpenLoopPolicy) -> None:
        self._compiler = compiler
        self._policy = policy

    @property
    def graph(self) -> tf.Graph:
        '''Returns the compiler's graph.'''
        return self._compiler.graph

    @property
    def batch_size(self) -> int:
        '''Returns the policy's batch size.'''
        return self._policy.batch_size

    @property
    def horizon(self) -> int:
        '''Returns the policy's horizon.'''
        return self._policy.horizon

    def run(self) -> SimulationOutput:
        '''Runs the trajectory simulation ops for the open-loop policy.

        Returns:
            Tuple[NonFluentsArray, StatesArray, ActionsArray, IntermsArray, np.array]: Simulation ouput tuple.
        '''
        self._simulator = PolicySimulator(self._compiler, self._policy, self.batch_size)
        return self._simulator.run(self.horizon)
