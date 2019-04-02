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


import rddl2tf
from rddl2tf.compiler import Compiler
from tfrddlsim.simulation.transition_simulator import ActionSimulationCell

import numpy as np
import tensorflow as tf

from typing import Callable, Sequence, Tuple

StateTensor = Sequence[tf.Tensor]

NonFluentsArray = Sequence[np.array]
StateArray = Sequence[np.array]
ActionArray = Sequence[np.array]
PolicyVarsArray = Sequence[np.array]

Planner = Callable[[StateTensor, int], Tuple[ActionArray, PolicyVarsArray]]


class OnlinePlanning(object):
    '''OnlinePlanning implements the plan-execute-monitor cycle.

    In each timestep it calls the `planner` for the next action
    to be executed in the current state. The environment returns
    with the next state and the transition reward.

    Args:
        compiler (:obj:`rddl2tf.compiler.Compiler`): A RDDL2TensorFlow compiler.
        planner (Callable[[StateTensor, int], Tuple[ActionArray, PolicyVarsArray]]): The online planner.
    '''

    def __init__(self, compiler: Compiler, planner: Planner) -> None:
        self._compiler = compiler
        self._planner = planner

    @property
    def graph(self) -> tf.Graph:
        '''Returns the compiler's graph.'''
        return self._compiler.graph

    def build(self) -> None:
        '''Builds the plan-execute-monitor graph ops.'''
        with self.graph.as_default():
            self._build_execution_graph()

    def run(self,
            horizon: int,
            show_progress: bool = True) -> Tuple[ActionArray, PolicyVarsArray]:
        '''Runs the plan-execute-monitor cycle for the given `horizon`.

        Args:
            horizon (int): The number of timesteps.
            show_progress (bool): The boolean flag for showing current progress.

        Returns:
            Tuple[ActionArray, PolicyVarsArray]: The sequence of actions and
            policy variables optimized and executed during the interaction
            with the environment.
        '''

        # initialize solution
        states, actions, interms, rewards = self._initialize_trajectory(horizon)

        # initialize non fluents
        non_fluents = self._initialize_non_fluents()

        # initialize initial state
        initial_state = self._initialize_state()

        building_times, optimization_times = [], []

        state = initial_state
        for step in range(horizon):

            # plan
            action, _, time1, time2 = self._planner(state, step)
            building_times.append(time1)
            optimization_times.append(time2)

            # execute
            with tf.Session(graph=self.graph) as sess:

                feed_dict = {
                    self.state: state,
                    self.action: self._feed_action(action)
                }
                next_state, interm_state, reward = sess.run(
                    [self.next_state, self.interm_state, self.reward],
                    feed_dict=feed_dict)

            # monitor
            rewards[step] = reward

            for i, (fluent) in enumerate(next_state):
                states[i][1][0][step] = fluent

            for i, (fluent) in enumerate(action):
                actions[i][1][0][step] = fluent

            for i, (fluent) in enumerate(interm_state):
                interms[i][1][0][step] = fluent

            # update state
            state = next_state

        trajectories = non_fluents, initial_state, states, actions, interms, rewards

        stats = {
            'build': building_times,
            'optimization': optimization_times
        }
        return trajectories, stats

    def _build_execution_graph(self) -> None:
        '''Builds the execution graph ops.'''
        self._transition = ActionSimulationCell(self._compiler)
        self.state = self._build_state_inputs()
        self.action = self._build_action_inputs()
        output, self.next_state = self._transition(self.action, self.state)
        _, _, self.interm_state, self.reward = output

    def _build_state_inputs(self) -> Sequence[tf.Tensor]:
        '''Builds and returns the current state fluents as placeholders.'''
        fluents = self._compiler.rddl.domain.state_fluent_ordering
        sizes = self._compiler.rddl.state_size
        dtypes = map(rddl2tf.utils.range_type_to_dtype, self._compiler.rddl.state_range_type)
        state_inputs = []
        for fluent, size, dtype in zip(fluents, sizes, dtypes):
            shape = [1, *size]
            state_inputs.append(tf.placeholder(dtype, shape=shape))
        return tuple(state_inputs)

    def _build_action_inputs(self) -> Sequence[tf.Tensor]:
        '''Builds and returns the action fluents as placeholders.'''
        fluents = self._compiler.rddl.domain.action_fluent_ordering
        sizes = self._compiler.rddl.action_size
        dtypes = map(rddl2tf.utils.range_type_to_dtype, self._compiler.rddl.action_range_type)
        action_inputs = []
        for fluent, size, dtype in zip(fluents, sizes, dtypes):
            shape = [1, *size]
            action_inputs.append(tf.placeholder(dtype, shape=shape))
        return tuple(action_inputs)

    def _initialize_non_fluents(self) -> NonFluentsArray:
        '''Returns non fluents.'''
        with tf.Session(graph=self.graph) as sess:
            non_fluents = tuple(nf.tensor for _, nf in self._compiler.non_fluents)
            non_fluents = sess.run(non_fluents)
            non_fluents = tuple(zip(self._compiler.rddl.domain.non_fluent_ordering, non_fluents))
            return non_fluents

    def _initialize_state(self) -> StateArray:
        '''Returns initial state.'''
        with tf.Session(graph=self.graph) as sess:
            initial_state = self._compiler.compile_initial_state(batch_size=1)
            initial_state = sess.run(initial_state)
            return initial_state

    def _initialize_trajectory(self, horizon: int) -> Tuple[ActionArray, PolicyVarsArray]:
        '''Returns placeholder arrays for states, actions, and interm-states.'''
        states = self._initialize_placeholders(
            horizon,
            self._compiler.rddl.domain.state_fluent_ordering,
            self._compiler.rddl.state_size,
            map(rddl2tf.utils.range_type_to_dtype, self._compiler.rddl.state_range_type))

        actions = self._initialize_placeholders(
            horizon,
            self._compiler.rddl.domain.action_fluent_ordering,
            self._compiler.rddl.action_size,
            map(rddl2tf.utils.range_type_to_dtype, self._compiler.rddl.action_range_type))

        interms = self._initialize_placeholders(
            horizon,
            self._compiler.rddl.domain.interm_fluent_ordering,
            self._compiler.rddl.interm_size,
            map(rddl2tf.utils.range_type_to_dtype, self._compiler.rddl.interm_range_type))

        rewards = np.zeros([horizon], dtype=np.float32)

        return states, actions, interms, rewards

    def _initialize_placeholders(self, horizon, names, sizes, dtypes):
        '''Returns placeholder arrays with given fluent's `names`, `sizes` and `dtypes`.'''
        placeholder = []
        for name, size, dtype in zip(names, sizes, dtypes):
            shape = [1, horizon] + list(size) # [batch_size, horizon, fluent_shape]
            placeholder.append((name, np.zeros(shape, dtype=np.float32))) # TODO: use dtype parameter
        return placeholder

    def _feed_action(self, action):
        actions = []
        for size, a in zip(self._compiler.rddl.action_size, action):
            if size != ():
                a = np.expand_dims(a, axis=0)
                actions.append(a)
            else:
                actions.append(a)
        return tuple(actions)
