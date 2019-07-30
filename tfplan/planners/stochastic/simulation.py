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


import collections
import tensorflow as tf
from typing import Dict, Optional, Sequence, Tuple, Union

from rddl2tf import ReparameterizationCompiler
from rddl2tf.core.fluent import TensorFluent

from tfplan.train.policy import OpenLoopPolicy
from tfplan.planners.stochastic import utils


Shape = Sequence[int]
FluentPair = Tuple[str, TensorFluent]

NonFluentsTensor = Sequence[tf.Tensor]
StateTensor = Sequence[tf.Tensor]
StatesTensor = Sequence[tf.Tensor]
ActionsTensor = Sequence[tf.Tensor]
IntermsTensor = Sequence[tf.Tensor]

CellOutput = Tuple[StatesTensor, ActionsTensor, IntermsTensor, tf.Tensor]
CellState = Sequence[tf.Tensor]

OutputTuple = collections.namedtuple('OutputTuple', 'state action interm reward')
Trajectory = collections.namedtuple('Trajectory', 'states actions interms rewards')


class SimulationCell(tf.nn.rnn_cell.RNNCell):
    def __init__(self,
                 compiler: ReparameterizationCompiler,
                 policy: OpenLoopPolicy,
                 config: Optional[Dict] = None):
        self.compiler = compiler
        self.policy = policy
        self.config = config

    @property
    def state_size(self) -> Sequence[Shape]:
        '''Returns the MDP state size.'''
        return utils.cell_size(self.compiler.rddl.state_size)

    @property
    def action_size(self) -> Sequence[Shape]:
        '''Returns the MDP action size.'''
        return utils.cell_size(self.compiler.rddl.action_size)

    @property
    def interm_size(self) -> Sequence[Shape]:
        '''Returns the MDP intermediate state size.'''
        return utils.cell_size(self.compiler.rddl.interm_size)

    @property
    def output_size(self) -> Tuple[Sequence[Shape], Sequence[Shape], Sequence[Shape], int]:
        '''Returns the simulation cell output size.'''
        return (self.state_size, self.action_size, self.interm_size, 1)

    def __call__(self,
                 inputs: tf.Tensor,
                 state: Sequence[tf.Tensor],
                 scope: Optional[str] = None) -> Tuple[CellOutput, CellState]:
        '''Returns the cell's output tuple and next state tensors.

        Output tuple packs together the next state, action, interms,
        and reward tensors in order.

        Args:
            inputs (tf.Tensor): The encoded (timestep, noise) input tensor.
            state (Sequence[tf.Tensor]): The current state tensors.
            scope (Optional[str]): The cell name scope.

        Returns:
            (CellOutput, CellState): A pair with the cell's output tuple and next state.
        '''

        # inputs
        timestep = tf.expand_dims(inputs[:, 0], -1)
        noise = inputs[:, 1:]

        # noise
        noise = utils.decode_inputs_as_noise_samples(noise, self.config['encoding'])
        noise = dict(noise)

        # action
        action = self.policy(state, timestep)

        # next state
        interm, next_state = self.compiler.cpfs(state, action, noise=noise)

        # reward
        reward = self.compiler.reward(state, action, next_state)

        # outputs
        next_state = utils.to_tensor(next_state)
        interm = utils.to_tensor(interm)
        output = OutputTuple(next_state, action, interm, reward)

        return (output, next_state)


class Simulator(object):

    def __init__(self,
            compiler: ReparameterizationCompiler,
            policy: OpenLoopPolicy,
            config: Dict) -> None:
        self.compiler = compiler
        self.policy = policy
        self.config = config

    @property
    def graph(self):
        return self.compiler.graph

    @property
    def batch_size(self):
        return self.policy.batch_size

    @property
    def horizon(self):
        return self.policy.horizon

    def build(self) -> None:
        '''Builds the recurrent cell ops by embedding the `policy` in the transition sampling.

        Args:
            policy (:obj:`OpenLoopPolicy`): A deep reactive policy.
        '''
        with self.graph.as_default():
            with tf.name_scope('reparameterization'):
                self.reparameterization_map = self.compiler.get_cpfs_reparameterization()
                self.samples = utils.get_noise_samples(self.reparameterization_map, self.batch_size, self.horizon)
                self.noise, encoding = utils.encode_noise_samples_as_inputs(self.samples)

        self.cell = SimulationCell(self.compiler, self.policy, config={'encoding': encoding})

    def trajectory(self, initial_state):
        with self.graph.as_default():

            with tf.name_scope('inputs'):
                self.timesteps = self.timesteps(self.batch_size, self.horizon)
                self.inputs = tf.concat([self.timesteps, self.noise], axis=2)

            with tf.name_scope('trajectory'):
                outputs, final_state = tf.nn.dynamic_rnn(self.cell,
                                                         self.inputs,
                                                         initial_state=initial_state,
                                                         dtype=tf.float32)

            with tf.name_scope('total_reward'):
                total_reward = tf.reduce_sum(tf.squeeze(outputs[3]), axis=1)

        states = tuple(fluent[0] for fluent in outputs[0])
        actions = tuple(fluent[0] for fluent in outputs[1])
        interms = tuple(fluent[0] for fluent in outputs[2])
        rewards = outputs[3]
        trajectory = Trajectory(states, actions, interms, rewards)

        return trajectory, final_state, total_reward

    def run(self, trajectory):
        with tf.Session(graph=self.graph) as sess:
            sess.run(tf.global_variables_initializer())

            noise_inputs = utils.evaluate_noise_samples_as_inputs(sess, self.samples)
            feed_dict = {
                self.noise: noise_inputs
            }
            return sess.run(trajectory, feed_dict=feed_dict)

    @classmethod
    def timesteps(cls, batch_size: int, horizon: int) -> tf.Tensor:
        with tf.name_scope('timesteps'):
            start, limit, delta = horizon - 1, -1, -1
            timesteps_range = tf.range(start, limit, delta, dtype=tf.float32)
            timesteps_range = tf.expand_dims(timesteps_range, -1)
            batch_timesteps = tf.stack([timesteps_range] * batch_size)
            return batch_timesteps
