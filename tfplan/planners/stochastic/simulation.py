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


from collections import namedtuple
import tensorflow as tf

from tfplan.planners.stochastic import utils


Trajectory = namedtuple("Trajectory", "states actions interms rewards")


class SimulationCell(tf.compat.v1.nn.rnn_cell.RNNCell):
    """
    SimulationCell class implements an RNN cell that simulates the
    next state and reward for the MDP transition given by the RDDL model.

    Args:
        compiler (rddl2tf.compilers.ReparameterizationCompiler): The RDDL2TF compiler.
        policy (tfplan.train.OpenLoopPolicy): The state-independent policy (e.g., a plan).
        config (Dict[str, Any]): A config dict.
    """

    def __init__(self, compiler, policy, config=None):
        self.compiler = compiler
        self.policy = policy
        self.config = config

    @property
    def state_size(self):
        """Returns the MDP state size."""
        return utils.cell_size(self.compiler.rddl.state_size)

    @property
    def action_size(self):
        """Returns the MDP action size."""
        return utils.cell_size(self.compiler.rddl.action_size)

    @property
    def interm_size(self):
        """Returns the MDP intermediate state size."""
        return utils.cell_size(self.compiler.rddl.interm_size)

    @property
    def output_size(self):
        """Returns the simulation cell output size."""
        return (self.state_size, self.action_size, self.interm_size, 1)

    def __call__(self, inputs, state, scope=None):
        """Returns the cell's output tuple and next state tensors.

        Output tuple packs together the next state, action, interms,
        and reward tensors in order.

        Args:
            inputs (tf.Tensor): The encoded (timestep, noise) input tensor.
            state (Sequence[tf.Tensor]): The current state tensors.
            scope (Optional[str]): The cell name scope.

        Returns:
            (CellOutput, CellState): A pair with the cell's output tuple and next state.
        """

        # inputs
        timestep = tf.expand_dims(inputs[:, 0], -1)
        noise = inputs[:, 1:]

        # noise
        noise = utils.decode_inputs_as_noise_samples(noise, self.config["encoding"])
        noise = dict(noise)

        # action
        action = self.policy(state, timestep)

        # next state
        interm, next_state = self.compiler.cpfs(state, action, noise=noise)

        # reward
        reward = self.compiler.reward(state, action, next_state)

        # outputs
        next_state = utils.to_tensor(next_state)
        action = tuple((tensor,) for tensor in action)
        interm = utils.to_tensor(interm)
        output = next_state, action, interm, reward

        next_state = tuple(tensor[0] for tensor in next_state)

        return (output, next_state)


class Simulator:
    """
    Simulator class implements an RNN-based trajctory simulator
    for the RDDL model.

    Args:
        compiler (rddl2tf.compilers.DefaulCompiler): The RDDL2TF compiler.
        policy (tfplan.train.OpenLoopPolicy): The state-independent policy (e.g., a plan).
        config (Dict[str, Any]): A config dict.
    """

    # pylint: disable=too-many-instance-attributes

    def __init__(self, compiler, policy, config):
        self.compiler = compiler
        self.policy = policy
        self.config = config

        self.reparameterization_map = None
        self.samples = None
        self.noise = None
        self.cell = None
        self.inputs = None

    @property
    def graph(self):
        """Returns the compiler's graph."""
        return self.compiler.graph

    @property
    def batch_size(self):
        """Returns the policy's batch size."""
        return self.policy.batch_size

    @property
    def horizon(self):
        """Returns the policy's batch size."""
        return self.policy.horizon

    def build(self):
        """Builds the reparametrized recurrent cell."""
        with self.graph.as_default():
            with tf.name_scope("reparameterization"):
                self.reparameterization_map = (
                    self.compiler.get_cpfs_reparameterization()
                )
                self.samples = utils.get_noise_samples(
                    self.reparameterization_map, self.batch_size, self.horizon
                )
                self.noise, encoding = utils.encode_noise_samples_as_inputs(
                    self.samples
                )

        self.cell = SimulationCell(
            self.compiler, self.policy, config={"encoding": encoding}
        )

    def trajectory(self, initial_state, sequence_length=None):
        """Returns the state-action-reward trajectory induced by
        the given `initial_state` and policy.

        Args:
            initial_state (Sequence[tf.Tensor]): The trajectory's initial state.
            sequence_length (tf.Tensor(shape=(batch_size,))): An integer vector
            defining the trajectories' number of timesteps.

        Returns:
            trajectory: The collection of states-actions-interms-rewards trajectory.
            final_state (Sequence[tf.Tensor]): The trajectory's final state.
            total_reward (tf.Tensor(shape=(batch_size,))): The trajectory's total reward.
        """
        with self.graph.as_default():

            with tf.name_scope("inputs"):
                timesteps = Simulator.timesteps(self.batch_size, self.horizon)
                self.inputs = tf.concat([timesteps, self.noise], axis=2)

            with tf.name_scope("trajectory"):
                outputs, final_state = tf.nn.dynamic_rnn(
                    self.cell,
                    self.inputs,
                    initial_state=initial_state,
                    sequence_length=sequence_length,
                    dtype=tf.float32,
                )

            with tf.name_scope("total_reward"):
                total_reward = tf.reduce_sum(tf.squeeze(outputs[3]), axis=1)

        states = tuple(fluent[0] for fluent in outputs[0])
        actions = tuple(fluent[0] for fluent in outputs[1])
        interms = tuple(fluent[0] for fluent in outputs[2])
        rewards = outputs[3]
        trajectory = Trajectory(states, actions, interms, rewards)

        return trajectory, final_state, total_reward

    def run(self, trajectory):
        """Evaluates the given `trajectory`."""
        with tf.Session(graph=self.graph) as sess:
            sess.run(tf.global_variables_initializer())

            noise_inputs = utils.evaluate_noise_samples_as_inputs(sess, self.samples)
            feed_dict = {self.noise: noise_inputs}
            return sess.run(trajectory, feed_dict=feed_dict)

    @classmethod
    def timesteps(cls, batch_size, horizon):
        """Returns the batch-sized increasing-horizon timesteps tensor."""
        with tf.name_scope("timesteps"):
            start, limit = 0, horizon
            timesteps_range = tf.range(start, limit, dtype=tf.float32)
            timesteps_range = tf.expand_dims(timesteps_range, -1)
            batch_timesteps = tf.stack([timesteps_range] * batch_size)
            return batch_timesteps
