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

from tfplan.planners.deterministic import utils


OutputTuple = namedtuple("OutputTuple", "state action interm reward")
Trajectory = namedtuple("Trajectory", "states actions interms rewards")


class SimulationCell(tf.compat.v1.nn.rnn_cell.RNNCell):
    """SimulationCell class implements an RNN cell that simulates the
    next state and reward for the MDP transition given by the RDDL model.

    Args:
        compiler (rddl2tf.compilers.DefaulCompiler): The RDDL2TF compiler.
        policy (tfplan.train.OpenLoopPolicy): The state-independent policy (e.g., a plan).
    """

    def __init__(self, compiler, policy):
        self.compiler = compiler
        self.policy = policy

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

        # timestep
        timestep = inputs

        # action
        action = self.policy(state, timestep)

        # next state
        interm, next_state = self.compiler.cpfs(state, action)

        # reward
        reward = self.compiler.reward(state, action, next_state)

        # outputs
        next_state = utils.to_tensor(next_state)
        interm = utils.to_tensor(interm)
        output = OutputTuple(next_state, action, interm, reward)

        return (output, next_state)


class Simulator:
    """Simulator class implements an RNN-based trajctory simulator
    for the RDDL model.

    Args:
        compiler (rddl2tf.compilers.DefaulCompiler): The RDDL2TF compiler.
        policy (tfplan.train.OpenLoopPolicy): The state-independent policy (e.g., a plan).
    """

    def __init__(self, compiler, policy):
        self.compiler = compiler
        self.policy = policy

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
        """Builds the recurrent cell ops by embedding the
        policy in the transition sampling cell.
        """
        self.cell = SimulationCell(self.compiler, self.policy)

    def trajectory(self, initial_state):
        """Returns the state-action-reward trajectory induced by
        the given `initial_state` and policy.

        Args:
            initial_state (Sequence[tf.Tensor]): The trajectory's initial state.

        Returns:
            trajectory (Trajectory): The collection of states-actions-interms-rewards trajectory.
            final_state (Sequence[tf.Tensor]): The trajectory's final state.
            total_reward (tf.Tensor(shape=(batch_size,))): The trajectory's total reward.
        """
        with self.graph.as_default():

            with tf.name_scope("inputs"):
                self.inputs = self.timesteps(self.batch_size, self.horizon)

            with tf.name_scope("trajectory"):
                outputs, final_state = tf.nn.dynamic_rnn(
                    self.cell,
                    self.inputs,
                    initial_state=initial_state,
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
            return sess.run(trajectory)

    @classmethod
    def timesteps(cls, batch_size, horizon):
        """Returns the batch-sized decreasing-horizon timesteps tensor."""
        with tf.name_scope("timesteps"):
            start, limit, delta = horizon - 1, -1, -1
            timesteps_range = tf.range(start, limit, delta, dtype=tf.float32)
            timesteps_range = tf.expand_dims(timesteps_range, -1)
            batch_timesteps = tf.stack([timesteps_range] * batch_size)
            return batch_timesteps
