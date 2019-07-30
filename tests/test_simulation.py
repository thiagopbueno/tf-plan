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
import unittest

import rddlgym
from rddl2tf import ReparameterizationCompiler

from tfplan.train.policy import OpenLoopPolicy
from tfplan.planners.stochastic.simulation import SimulationCell, Simulator, OutputTuple, Trajectory
from tfplan.planners.stochastic import utils


class TestSimulationCell(unittest.TestCase):

    def setUp(self):
        self.batch_size = 64
        self.horizon = 16

        self.rddl = rddlgym.make('Navigation-v2', mode=rddlgym.AST)

        self.compiler = ReparameterizationCompiler(self.rddl, self.batch_size)
        self.compiler.init()
        self.initial_state = self.compiler.initial_state()
        self.default_action = self.compiler.default_action()
        self.reparameterization_map = self.compiler.get_cpfs_reparameterization()

        self.policy = OpenLoopPolicy(self.compiler, self.horizon)
        self.policy.build('planning')

        with self.compiler.graph.as_default():
            self.samples = utils.get_noise_samples(
                self.reparameterization_map, self.batch_size, self.horizon)
            self.noise, self.encoding = utils.encode_noise_samples_as_inputs(self.samples)
            self.timesteps = Simulator.timesteps(self.batch_size, self.horizon)
            self.inputs = tf.concat([self.timesteps[:,0,...], self.noise[:,0,...]], axis=1)

        self.cell = SimulationCell(self.compiler, self.policy, config={'encoding': self.encoding})

    def test_call(self):
        output, next_state = self.cell(self.inputs, self.initial_state)

        self.assertIsInstance(output, OutputTuple)
        self.assertEqual(len(output), 4)

        self.assertEqual(output.state, output[0])
        self.assertEqual(output.action, output[1])
        self.assertEqual(output.interm, output[2])
        self.assertEqual(output.reward, output[3])

        self.assertEqual(output.state, next_state)

        for action_fluent, default_action_fluent in zip(output.action, self.default_action):
            self.assertEqual(action_fluent.shape, default_action_fluent.shape)

        self.assertListEqual(output.reward.shape.as_list(), [self.batch_size, 1])

        for fluent, next_fluent in zip(self.initial_state, next_state):
            self.assertEqual(fluent.shape, next_fluent.shape)


class TestSimulator(unittest.TestCase):

    def setUp(self):
        self.batch_size = 64
        self.horizon = 16

        self.rddl = rddlgym.make('Navigation-v2', mode=rddlgym.AST)

        self.compiler = ReparameterizationCompiler(self.rddl, self.batch_size)
        self.compiler.init()
        self.initial_state = self.compiler.initial_state()
        self.default_action = self.compiler.default_action()

        self.policy = OpenLoopPolicy(self.compiler, self.horizon)
        self.policy.build('planning')

        self.simulator = Simulator(self.compiler, self.policy, config=None)
        self.simulator.build()

        self.trajectory, self.final_state, self.total_reward = self.simulator.trajectory(self.initial_state)

    def test_build(self):
        self.assertIsInstance(self.simulator.cell, SimulationCell)
        self.assertIsNotNone(self.simulator.cell.config['encoding'])

    def test_trajectory(self):
        self.assertIsInstance(self.trajectory, Trajectory)

        for tensor, fluent in zip(self.trajectory.states, self.initial_state):
            self.assertEqual(int(tensor.shape[0]), self.batch_size)
            self.assertEqual(int(tensor.shape[1]), self.horizon)
            self.assertListEqual(tensor.shape.as_list()[2:], fluent.shape.as_list()[1:])

        for tensor, fluent in zip(self.trajectory.actions, self.default_action):
            self.assertEqual(int(tensor.shape[0]), self.batch_size)
            self.assertEqual(int(tensor.shape[1]), self.horizon)
            self.assertListEqual(tensor.shape.as_list()[2:], fluent.shape.as_list()[1:])

        self.assertIsInstance(self.final_state, tuple)
        self.assertEqual(len(self.final_state), len(self.initial_state))

        self.assertIsInstance(self.total_reward, tf.Tensor)
        self.assertEqual(self.total_reward.dtype, tf.float32)
        self.assertListEqual(self.total_reward.shape.as_list(), [self.batch_size])

    def test_run(self):
        trajectory_ = self.simulator.run(self.trajectory)
        self.assertEqual(len(trajectory_), len(self.trajectory))

        for tensors, values  in zip(self.trajectory[:-1], trajectory_[:-1]):

            for tensor, value in zip(tensors, values):
                self.assertIsInstance(value, np.ndarray)
                self.assertListEqual(list(value.shape), tensor.shape.as_list())

        self.assertIsInstance(trajectory_[-1], np.ndarray)
        self.assertListEqual(list(trajectory_[-1].shape), self.trajectory.rewards.shape.as_list())
