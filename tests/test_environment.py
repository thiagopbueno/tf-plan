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


import rddlgym

from tfplan.planners.environment import OnlinePlanning
from tfplan.planners.online import OnlineOpenLoopPlanner

import numpy as np
import tensorflow as tf

import unittest


class TestOnlinePlanning(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.batch_size = 16
        cls.horizon = 3
        cls.learning_rate = 0.001
        cls.epochs = 5

    def setUp(self):
        model_ids = ['Navigation-v2', 'HVAC-3', 'Reservoir-8']
        self.domains = { model_id: self._init_domain(model_id) for model_id in model_ids }

    def _init_domain(self, model_id):
        compiler = rddlgym.make(model_id, mode=rddlgym.SCG)
        compiler.batch_mode_on()

        initial_state = compiler.compile_initial_state(batch_size=1)
        default_action = compiler.compile_default_action(batch_size=1)

        planner = OnlineOpenLoopPlanner(compiler, self.batch_size, self.horizon)
        planner.build(self.learning_rate, epochs=self.epochs, show_progress=False)

        online_planner = OnlinePlanning(compiler, planner)
        online_planner.build()

        return {
            'initial_state': initial_state,
            'default_action': default_action,
            'online_planner': online_planner
        }

    def test_execution_graph(self):

        for model_id, domain in self.domains.items():

            online_planner = domain['online_planner']
            default_action = domain['default_action']
            initial_state = domain['initial_state']

            action = online_planner.action
            self.assertIsInstance(action, tuple)
            self.assertEqual(len(action), len(default_action))
            self.assertTrue(all(isinstance(fluent, tf.Tensor) for fluent in action))
            for action_fluent, default_action_fluent in zip(action, default_action):
                self.assertEqual(action_fluent.shape, default_action_fluent.shape)
                self.assertEqual(action_fluent.dtype, default_action_fluent.dtype)

            state = online_planner.state
            self.assertIsInstance(state, tuple)
            self.assertEqual(len(state), len(initial_state))
            self.assertTrue(all(isinstance(fluent, tf.Tensor) for fluent in state))
            for state_fluent, initial_state_fluent in zip(state, initial_state):
                self.assertEqual(state_fluent.shape, initial_state_fluent.shape)
                self.assertEqual(state_fluent.dtype, initial_state_fluent.dtype)

            next_state = online_planner.next_state
            self.assertIsInstance(next_state, tuple)
            self.assertEqual(len(next_state), len(initial_state))
            self.assertTrue(all(isinstance(fluent, tf.Tensor) for fluent in next_state))
            for state_fluent, initial_state_fluent in zip(next_state, initial_state):
                self.assertEqual(state_fluent.shape, initial_state_fluent.shape)
                self.assertEqual(state_fluent.dtype, initial_state_fluent.dtype)

            reward = online_planner.reward
            self.assertIsInstance(reward, tf.Tensor)
            self.assertListEqual(reward.shape.as_list(), [1, 1])
            self.assertEqual(reward.dtype, tf.float32)

    def test_online_planning_cycle(self):
        for model_id, domain in self.domains.items():
            online_planner = domain['online_planner']
            trajectories, _ = online_planner.run(self.horizon, show_progress=False)
            self.assertIsInstance(trajectories, tuple)
            self.assertEqual(len(trajectories), 6)
