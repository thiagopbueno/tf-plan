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


from pyrddl.parser import RDDLParser
from rddl2tf.compiler import Compiler
from tfplan.train.policy import OpenLoopPolicy
from tfplan.train.optimizer import ActionOptimizer

import numpy as np
import tensorflow as tf
import unittest


class TestActionOptimizer(unittest.TestCase):

    @classmethod
    def setUpClass(cls):

        # initialize hyper-parameters
        cls.horizon = 40
        cls.batch_size = 64
        cls.epochs = 50
        cls.learning_rate = 0.01

        # parse RDDL file
        with open('rddl/deterministic/Navigation.rddl') as file:
            parser = RDDLParser()
            parser.build()
            rddl = parser.parse(file.read())
            rddl.build()

        # initializer RDDL2TensorFlow compiler
        cls.rddl2tf = Compiler(rddl, batch_mode=True)

        # initialize open-loop policy
        cls.policy = OpenLoopPolicy(cls.rddl2tf, cls.batch_size, cls.horizon)
        cls.policy.build('test')

        # initialize ActionOptimizer
        cls.optimizer = ActionOptimizer(cls.rddl2tf, cls.policy)
        cls.optimizer.build(cls.learning_rate, cls.batch_size, cls.horizon)

    def test_state_trajectory(self):
        states = self.optimizer.states
        state_size = self.rddl2tf.rddl.state_size
        self.assertIsInstance(states, tuple, 'state trajectory is factored')
        self.assertEqual(len(states), len(state_size), 'state trajectory has all states fluents')
        for fluent, fluent_size in zip(states, state_size):
            tensor_size = [self.batch_size, self.horizon] + list(fluent_size)
            self.assertIsInstance(fluent, tf.Tensor, 'state fluent is a tensor')
            self.assertListEqual(fluent.shape.as_list(), tensor_size, 'fluent size is [batch_size, horizon, state_fluent_size]')

    def test_action_trajectory(self):
        actions = self.optimizer.actions
        action_size = self.rddl2tf.rddl.action_size
        self.assertIsInstance(actions, tuple, 'action trajectory is factored')
        self.assertEqual(len(actions), len(action_size),
            'action trajectory has all actions fluents')
        for fluent, fluent_size in zip(actions, action_size):
            tensor_size = [self.batch_size, self.horizon] + list(fluent_size)
            self.assertIsInstance(fluent, tf.Tensor, 'action fluent is a tensor')
            self.assertListEqual(fluent.shape.as_list(), tensor_size,
                'fluent size is [batch_size, horizon, action_fluent_size]')

    def test_rewards_trajectory(self):
        rewards = self.optimizer.rewards
        rewards_shape = [self.batch_size, self.horizon, 1]
        self.assertIsInstance(rewards, tf.Tensor, 'reward trajectory is a tensor')
        self.assertListEqual(rewards.shape.as_list(), rewards_shape ,
            'reward shape is [batch_size, horizon, 1]')

    def test_optimization_variables(self):
        action_size = self.rddl2tf.rddl.action_size
        with self.rddl2tf.graph.as_default():
            policy_variables = tf.trainable_variables()
            self.assertEqual(len(policy_variables), len(action_size),
                'one variable per action fluent per timestep')

    def test_total_reward(self):
        total_reward = self.optimizer.total_reward
        self.assertIsInstance(total_reward, tf.Tensor, 'total reward is a tensor')
        self.assertEqual(total_reward.dtype, tf.float32, 'total reward is a real tensor')
        self.assertListEqual(total_reward.shape.as_list(), [self.batch_size],
            'total reward has a scalar value for each trajectory')

    def test_avg_total_reward(self):
        avg_total_reward = self.optimizer.avg_total_reward
        self.assertIsInstance(avg_total_reward, tf.Tensor, 'average total reward is a tensor')
        self.assertEqual(avg_total_reward.dtype, tf.float32, 'average total reward is a real tensor')
        self.assertListEqual(avg_total_reward.shape.as_list(), [], 'average total reward is a scalar')

    def test_optimization_objective(self):
        loss = self.optimizer.loss
        self.assertIsInstance(loss, tf.Tensor, 'loss function is a tensor')
        self.assertEqual(loss.dtype, tf.float32, 'loss function is a real tensor')
        self.assertListEqual(loss.shape.as_list(), [], 'loss function is a scalar')

    def test_loss_optimizer(self):
        optimizer = self.optimizer._optimizer
        self.assertIsInstance(optimizer, tf.train.RMSPropOptimizer)
        train_op = self.optimizer._train_op
        self.assertEqual(train_op.name, 'action_optimizer/RMSProp')

    def test_optimizer_solution(self):
        action_size = self.rddl2tf.rddl.action_size
        solution, variables = self.optimizer.run(self.epochs, show_progress=False)
        self.assertIsInstance(solution, tuple)
        self.assertEqual(len(solution), len(action_size))
        for param, fluent_size in zip(solution, action_size):
            self.assertIsInstance(param, np.ndarray, 'solution is factored n-dimensional array')
            self.assertListEqual(list(param.shape), [self.horizon] + list(fluent_size))
