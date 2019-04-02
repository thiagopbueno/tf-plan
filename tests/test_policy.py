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

import tensorflow as tf
import unittest


class TestOpenLoopPolicy(unittest.TestCase):

    @classmethod
    def setUpClass(cls):

        # initialize hyper-parameters
        cls.horizon = 40
        cls.batch_size = 64

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

        # execute policy for the given horizon and initial state
        with cls.rddl2tf.graph.as_default():
            cls.state = cls.rddl2tf.compile_initial_state(cls.batch_size)
            cls.actions = []
            for t in range(cls.horizon-1, -1, -1):
                timestep = tf.constant(t, dtype=tf.float32, shape=(cls.batch_size, 1))
                action = cls.policy(cls.state, timestep)
                cls.actions.append(action)

    def test_policy_variables(self):
        action_fluents = self.rddl2tf.rddl.domain.action_fluent_ordering
        action_size = self.rddl2tf.rddl.action_size

        with self.rddl2tf.graph.as_default():
            policy_variables = tf.trainable_variables()
            name2variable = { var.name: var for var in policy_variables }

            self.assertEqual(len(policy_variables), len(action_fluents),
                'one variable per action fluent')

            for fluent, size, var in zip(action_fluents, action_size, policy_variables):
                var_name = 'test/' + fluent.replace('/', '-') + ':0'
                self.assertIn(var_name, name2variable, 'variable has fluent name')

                self.assertIsInstance(var, tf.Variable,
                    'policy variable is a variable tensor')

                shape = [self.batch_size, self.horizon] + list(size)
                self.assertListEqual(var.shape.as_list(), shape,
                    'policy variable has shape (batch_size, horizon, action_fluent_shape')

    def test_policy_actions(self):
        action_fluents = self.rddl2tf.rddl.domain.action_fluent_ordering
        action_size = self.rddl2tf.rddl.action_size

        for action in self.actions:
            self.assertIsInstance(action, tuple, 'action is factored')
            self.assertEqual(len(action), len(action_fluents),
                'one action tensor per action fluent in each timestep')

            for fluent, size, tensor in zip(action_fluents, action_size, action):
                self.assertIsInstance(tensor, tf.Tensor, 'action fluent is a tf.Tensor')

                shape = [self.batch_size] + list(size)
                self.assertListEqual(tensor.shape.as_list(), shape,
                    'action tensor has shape (batch_size, fluent_shape)')
