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


import tensorflow as tf

from pyrddl.parser import RDDLParser
from tfrddlsim.compiler import Compiler

from tfplan.train.policy import OpenLoopPolicy

import unittest


class TestOpenLoopPolicy(unittest.TestCase):

    @classmethod
    def setUpClass(cls):

        # initialize hyper-parameters
        cls.horizon = 3
        cls.batch_size = 64

        # parse RDDL file
        with open('rddl/Navigation.rddl') as file:
            parser = RDDLParser()
            parser.build()
            rddl = parser.parse(file.read())

        # initializer RDDL2TensorFlow compiler
        cls.rddl2tf = Compiler(rddl, batch_mode=True)

        # initialize open-loop policy
        cls.policy = OpenLoopPolicy(cls.rddl2tf, cls.batch_size)

        # execute policy for the given horizon and initial state
        with cls.rddl2tf.graph.as_default():
            cls.state = cls.rddl2tf.compile_initial_state(cls.batch_size)
            cls.actions = []
            for t in range(cls.horizon, 0, -1):
                timestep = tf.constant(t, dtype=tf.float32, shape=(cls.batch_size, 1))
                with tf.variable_scope('timestep{}'.format(t)):
                    action = cls.policy(cls.state, timestep)
                    cls.actions.append(action)

    def test_policy_variables(self):
        action_fluents = self.rddl2tf.action_fluent_ordering
        action_size = self.rddl2tf.action_size

        with self.rddl2tf.graph.as_default():
            policy_variables = tf.trainable_variables()
            name2variable = { var.name: var for var in policy_variables }

            self.assertEqual(len(policy_variables), self.horizon * len(action_fluents),
                'one variable per action fluent per timestep')

            for i, t in enumerate(range(self.horizon, 0, -1)):
                scope = 'timestep{}'.format(t)
                policy_variables = tf.trainable_variables(scope=scope)
                action = self.actions[i]

                self.assertEqual(len(policy_variables), len(action_fluents),
                    'one variable per action fluent in each timestep')

                self.assertEqual(len(action), len(action_fluents),
                    'one action tensor per action fluent in each timestep')

                for fluent, shape, tensor in zip(action_fluents, action_size, action):
                    self.assertIsInstance(tensor, tf.Tensor, 'action fluent is a tf.Tensor')

                    var_name = scope + '/' + fluent.replace('/', '-') + ':0'
                    self.assertIn(var_name, name2variable,
                        'variable name is the concatenation of timestep scope and fluent name')

                    var = name2variable[var_name]
                    shape = [self.batch_size] + list(shape)
                    self.assertListEqual(var.shape.as_list(), shape,
                        'policy variable has shape (batch_size, fluent_shape)')
                    self.assertListEqual(tensor.shape.as_list(), shape,
                        'action tensor has shape (batch_size, fluent_shape)')

