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
from tfplan.test.evaluator import ActionEvaluator

import numpy as np
import tensorflow as tf

import unittest


class TestActionEvaluator(unittest.TestCase):

    @classmethod
    def setUpClass(cls):

        # initialize hyper-parameters
        cls.horizon = 40
        cls.batch_size = 1

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

        # sample policy variables to initialize open-loop policy
        cls.policy_variables = []
        for shape in cls.rddl2tf.rddl.action_size:
            size = [cls.horizon] + list(shape)
            cls.policy_variables.append(np.random.uniform(low=-1.0, high=1.0, size=size))

        # initialize action evaluator
        cls.evaluator = ActionEvaluator(cls.rddl2tf, cls.policy)

    def test_run(self):
        trajectories = self.evaluator.run()
        self.assertIsInstance(trajectories, tuple)
        self.assertEqual(len(trajectories), 6)
