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

# pylint: disable=missing-docstring,too-many-instance-attributes


import unittest

import numpy as np
import tensorflow as tf

import rddlgym
from rddl2tf.compilers.modes.reparameterization import ReparameterizationCompiler

from tfplan.planners.stochastic import utils


class TestPlannersStochasticUtils(unittest.TestCase):
    def setUp(self):
        self.batch_size = 64
        self.horizon = 16

        self.rddl = rddlgym.make("Navigation-v2", mode=rddlgym.AST)

        self.compiler = ReparameterizationCompiler(self.rddl, self.batch_size)
        self.compiler.init()

        with self.compiler.graph.as_default():
            self.reparameterization_map = self.compiler.get_cpfs_reparameterization()

            self.samples = utils.get_noise_samples(
                self.reparameterization_map, self.batch_size, self.horizon
            )

            self.inputs, self.encoding = utils.encode_noise_samples_as_inputs(
                self.samples
            )

            self.decoded_samples = utils.decode_inputs_as_noise_samples(
                self.inputs[:, 0, ...], self.encoding
            )

    def test_get_noise_samples(self):
        self.assertEqual(len(self.reparameterization_map), len(self.samples))

        for noise, sample in zip(self.reparameterization_map, self.samples):
            self.assertEqual(noise[0], sample[0])
            self.assertEqual(len(noise[1]), len(sample[1]))

            for (_, shape), xi_noise in zip(noise[1], sample[1]):
                self.assertIsInstance(xi_noise, tf.Tensor)
                self.assertListEqual(
                    list(xi_noise.shape), [self.batch_size, self.horizon] + shape
                )

    def test_encode_noise_samples_as_inputs(self):
        samples_dict = dict(self.samples)
        encoding_dict = dict(self.encoding)

        for name, samples_lst in self.samples:
            if samples_lst:
                self.assertIn(name, encoding_dict)
            else:
                self.assertNotIn(name, encoding_dict)

        i = 0
        for name, slices in self.encoding:
            samples_lst = samples_dict[name]
            self.assertEqual(len(slices), len(samples_lst))

            for (start, end, slice_shape), sample in zip(slices, samples_lst):
                sample_shape = sample.shape.as_list()
                self.assertListEqual(slice_shape, sample_shape[2:])
                self.assertEqual(start, i)
                self.assertEqual(end, i + (end - start))
                self.assertEqual(end, np.prod(slice_shape) - 1)

                i += end - start + 1

        self.assertIsInstance(self.inputs, tf.Tensor)
        self.assertListEqual(
            self.inputs.shape.as_list(), [self.batch_size, self.horizon, i]
        )

    def test_decode_inputs_as_noise_samples(self):
        decoded_samples_dict = dict(self.decoded_samples)
        for name, samples_lst in self.samples:
            if samples_lst:
                self.assertIn(name, decoded_samples_dict)
            else:
                self.assertNotIn(name, decoded_samples_dict)

        samples_dict = dict(self.samples)
        for name, decoded_sample_lst in decoded_samples_dict.items():
            sample_lst = samples_dict[name]
            self.assertEqual(len(decoded_sample_lst), len(sample_lst))

            for decoded_sample, sample in zip(decoded_sample_lst, sample_lst):
                sample = sample[:, 0, ...]
                self.assertEqual(decoded_sample.dtype, sample.dtype)
                self.assertListEqual(
                    decoded_sample.shape.as_list(), sample.shape.as_list()
                )

    def test_evaluate_noise_samples_as_inputs(self):
        with tf.Session(graph=self.compiler.graph) as sess:
            noise_inputs = utils.evaluate_noise_samples_as_inputs(sess, self.samples)
            self.assertIsInstance(noise_inputs, np.ndarray)
            self.assertListEqual(list(noise_inputs.shape), self.inputs.shape.as_list())
