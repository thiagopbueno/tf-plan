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

# pylint: disable=missing-docstring,redefined-outer-name,invalid-name


from collections import namedtuple
import pytest
import numpy as np
import tensorflow as tf

import rddlgym
from rddl2tf.compilers.modes.reparameterization import ReparameterizationCompiler

from tfplan.planners.stochastic import utils


BATCH_SIZE = 32
HORIZON = 20


Reparameterization = namedtuple(
    "Reparameterization", "compiler mapping samples inputs encoding decoded_samples"
)


@pytest.fixture(scope="module", params=["Navigation-v3", "Reservoir-8", "HVAC-3"])
def reparameterization(request):
    rddl = request.param
    model = rddlgym.make(rddl, mode=rddlgym.AST)
    compiler = ReparameterizationCompiler(model, batch_size=BATCH_SIZE)
    compiler.init()

    with compiler.graph.as_default():
        mapping = compiler.get_cpfs_reparameterization()
        samples = utils.get_noise_samples(mapping, BATCH_SIZE, HORIZON)

        inputs, encoding = utils.encode_noise_samples_as_inputs(samples)

        decoded_samples = utils.decode_inputs_as_noise_samples(
            inputs[:, 0, ...], encoding
        )

    return Reparameterization(
        compiler, mapping, samples, inputs, encoding, decoded_samples
    )


def test_get_noise_samples(reparameterization):
    reparameterization_map = reparameterization.mapping
    samples = reparameterization.samples

    assert len(reparameterization_map) == len(samples)

    for noise, sample in zip(reparameterization_map, samples):
        assert noise[0] == sample[0]
        assert len(noise[1]) == len(sample[1])

        for (_, shape), xi_noise in zip(noise[1], sample[1]):
            assert isinstance(xi_noise, tf.Tensor)
            if shape == []:
                shape = xi_noise.shape.as_list()[-1:]
            assert list(xi_noise.shape) == [BATCH_SIZE, HORIZON] + shape


def test_encode_noise_samples_as_inputs(reparameterization):
    samples = reparameterization.samples
    encoding = reparameterization.encoding
    inputs = reparameterization.inputs

    samples_dict = dict(samples)
    encoding_dict = dict(encoding)

    for name, samples_lst in samples:
        if samples_lst:
            assert name in encoding_dict
        else:
            assert name not in encoding_dict

    i = 0
    for name, slices in encoding:
        samples_lst = samples_dict[name]
        assert len(slices) == len(samples_lst)

        for (start, end, slice_shape), sample in zip(slices, samples_lst):
            sample_shape = sample.shape.as_list()
            assert slice_shape == sample_shape[2:]
            assert start == i
            assert end == i + (end - start)
            assert end == i + np.prod(slice_shape) - 1

            i += end - start + 1

    assert isinstance(inputs, tf.Tensor)
    assert inputs.shape.as_list() == [BATCH_SIZE, HORIZON, i]


def test_decode_inputs_as_noise_samples(reparameterization):
    samples = reparameterization.samples
    decoded_samples = reparameterization.decoded_samples

    decoded_samples_dict = dict(decoded_samples)
    for name, samples_lst in samples:
        if samples_lst:
            assert name in decoded_samples_dict
        else:
            assert name not in decoded_samples_dict

    samples_dict = dict(samples)
    for name, decoded_sample_lst in decoded_samples_dict.items():
        sample_lst = samples_dict[name]
        assert len(decoded_sample_lst) == len(sample_lst)

        for decoded_sample, sample in zip(decoded_sample_lst, sample_lst):
            sample = sample[:, 0, ...]
            assert decoded_sample.dtype == sample.dtype
            assert decoded_sample.shape.as_list() == sample.shape.as_list()


def test_evaluate_noise_samples_as_inputs(reparameterization):
    compiler = reparameterization.compiler
    samples = reparameterization.samples
    inputs = reparameterization.inputs

    with tf.Session(graph=compiler.graph) as sess:
        noise_inputs = utils.evaluate_noise_samples_as_inputs(sess, samples)
        assert isinstance(noise_inputs, np.ndarray)
        assert list(noise_inputs.shape) == inputs.shape.as_list()
