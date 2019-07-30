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
from typing import List, Optional, Sequence, Tuple, Union

from rddl2tf.core.fluent import TensorFluent


Shape = Sequence[int]
ShapeAsList = List[int]

Noise = Tuple[tf.distributions.Distribution, ShapeAsList]
NoiseList = List[Noise]
NoiseMap = List[Tuple[str, NoiseList]]

Samples = List[Tuple[str, List[tf.Tensor]]]
SamplesEncoding = Sequence[Tuple[str, Sequence[Tuple[int, int, Shape]]]]


def cell_size(sizes: Sequence[Shape]) -> Sequence[Union[Shape, int]]:
    return tuple(sz if sz != () else (1,) for sz in sizes)


def to_tensor(fluents: Sequence[TensorFluent]) -> Sequence[tf.Tensor]:
    return tuple(f.tensor for f in fluents)


def get_noise_samples(
        reparameterization_map: NoiseMap,
        batch_size: int,
        horizon: int) -> Samples:
    samples = []

    for name, noise_lst in reparameterization_map:
        cpf_scope = name.replace("'", '').replace('/', '_')

        sample_lst = []

        with tf.name_scope(cpf_scope):
            for dist, shape in noise_lst:
                sample_shape = [batch_size, horizon] + shape
                xi = dist.sample(sample_shape)
                sample_lst.append(xi)

        samples.append((name, sample_lst))

    return samples


def encode_noise_samples_as_inputs(
        samples: Samples) -> Tuple[tf.Tensor, SamplesEncoding]:
    xi_placeholders = []
    encoding = []

    i = 0
    for name, xi_lst in samples:
        if not xi_lst:
            continue

        slices = []

        for xi in xi_lst:
            batch_size = xi.shape[0]
            horizon = xi.shape[1]
            xi_shape = xi.shape.as_list()[2:]
            xi_size =  np.prod(xi_shape)

            xi = tf.placeholder(xi.dtype, shape=(batch_size, horizon, xi_size))
            xi_placeholders.append(xi)

            slices.append((i, i+xi_size-1, xi_shape))
            i += xi_size

        encoding.append((name, slices))

    inputs = tf.concat(xi_placeholders, axis=2)

    return (inputs, encoding)


def decode_inputs_as_noise_samples(
        inputs: tf.Tensor,
        encoding: SamplesEncoding) -> Noise:
    samples = []

    for name, slices in encoding:
        xi_lst = []

        for start, end, shape in slices:
            xi = inputs[:, start:end+1]
            xi = tf.reshape(xi, [-1, *shape])
            xi_lst.append(xi)

        samples.append((name, xi_lst))

    return samples


def evaluate_noise_samples_as_inputs(
        sess: tf.Session,
        samples: Samples) -> Tuple[tf.Tensor, SamplesEncoding]:
    xi_values = []

    for name, xi_lst in samples:
        if not xi_lst:
            continue

        for xi in xi_lst:
            xi = sess.run(xi)
            xi_values.append(xi)

    noise_inputs = np.concatenate(xi_values, axis=2)
    return noise_inputs
