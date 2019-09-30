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


"""Collection of RNN-based simulation utility functions."""

# pylint: disable=missing-docstring


def cell_size(sizes):
    return tuple(sz if sz != () else (1,) for sz in sizes)


def to_tensor(fluents):
    return tuple(f.tensor for f in fluents)
