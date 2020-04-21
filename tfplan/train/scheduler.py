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


class EpochScheduler:
    """EpochScheduler generates the number of training epochs
    for each timestep according to a linear schedule.

    Args:
        start (int): initial value.
        end (int): final value.
        delta (int): increment (>0) or decrement (<0).
    """

    def __init__(self, start, end, delta):
        self.start = start
        self.end = end
        self.delta = delta

    def __call__(self, timestep):
        epochs = self.start + timestep * self.delta
        clip = max if self.delta < 0 else min
        return clip(epochs, self.end)

    def __repr__(self):
        return f"EpochScheduler(start={self.start}, end={self.end}, delta={self.delta})"
