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

# pylint: disable=missing-docstring


def make(planner, rddl, config):
    """tf-plan planner factory."""
    from tfplan.planners import Tensorplan, StraightLinePlanner, HindsightPlanner

    planner_cls = {
        "tensorplan": Tensorplan,
        "straightline": StraightLinePlanner,
        "hindsight": HindsightPlanner,
    }

    return planner_cls[planner](rddl, config)
