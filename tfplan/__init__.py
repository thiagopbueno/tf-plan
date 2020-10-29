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


def run(config):
    # pylint: disable=import-outside-toplevel

    import os

    import psutil
    import rddlgym
    import tensorflow as tf

    import tfplan

    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

    # os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    os.environ["OMP_NUM_THREADS"] = str(psutil.cpu_count(logical=False))

    planner = config["planner"]
    rddl = config["rddl"]
    filepath = os.path.join(config["logdir"], "data.csv")

    config["optimization"] = {
        "optimizer": config["optimizer"],
        "learning_rate": config["learning_rate"],
    }

    env = rddlgym.make(rddl, mode=rddlgym.GYM, config=config)
    env.set_horizon(config["horizon"])

    planner = tfplan.make(planner, rddl, config)

    with rddlgym.Runner(env, planner, debug=config["verbose"]) as runner:
        trajectory, _ = runner.run()
        trajectory.save(filepath)
        print(trajectory.as_dataframe())

    # planner.save_stats()
