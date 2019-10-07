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

# pylint: disable=too-many-locals


"""tfplan CLI application."""


import click


@click.command()
@click.argument(
    "planner", type=click.Choice(["tensorplan", "straightline", "hindsight"])
)
@click.argument("rddl")
@click.option(
    "--batch-size",
    "-b",
    default=128,
    help="Number of trajectories in a batch.",
    show_default=True,
)
@click.option(
    "--horizon", "-h", default=40, help="Number of timesteps.", show_default=True
)
@click.option(
    "--epochs", "-e", default=500, help="Number of training epochs.", show_default=True
)
@click.option(
    "--optimizer",
    type=click.Choice(
        [
            "Adadelta",
            "Adagrad",
            "Adam",
            "GradientDescent",
            "ProximalGradientDescent",
            "ProximalAdagrad",
            "RMSProp",
        ]
    ),
    default="GradientDescent",
    show_default=True,
)
@click.option(
    "--learning-rate",
    "-lr",
    default=1e-3,
    help="Optimizer's learning rate.",
    show_default=True,
)
@click.option("-v", "--verbose", is_flag=True, help="Verbosity flag.")
@click.version_option()
def cli(*args, **kwargs):
    """
    Planning via gradient-based optimization in TensorFlow.

    \b
    Args:
        RDDL Filename or rddlgym domain/instance id.
    """
    import rddlgym

    from tfplan.planners import Tensorplan, StraightLinePlanner

    PLANNERS = {"tensorplan": Tensorplan, "straightline": StraightLinePlanner}

    Planner = PLANNERS[kwargs["planner"]]

    rddl = kwargs["rddl"]
    env = rddlgym.make(rddl, mode=rddlgym.GYM)

    config = kwargs
    config["optimization"] = {
        "optimizer": kwargs["optimizer"],
        "learning_rate": kwargs["learning_rate"],
    }

    planner = Planner(rddl, config)

    debug = kwargs["verbose"]

    with rddlgym.Runner(env, planner, debug=debug) as runner:
        total_reward, trajectory = runner.run()
