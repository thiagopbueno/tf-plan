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

import json

import click
import psutil

import tuneconfig

import tfplan


@click.command()
@click.argument(
    "planner", type=click.Choice(["tensorplan", "straightline", "hindsight"])
)
@click.argument("rddl")
@click.option(
    "--horizon", "-hr",
    type=click.IntRange(min=1),
    default=40,
    help="Number of evaluation timesteps.",
    show_default=True
)
@click.option(
    "--planning-horizon", "-phr",
    type=click.IntRange(min=1),
    help="Number of planning timesteps."
)
@click.option(
    "--optimizer", "--opt",
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
@click.option(
    "--epochs", "-e",
    type=click.IntRange(min=1),
    default=500,
    show_default=True,
    help="Number of training epochs."
)
@click.option(
    "--epoch-scheduler", "-sch",
    nargs=3,
    type=int,
    help="Training epoch scheduler (start, final, delta)."
)
@click.option(
    "--batch-size", "-b",
    default=128,
    help="Number of trajectories in a batch.",
    show_default=True,
)
@click.option(
    "--num-samples", "-ns",
    type=int,
    default=1,
    help="Number of runs.",
    show_default=True,
)
@click.option(
    "--num-workers", "-nw",
    type=click.IntRange(min=1, max=psutil.cpu_count()),
    default=1,
    help=f"Number of worker processes (min=1, max={psutil.cpu_count()}).",
    show_default=True,
)
@click.option(
    "--logdir",
    type=click.Path(),
    default="/tmp/tfplan/",
    help="Directory used for logging training summaries.",
    show_default=True,
)
@click.option(
    "--config", "-c",
    type=click.File("r"),
    help="Configuration JSON file."
)
@click.option(
    "--verbose", "-v",
    is_flag=True,
    help="Verbosity flag."
)
@click.version_option()
def cli(**kwargs):
    """
    Planning via gradient-based optimization in TensorFlow.

    \b
    Args:
        RDDL Filename or rddlgym domain/instance id.
    """

    config = kwargs

    if kwargs["config"]:
        json_config = json.load(kwargs["config"])
        config.update(json_config)
        del config["config"]

    config["optimization"] = {
        "optimizer": config["optimizer"],
        "learning_rate": config["learning_rate"],
    }

    def format_fn(param):
        fmt = {
            "batch_size": "batch",
            "horizon": "hr",
            "learning_rate": "lr",
            "optimizer": "opt",
            "num_samples": None,
            "num_workers": None,
            "config": None,
            "logdir": None,
            "optimization": None,
            "planner": None,
            "rddl": None,
            "verbose": None,
        }
        return fmt.get(param, param)

    config_iterator = tuneconfig.ConfigFactory(config, format_fn)

    runner = tuneconfig.Experiment(config_iterator, config["logdir"])
    runner.start()
    runner.run(tfplan.run, config["num_samples"], config["num_workers"])
