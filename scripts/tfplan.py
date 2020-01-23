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
    "--horizon", "-hr", default=40, help="Number of timesteps.", show_default=True
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
@click.option(
    "--num-samples",
    "-n",
    type=int,
    default=1,
    help="Number of runs.",
    show_default=True,
)
@click.option(
    "--num-workers",
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
@click.option("--config", type=click.File("r"), help="Configuration JSON file.")
@click.option("-v", "--verbose", is_flag=True, help="Verbosity flag.")
@click.version_option()
def cli(*args, **kwargs):
    """
    Planning via gradient-based optimization in TensorFlow.

    \b
    Args:
        RDDL Filename or rddlgym domain/instance id.
    """
    import functools
    import multiprocessing
    import os
    import re

    import numpy as np
    import pandas as pd
    from tqdm import tqdm

    config = kwargs

    if kwargs["config"]:
        json_config = json.load(kwargs["config"])
        config.update(json_config)
        del config["config"]

    config["optimization"] = {
        "optimizer": config["optimizer"],
        "learning_rate": config["learning_rate"],
    }

    n_samples = kwargs["num_samples"]
    num_workers = kwargs["num_workers"]

    previous_runs = [
        path for path in os.listdir(config["logdir"]) if re.search(r"run\d+$", path)
    ]
    start_id = len(previous_runs)
    pids = range(start_id, start_id + n_samples)

    pool = multiprocessing.Pool(
        processes=num_workers, initializer=tqdm.set_lock, initargs=(tqdm.get_lock(),)
    )
    trajectories = pool.map(functools.partial(run, config), pids)
    pool.close()
    pool.join()

    avg_uptime = np.mean([uptime for _, uptime, _ in trajectories])
    stddev_uptime = np.std([uptime for _, uptime, _ in trajectories])
    results = pd.concat([stats for _, _, stats in trajectories])
    results = results.groupby(results.index, sort=False).mean()
    print()
    print(f"===== Average ===== ({avg_uptime:.4f} ± {stddev_uptime:.4f} sec)")
    print(results)
    print()

    if kwargs["verbose"]:
        for i, (pid, uptime, stats) in enumerate(trajectories):
            print(f"===== Run #{i} / pid={pid} ({uptime:.4f} sec) =====")
            print(stats)
            print()


def run(config, n):
    import os
    import time

    # os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    os.environ["OMP_NUM_THREADS"] = str(psutil.cpu_count(logical=False))

    import rddlgym
    import tfplan

    import tensorflow as tf

    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

    config["run_id"] = n

    planner = config["planner"]
    rddl = config["rddl"]
    horizon = config["horizon"]
    debug = config["verbose"]

    config["logdir"] = os.path.join(config["logdir"], f"run{n}")
    filepath = os.path.join(config["logdir"], "data.csv")

    start = time.time()

    env = rddlgym.make(rddl, mode=rddlgym.GYM, config=config)
    env.set_horizon(horizon)

    planner = tfplan.make(planner, rddl, config)

    with rddlgym.Runner(env, planner, debug=debug) as runner:
        trajectory = runner.run()
        df = trajectory.save(filepath)
        stats = df.describe()

    uptime = time.time() - start

    pid = os.getpid()

    return pid, uptime, stats
