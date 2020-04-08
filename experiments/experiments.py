# pylint: disable=missing-docstring

import datetime
import os

import click
import psutil
import tuneconfig


PLANNERS = ["straightline", "hindsight"]


BASE_CONFIG = {
    "batch_size": tuneconfig.grid_search([64]),
    "horizon": 5,
    "learning_rate": tuneconfig.grid_search([0.005]),
    "epochs": tuneconfig.grid_search([100]),
    "optimizer": tuneconfig.grid_search(["Adam", "RMSProp", "GradientDescent"])
}


PLOT_CONFIG = {
    "targets": ["loss:0", "loss:4"],
    "anchors": ["batch=64", "lr=0.005"],
    "x_axis": None,
    "y_axis": "optimizer",
    "kwargs": {"target_x_axis_label": "Epochs", "target_y_axis_label": "Loss"}
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


@click.group()
def cli():
    pass


@cli.command()
@click.argument("rddl")
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
    default=f"{datetime.date.today()}/",
    help=f"Root directory for logging trial results.",
    show_default=True,
)
@click.option("-v", "--verbose", is_flag=True, help="Verbosity flag.")
def run(**kwargs):
    """Run experiments."""
    rddl = kwargs["rddl"]
    base_dir = kwargs["logdir"]
    num_samples = kwargs["num_samples"]
    num_workers = kwargs["num_workers"]
    verbose = kwargs["verbose"]

    BASE_CONFIG["rddl"] = rddl
    BASE_CONFIG["num_samples"] = num_samples
    BASE_CONFIG["num_workers"] = num_workers
    BASE_CONFIG["verbose"] = False

    for planner in PLANNERS:
        print(f"::: planner={planner} :::")
        logdir = os.path.join(base_dir, rddl, planner)
        BASE_CONFIG["planner"] = planner
        BASE_CONFIG["logdir"] = logdir

        analysis = tuneconfig.run_experiment(
            tf_plan_runner,
            tuneconfig.ConfigFactory(BASE_CONFIG, format_fn=format_fn),
            logdir,
            num_samples=num_samples,
            num_workers=num_workers,
            verbose=verbose
        )
        print()
        analysis.info()
        print()


@cli.command()
@click.argument("paths", nargs=-1, required=True)
@click.option(
    "-s", "--show-fig",
    is_flag=True,
    help="Show figure."
)
@click.option(
    "--filename",
    help="Output filepath."
)
def plot(paths, show_fig, filename):
    """Plot results."""
    prefix = os.path.commonprefix(paths)

    analysis_lst = []
    for path in paths:
        name = path.replace(prefix, "")
        analysis = tuneconfig.ExperimentAnalysis(path, name=name)
        analysis.setup()
        analysis.info()
        print()
        analysis_lst.append(analysis)

    plotter = tuneconfig.ExperimentPlotter(analysis_lst)
    kwargs = {
        **PLOT_CONFIG,
        "show_fig": show_fig,
        "filename": filename,
    }
    plotter.plot(**kwargs)


def tf_plan_runner(config):
    # pylint: disable=import-outside-toplevel

    # os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    os.environ["OMP_NUM_THREADS"] = str(psutil.cpu_count(logical=False))

    import rddlgym
    import tfplan

    import tensorflow as tf

    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

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
        trajectory = runner.run()
        trajectory.save(filepath)

    planner.save_stats()


if __name__ == "__main__":
    cli()
