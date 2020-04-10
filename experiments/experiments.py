# pylint: disable=missing-docstring

import datetime
import os

import click
import psutil
import tuneconfig


PLANNERS = ["straightline", "hindsight"]


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
@click.argument("config", type=click.Path())
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

    config_factory = tuneconfig.ConfigFactory.from_json(
        kwargs["config"], format_fn=format_fn
    )

    config_factory.update(
        {
            "rddl": rddl,
            "num_samples": num_samples,
            "num_workers": num_workers,
            "verbose": False,
        }
    )

    for planner in PLANNERS:
        print(f"::: planner={planner} :::")

        logdir = os.path.join(base_dir, rddl, planner)

        config_factory.update(
            {"planner": planner, "logdir": logdir,}
        )

        analysis = tuneconfig.run_experiment(
            tf_plan_runner,
            config_factory,
            logdir,
            num_samples=num_samples,
            num_workers=num_workers,
            verbose=verbose,
        )
        print()
        analysis.info()
        print()


@cli.command()
@click.argument("config", type=click.Path())
@click.argument("paths", nargs=-1, required=True)
@click.option("-s", "--show-fig", is_flag=True, help="Show figure.")
@click.option("-o", "--output", type=click.Path(), help="Output filepath.")
def plot(config, paths, show_fig, output):
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
    plotter.plot_chart_from_spec(config, show_fig=show_fig, filename=output)


def tf_plan_runner(config):
    # pylint: disable=import-outside-toplevel

    import rddlgym
    import tfplan

    import tensorflow as tf

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
        trajectory = runner.run()
        trajectory.save(filepath)

    planner.save_stats()


if __name__ == "__main__":
    cli()
