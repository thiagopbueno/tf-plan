#! /usr/bin/env python3

# pylint: disable=invalid-name

import os

import click
import psutil
import wandb

from rddlgym.builders.reservoir import ReservoirBuilder
import tuneconfig

import tfplan


def solve(config):
    """Solve online RDDL planning problem."""
    run_id = config["run_id"]

    rddl = config["rddl"]
    planner = config["planner"]

    name = f"{rddl}-{planner}-run{run_id}"
    group = config.get("group") or rddl
    job_type = config.get("job_type") or f"evaluation-{planner}"

    run = wandb.init(
        project="online-tfplan",
        name=name,
        config=config,
        tags=[planner, rddl],
        group=group,
        job_type=job_type,
        reinit=True
    )

    with run:
        tfplan.run(config)


@click.group()
def cli():
    """Setup and run experiments on Reservoir domain."""


@cli.command()
@click.argument("name")
@click.option(
    "--n-reservoirs", "-n",
    type=click.IntRange(min=1),
    default=10,
    show_default=True,
    help="Number of reservoirs."
)
@click.option(
    "--level-set-point", "-sp",
    type=click.FloatRange(min=0.0, max=1.0),
    default=0.5,
    show_default=True,
    help="Set point of each reservoir water level."
)
@click.option(
    "--level-nominal-range", "-r",
    type=click.FloatRange(min=0.1, max=0.5),
    default=0.25,
    show_default=True,
    help="Nominal range of each reservoir (i.e., [setpoint - range/2, setpoint + range/2])."
)
@click.option(
    "--init-relative-level", "-init",
    type=click.FloatRange(min=-0.5, max=0.5),
    default=-0.45,
    show_default=True,
    help="Initial water level relative to set point (underflow if < 0, overflow if > 0)."
)
@click.option(
    "--rain-mean", "-mean",
    type=click.FloatRange(min=0.0, max=1.0),
    default=0.15,
    show_default=True,
    help="Mean of the Gamma probabilistic model of rainfall."
)
@click.option(
    "--rain-variance", "-var",
    type=click.FloatRange(min=0.0, max=1.0),
    default=0.025,
    show_default=True,
    help="Variance of the Gamma probabilistic model of rainfall."
)
@click.option(
    "--json-config-dir", "-jdir",
    default="~/Projects/tf-mpc/experiments",
    show_default=True,
    help="Logdir for dumping JSON config file."
)
def setup(name, **kwargs):
    """Build RDDL file and JSON config files."""
    n_reservoirs = kwargs["n_reservoirs"]
    json_config_dir = kwargs.pop("json_config_dir")

    builder = ReservoirBuilder(
        domain_id="reservoir",
        non_fluents_id=f"res{n_reservoirs}",
        instance_id=f"inst_reservoir_res{n_reservoirs}",
        **kwargs
    )

    rddl = builder.build()
    print(rddl)

    filename = f"{name}.rddl"
    builder.save(filename)

    if not os.path.exists(json_config_dir):
        os.makedirs(json_config_dir)
    jsonfile = os.path.join(os.path.expanduser(json_config_dir), f"{name}.json")
    builder.dump_config(jsonfile)


@cli.command()
@click.argument("rddl")
@click.argument("planner", type=click.Choice(["hindsight", "straightline"]))
@click.option(
    "--group", "-g",
    help="Experiment group name."
)
@click.option(
    "--job-type", "-jt",
    help="Experiment job type."
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
def run(rddl, planner, **kwargs):
    """Run online planner for the given `rddl` file."""

    num_samples = kwargs.pop("num_samples")
    num_workers = kwargs.pop("num_workers")

    config = {
        "rddl": rddl,
        "planner": planner,
        "horizon": 40,
        "optimizer": "Adam",
        "learning_rate": 5e-2,
        "epochs": 500,
        "batch_size": 64,
        "warm_start": True,
        "verbose": False,
        "logdir": "results",
        "logger": "wandb",
        "job_type": kwargs.pop("job_type"),
        "group": kwargs.pop("group"),
    }

    def format_fn(param):
        fmt = {
            "batch_size": "batch",
            "horizon": "hr",
            "learning_rate": "lr",
            "optimizer": "opt",
            "config": None,
            "logger": None,
            "logdir": None,
            "optimization": None,
            "verbose": None,
        }
        return fmt.get(param, param)

    config_iterator = tuneconfig.ConfigFactory(config, format_fn)

    runner = tuneconfig.Experiment(config_iterator, config["logdir"])
    runner.start()

    runner.run(solve, num_samples, num_workers)


if __name__ == "__main__":
    cli()
