# pylint: disable=missing-docstring

import os
import itertools

from tuneconfig import TuneConfig, grid_search
from tuneconfig.experiment import Experiment
from tuneconfig.analysis import ExperimentAnalysis
from tuneconfig.plotting import ExperimentPlotter


BASE_DIR = "./20200406/Navigation-v3/straightline"

NUM_SAMPLES = 10
NUM_WORKERS = 10


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


CONFIG_FACTORY = TuneConfig(
    {
        "planner": grid_search(["straightline"]),
        "rddl": grid_search(["Navigation-v3"]),
        "logdir": BASE_DIR,
        "verbose": False,
        "batch_size": grid_search([32]),
        "horizon": 20,
        "learning_rate": grid_search([0.005]),
        "epochs": grid_search([300]),
        "optimizer": grid_search(["Adam", "RMSProp", "GradientDescent"]),
        "num_samples": NUM_SAMPLES,
        "num_workers": NUM_WORKERS,
    },
    format_fn=format_fn,
)

# IGNORE = [
#     {"learning_rate": 0.01, "epochs": 100},
#     {"learning_rate": 0.1, "epochs": 300},
# ]


def run(config):
    import os
    import psutil
    import time

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

    start = time.time()

    env = rddlgym.make(rddl, mode=rddlgym.GYM, config=config)
    env.set_horizon(config["horizon"])

    planner = tfplan.make(planner, rddl, config)

    with rddlgym.Runner(env, planner, debug=config["verbose"]) as runner:
        trajectory = runner.run()
        df = trajectory.save(filepath)
        stats = df.describe()

    planner.save_stats()

    uptime = time.time() - start

    pid = os.getpid()

    return pid, uptime, stats


if __name__ == "__main__":

    experiment = Experiment(CONFIG_FACTORY, BASE_DIR)
    experiment.start()

    _ = experiment.run(run, NUM_SAMPLES, num_workers=NUM_WORKERS, verbose=True)

    analysis = ExperimentAnalysis(experiment.logdir)
    analysis.setup()
    analysis.info()

    plotter = ExperimentPlotter(analysis)
    targets = ["loss:0", "loss:5", "loss:10", "loss:15"]
    anchors = ["batch=32", "lr=0.005"]
    x_axis = None
    y_axis = "optimizer"
    kwargs = {"target_x_axis_label": "Epochs", "target_y_axis_label": "Loss"}
    plotter.plot(targets, anchors, x_axis, y_axis, show_fig=True, **kwargs)
