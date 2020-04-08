# pylint: disable=missing-docstring

import os

from tuneconfig import TuneConfig, grid_search
from tuneconfig.experiment import Experiment
from tuneconfig.analysis import ExperimentAnalysis
from tuneconfig.plotting import ExperimentPlotter


PLANNERS = ["straightline", "hindsight"]
RDDL_ID = "Navigation-v3"
BASE_DIR = f"20200408/{RDDL_ID}"

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


BASE_CONFIG = {
    "planner": None,
    "rddl": RDDL_ID,
    "logdir": BASE_DIR,
    "verbose": False,
    "batch_size": grid_search([64]),
    "horizon": 20,
    "learning_rate": grid_search([0.005]),
    "epochs": grid_search([200]),
    "optimizer": grid_search(["Adam", "RMSProp", "GradientDescent"]),
    "num_samples": NUM_SAMPLES,
    "num_workers": NUM_WORKERS,
}


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
    for planner in PLANNERS:
        print(f"::: planner={planner} :::")
        logdir = os.path.join(BASE_DIR, planner)
        BASE_CONFIG["planner"] = planner
        BASE_CONFIG["logdir"] = logdir
        config_factory = TuneConfig(BASE_CONFIG, format_fn=format_fn)
        experiment = Experiment(config_factory, logdir)
        experiment.start()
        experiment.run(run, NUM_SAMPLES, num_workers=NUM_WORKERS, verbose=True)

    analysis_lst = []
    for planner in PLANNERS:
        logdir = os.path.join(BASE_DIR, planner)
        analysis = ExperimentAnalysis(logdir, name=planner)
        analysis.setup()
        analysis.info()
        print()
        analysis_lst.append(analysis)

    plotter = ExperimentPlotter(analysis_lst)
    targets = ["loss:0", "loss:5", "loss:10", "loss:15"]
    anchors = ["batch=64", "lr=0.005"]
    x_axis = None
    y_axis = "optimizer"
    kwargs = {"target_x_axis_label": "Epochs", "target_y_axis_label": "Loss"}
    plotter.plot(targets, anchors, x_axis, y_axis, show_fig=True, **kwargs)
