# pylint: disable=missing-docstring

import os
import itertools

from tuneconfig import TuneConfig, grid_search


BASE_DIR = "./20200123/"

RDDL = ["Reservoir-10"]
PLANNERS = ["straightline", "hindsight"]


def format_fn(param):
    fmt = {
        "batch_size": "batch",
        "horizon": "hr",
        "learning_rate": "lr",
        "optimizer": "opt",
        "num_samples": None,
        "num_workers": None,
    }
    return fmt.get(param, param)


CONFIG_TEMPLATE = TuneConfig(
    {
        "batch_size": grid_search([32, 512]),
        "horizon": 40,
        "learning_rate": grid_search([0.01, 0.1]),
        "epochs": grid_search([100, 300]),
        "optimizer": grid_search(["Adam", "RMSProp", "GradientDescent"]),
        "num_samples": 10,
        "num_workers": 10,
    },
    format_fn=format_fn,
)

IGNORE = [
    {"learning_rate": 0.01, "epochs": 100},
    {"learning_rate": 0.1, "epochs": 300},
]


if __name__ == "__main__":

    for rddl, planner in itertools.product(RDDL, PLANNERS):
        experiment_dir = os.path.join(BASE_DIR, rddl, planner)
        json_config_files = CONFIG_TEMPLATE.dump(experiment_dir, ignore=IGNORE)
        print(f">> Created JSON config files ({rddl} / {planner}):")
        print("\n".join(json_config_files))
        print()
