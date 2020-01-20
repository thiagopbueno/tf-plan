# pylint: disable=missing-docstring

import os
import itertools

from tuneconfig import TuneConfig, grid_search


BASE_DIR = "./20200119/"

RDDL = ["Reservoir-10", "Reservoir-20", "Reservoir-30"]
PLANNERS = ["straightline", "hindsight"]


def format_func(param):
    fmt = {
        "batch_size": "batch",
        "horizon": "hr",
        "learning_rate": "lr",
        "optimizer": "opt",
    }
    return fmt.get(param, param)


CONFIG_TEMPLATE = TuneConfig(
    {
        "batch_size": grid_search([32, 128, 512]),
        "horizon": 40,
        "learning_rate": grid_search([0.01, 0.1]),
        "epochs": 300,
        "optimizer": grid_search(["Adam", "RMSProp", "GradientDescent"]),
        "num_samples": 30,
        "num_workers": 10,
    },
    format_func=format_func,
)


if __name__ == "__main__":

    for rddl, planner in itertools.product(RDDL, PLANNERS):
        experiment_dir = os.path.join(BASE_DIR, rddl, planner)
        json_files_created = CONFIG_TEMPLATE.dump(
            dirpath=experiment_dir, subfolders=True
        )
        print(f">> Created JSON config files ({rddl} / {planner}):")
        print("\n".join(json_files_created))
        print()
