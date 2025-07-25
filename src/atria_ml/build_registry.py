"""
Build Registry Script

This script is responsible for building the registry of modules used in the Atria project. It imports various components from the Atria framework and writes the registry configuration to YAML files.

Usage:
    Run this script to generate the registry configuration files in the `conf` directory.

Modules Imported:
    - Core utilities for registry management.
    - Data batch samplers, pipelines, and storage managers.
    - Model definitions and pipelines.
    - Task pipelines for training and evaluation.
    - Training optimizers and schedulers.

Author: Saifullah
Date: April 14, 2025
"""

from pathlib import Path

from atria_registry.utilities import write_registry_to_yaml

# from atria_ml.task_pipelines._data_visualizer import *  # noqa
# from atria_ml.task_pipelines._evaluator import *  # noqa
# from atria_ml.task_pipelines._inferencer import *  # noqa
from atria_ml.task_pipelines._trainer import *  # noqa
from atria_ml.training.optimizers.ext_modules import *  # noqa
from atria_ml.training.optimizers.lars import *  # noqa
from atria_ml.training.schedulers.cosine_annealing_lr import *  # noqa
from atria_ml.training.schedulers.ext_modules import *  # noqa
from atria_ml.training.schedulers.lambda_lr import *  # noqa
from atria_ml.training.schedulers.polynomial_decay_lr import *  # noqa


class AtriaConfig:
    """
    Configuration details for the Atria package.

    This class provides metadata about the Atria package, including its
    name and version. It is used to centralize configuration details for the
    package.

    Attributes:
        package (str): The name of the package. Defaults to "atria".
        version (str): The version of the package, retrieved from `atria_ml.__version__`.
    """

    @classmethod
    def initialize(self):
        from hydra.core.config_store import ConfigStore
        from hydra_zen import MISSING

        ConfigStore.instance().store(
            name="__atria__",
            node={
                "defaults": ["_self_", {"task_pipeline": MISSING}],
                "hydra": {
                    "run": {"dir": "${output_dir}"},
                    "output_subdir": "hydra",
                    "job": {"chdir": False},
                    "searchpath": [
                        "pkg://atria_ml/conf",
                        "pkg://atria_metrics/conf",
                        "pkg://atria_models/conf",
                        "pkg://atria_datasets/conf",
                        "pkg://atria_transforms/conf",
                    ],
                },
                "package": "atria_ml",
                "version": "0.1.0",
                "_zen_exclude": ["hydra", "package", "version"],
            },
            provider="atria_ml",
            package="__global__",
        )


if __name__ == "__main__":
    import shutil

    config_path = Path(__file__).parent / "conf"
    if config_path.exists():
        shutil.rmtree(config_path)
    AtriaConfig.initialize()
    write_registry_to_yaml(
        str(Path(__file__).parent / "conf"),
        types=["task_pipeline", "lr_scheduler", "optimizer", "engine"],
    )
