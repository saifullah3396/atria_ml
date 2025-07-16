"""
Optimizer Extension Module

This module registers additional optimizer modules from PyTorch into the Atria framework's
optimizer registry. It includes commonly used optimizers such as Adam, SGD, RMSprop, and others.

Dependencies:
    - atria_ml.registry: For registering optimizer modules.
    - torch.optim: For PyTorch optimizer implementations.

Author: Your Name (your.email@example.com)
Date: 2025-04-14
Version: 1.0.0
License: MIT
"""

from atria_ml.registry import OPTIMIZER

OPTIMIZER.register_modules(
    module_paths=[
        f"torch.optim.{x}"
        for x in [
            "Adadelta",
            "Adam",
            "AdamW",
            "SparseAdam",
            "Adagrad",
            "Adamax",
            "ASGD",
            "LBFGS",
            "RMSprop",
            "Rprop",
            "SGD",
        ]
    ],
    module_names=[
        "adadelta",
        "adam",
        "adamw",
        "sparse_adam",
        "adagrad",
        "adamax",
        "asgd",
        "lbfgs",
        "rmsprop",
        "rprop",
        "sgd",
    ],
)
