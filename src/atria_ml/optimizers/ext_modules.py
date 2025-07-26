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

for optimizer_name, optimizer_path in [
    ("adadelta", "torch.optim.Adadelta"),
    ("adam", "torch.optim.Adam"),
    ("adamw", "torch.optim.AdamW"),
    ("sparse_adam", "torch.optim.SparseAdam"),
    ("adagrad", "torch.optim.Adagrad"),
    ("adamax", "torch.optim.Adamax"),
    ("asgd", "torch.optim.ASGD"),
    ("lbfgs", "torch.optim.LBFGS"),
    ("rmsprop", "torch.optim.RMSprop"),
    ("rprop", "torch.optim.Rprop"),
    ("sgd", "torch.optim.SGD"),
    ("lars", "atria_ml.optimizers.lars.LARS"),
]:
    OPTIMIZER.register(name=optimizer_name)(optimizer_path)
