"""
This module defines type aliases for optimizer and learning rate scheduler types.

The type aliases are used to standardize the expected types for optimizers and learning rate schedulers
throughout the training module.

Imports:
    - partial: A higher-order function from functools that allows partial application of arguments.
    - Any, Dict, TypeAlias, Union: Typing utilities from the typing module to define flexible type annotations.

Type Aliases:
    - OptimizerType: Represents either a partially applied optimizer or a dictionary of such partial optimizers.
    - LRSchedulerType: Represents either a partially applied learning rate scheduler or a dictionary of such partial schedulers.
"""

from functools import partial
from typing import Any, TypeAlias

OptimizerType: TypeAlias = partial[Any] | dict[str, partial[Any]]
"""
OptimizerType:
    A type alias representing an optimizer configuration.

    This can either be a single partially applied optimizer function or a dictionary
    where the keys are strings (e.g., optimizer names) and the values are partially
    applied optimizer functions. We use Any type here to bypass torch imports as it can lead to
    significant start time delays, but essentially Any object will be replaced with the
    torch.optim.Optimizer type.
"""

LRSchedulerType: TypeAlias = partial[Any] | dict[str, partial[Any]]
"""
LRSchedulerType:
    A type alias representing a learning rate scheduler configuration.

    This can either be a single partially applied learning rate scheduler function
    or a dictionary where the keys are strings (e.g., scheduler names) and the values
    are partially applied scheduler functions. We use Any type here to bypass torch imports as it can lead to
    significant start time delays. We use Any type here to bypass torch imports as it can lead to
    significant start time delays, but essentially Any object will be replaced with the
    torch.optim.lr_scheduler.LRScheduler type.
"""
