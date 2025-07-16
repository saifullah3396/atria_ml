"""
This module registers external learning rate scheduler modules to the LR_SCHEDULER registry.

The `ext_modules` module is responsible for adding PyTorch and Ignite learning rate schedulers
into the `LR_SCHEDULER` registry. This allows these schedulers to be easily accessed and used
throughout the training pipeline.

Imports:
    - LR_SCHEDULER: A registry for learning rate schedulers from the `atria_ml.registry` module.

Functionality:
    - Registers PyTorch learning rate schedulers such as StepLR, MultiStepLR, ExponentialLR, and CyclicLR.
    - Registers the Ignite ReduceLROnPlateauScheduler for advanced learning rate scheduling.
"""

from atria_ml.registry import LR_SCHEDULER

# Register PyTorch learning rate schedulers
LR_SCHEDULER.register_modules(
    module_paths=[
        f"torch.optim.lr_scheduler.{x}"
        for x in ["StepLR", "MultiStepLR", "ExponentialLR", "CyclicLR"]
    ],
    module_names=["step_lr", "multi_step_lr", "exponential_lr", "cyclic_lr"],
)
"""
Registers the following PyTorch learning rate schedulers:

- StepLR: Decays the learning rate of each parameter group by a factor of gamma every step_size epochs.
- MultiStepLR: Decays the learning rate of each parameter group by gamma once the number of epoch reaches one of the milestones.
- ExponentialLR: Decays the learning rate of each parameter group by gamma every epoch.
- CyclicLR: Cycles the learning rate between two boundaries with a constant frequency.
"""

# Register Ignite learning rate scheduler
LR_SCHEDULER.register_modules(
    module_paths="ignite.handlers.ReduceLROnPlateauScheduler",
    module_names="reduce_lr_on_plateau",
)
"""
Registers the following Ignite learning rate scheduler:

- ReduceLROnPlateauScheduler: Reduces the learning rate when a metric has stopped improving.
"""
