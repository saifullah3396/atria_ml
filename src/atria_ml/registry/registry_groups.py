from atria_registry import RegistryGroup
from atria_registry.module_builder import ModuleBuilder


class OptimizerBuilder(ModuleBuilder):
    pass


class LRSchedulerBuilder(ModuleBuilder):
    pass


class OptimizersRegistryGroup(RegistryGroup):
    __registers_as_module_builder__ = True
    __module_builder_class__ = OptimizerBuilder
    __exclude_from_builder__ = [
        "params"  # these are passed at runtime from trainer
    ]


class LRSchedulerRegistryGroup(RegistryGroup):
    __registers_as_module_builder__ = True
    __module_builder_class__ = LRSchedulerBuilder
    __exclude_from_builder__ = [  # these are passed at runtime from trainer
        "optimizer",
        "total_warmup_steps",
        "total_update_steps",
        "total_warmup_steps",
        "steps_per_epoch",
    ]
