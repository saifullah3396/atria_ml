_initialized = False


def init_registry():
    from atria_registry.module_registry import ModuleRegistry
    from atria_registry.registry_group import RegistryGroup

    from atria_ml.registry.registry_groups import (
        LRSchedulerRegistryGroup,
        OptimizersRegistryGroup,
    )

    global _initialized
    if _initialized:
        return
    _initialized = True
    ModuleRegistry().add_registry_group(
        name="TASK_PIPELINE",
        registry_group=RegistryGroup(name="task_pipeline", default_provider="atria_ml"),
    )
    ModuleRegistry().add_registry_group(
        name="LR_SCHEDULER",
        registry_group=LRSchedulerRegistryGroup(
            name="lr_scheduler", default_provider="atria_ml"
        ),
    )
    ModuleRegistry().add_registry_group(
        name="OPTIMIZER",
        registry_group=OptimizersRegistryGroup(
            name="optimizer", default_provider="atria_ml"
        ),
    )
    ModuleRegistry().add_registry_group(
        name="ENGINE",
        registry_group=RegistryGroup(name="engine", default_provider="atria_ml"),
    )
