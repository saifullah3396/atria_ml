from atria_registry.module_registry import ModuleRegistry
from atria_registry.registry_group import RegistryGroup

_initialized = False


def init_registry():
    global _initialized
    if _initialized:
        return
    _initialized = True
    ModuleRegistry().add_registry_group(
        name="TASK_PIPELINE", registry_group=RegistryGroup(name="task_pipeline")
    )
    ModuleRegistry().add_registry_group(
        name="METRIC", registry_group=RegistryGroup(name="metric", is_factory=True)
    )
    ModuleRegistry().add_registry_group(
        name="LR_SCHEDULER",
        registry_group=RegistryGroup(name="lr_scheduler", is_factory=True),
    )
    ModuleRegistry().add_registry_group(
        name="OPTIMIZER",
        registry_group=RegistryGroup(name="optimizer", is_factory=True),
    )
    ModuleRegistry().add_registry_group(
        name="ENGINE", registry_group=RegistryGroup(name="engine")
    )
