from collections.abc import Callable

from atria_registry import RegistryGroup
from pydantic import BaseModel, ConfigDict


class OptimizerConfig(BaseModel):  # or inherit from DataTransform if needed
    model_config = ConfigDict(extra="allow")
    optimizer: str

    def build(self, params):
        from atria_core.utilities.imports import _resolve_module_from_path

        optimizer_class = _resolve_module_from_path(f"torch.optim.{self.optimizer}")
        return optimizer_class(params=params, **self.model_extra)

    def __call__(self, params):
        return self.build(params)


class SchedulerConfig(BaseModel):  # or inherit from DataTransform if needed
    model_config = ConfigDict(extra="allow")
    scheduler: str

    def build(self, **kwargs):
        from atria_core.utilities.imports import _resolve_module_from_path

        optimizer_class = _resolve_module_from_path(
            f"torch.optim.lr_scheduler.{self.scheduler}"
        )
        return optimizer_class(**self.model_extra, **kwargs)

    def __call__(self, **kwargs):
        return self.build(**kwargs)


class CallableSchedulerConfig(
    SchedulerConfig
):  # or inherit from DataTransform if needed
    scheduler: Callable

    def build(self, optimizer, **kwargs):
        return self.scheduler(**self.model_extra, **kwargs)

    def __call__(self, optimizer, **kwargs):
        return self.build(optimizer=optimizer, **kwargs)


class OptimizersRegistryGroup(RegistryGroup):
    def register_optimizer(self, optimizer_path: str, optimizer_name: str):
        import inspect

        from atria_core.utilities.imports import _resolve_module_from_path
        from torch.optim.optimizer import required

        optimizer_class = _resolve_module_from_path(optimizer_path)
        signature = inspect.signature(optimizer_class.__init__)
        params = {}

        for name, param in signature.parameters.items():
            if name in ["self", "params"]:
                continue
            if param.default == param.empty or param.default is required:
                params[name] = "???"
            else:
                params[name] = param.default
        self.register_modules(
            module_paths=OptimizerConfig,
            module_names=optimizer_name,
            optimizer=optimizer_path,
            **params,
        )


class LRSchedulerRegistryGroup(RegistryGroup):
    def register_scheduler(self, scheduler_path: str, scheduler_name: str):
        import inspect

        from atria_core.utilities.imports import _resolve_module_from_path

        scheduler_class = _resolve_module_from_path(scheduler_path)
        signature = inspect.signature(scheduler_class.__init__)
        params = {}

        for name, param in signature.parameters.items():
            if name in ["self", "optimizer"]:
                continue
            if param.default == param.empty:
                params[name] = "???"
            else:
                params[name] = param.default
        self.register_modules(
            module_paths=SchedulerConfig,
            module_names=scheduler_name,
            scheduler=scheduler_path,
            **params,
        )

    def register_callable_scheduler(self, scheduler_name: str):
        """
        Register a callable scheduler with the registry.
        """

        def decorator(callable_scheduler: Callable):
            """
            Decorator to register a callable scheduler.
            """
            if not callable(callable_scheduler):
                raise ValueError("The provided scheduler must be callable.")
            self.register_modules(
                module_paths=CallableSchedulerConfig,
                module_names=scheduler_name,
                scheduler=callable_scheduler,
            )
            return callable_scheduler

        return decorator
