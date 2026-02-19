from collections.abc import Callable


class Registry[T]:
    """Generic registry for pluggable components."""

    def __init__(self, name: str):
        self._name = name
        self._registry: dict[str, type[T]] = {}

    def register(self, *names: str) -> Callable[[type[T]], type[T]]:
        """Decorator to register a component class under one or more names."""

        def decorator(cls: type[T]) -> type[T]:
            reg_names = list(names)
            if not reg_names:
                name = getattr(cls, "name", cls.__name__.lower())
                if not isinstance(name, str):
                    name = cls.__name__.lower()
                reg_names.append(name)

            for reg_name in reg_names:
                if reg_name in self._registry:
                    raise ValueError(
                        f"Component '{reg_name}' already registered in {self._name}."
                    )
                self._registry[reg_name] = cls
            return cls

        return decorator

    def get(self, name: str) -> type[T]:
        """Retrieve a component class by name."""
        if name not in self._registry:
            available = ", ".join(self.list_available())
            raise ValueError(
                f"Unknown component '{name}' in {self._name}. Available: {available}"
            )
        return self._registry[name]

    def list_available(self) -> list[str]:
        """List all registered component names."""
        return list(self._registry.keys())

    def __contains__(self, name: str) -> bool:
        return name in self._registry


# Global registries
from jssp_core.domain.observation import ObservationProvider
from jssp_core.domain.reward import RewardFunction


REWARD_REGISTRY = Registry[RewardFunction]("RewardFunction")
OBSERVATION_REGISTRY = Registry[ObservationProvider]("ObservationProvider")
