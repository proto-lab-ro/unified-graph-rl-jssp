"""
Reusable instance generator abstractions for JSSP environments.

These helpers make it easy to plug custom instance sources (datasets,
procedural generators, etc.) into environments without changing their logic.
"""

from __future__ import annotations

import csv
import glob
import inspect
import os
import pickle
import random
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable

from jssp_core.instances import get_instance
from jssp_core.instances.jssp import JSSPInstance, _parse_instance
from jssp_core.registry import Registry


@runtime_checkable
class InstanceGenerator(Protocol):
    """Protocol for objects that can provide JSSP instances on demand."""

    def generate(self, seed: int | None = None) -> JSSPInstance:
        """Return the next instance. Implementations may optionally use a seed."""
        ...

    def __iter__(self) -> InstanceGenerator:
        """Allow repeated iteration over the generator."""
        ...

    def __next__(self) -> JSSPInstance:
        """Yield the next instance so the generator is iterable."""
        ...


InstanceGeneratorLike = InstanceGenerator | Callable[..., JSSPInstance]
InstanceGeneratorSpec = str | InstanceGeneratorLike


GeneratorFactory = Callable[..., InstanceGenerator]


INSTANCE_GENERATOR_REGISTRY = Registry[InstanceGenerator]("InstanceGenerator")


@INSTANCE_GENERATOR_REGISTRY.register("curriculum_instance", "predefined_instances")
class PredefinedInstanceGenerator:
    """
    Instance generator that returns instances from a predefined list.

    Modes:
        - "sequential": returns instances in order and loops forever.
        - "random": returns a random instance each time.
    """

    def __init__(
        self,
        instances: list[JSSPInstance | str],
        num_jobs: int | None = None,
        num_machines: int | None = None,
        mode: str = "sequential",
        reset_idx: bool = True,
        **kwargs,
    ):
        if mode not in ("sequential", "random"):
            raise ValueError("mode must be 'sequential' or 'random'")

        if not instances:
            raise ValueError(
                "PredefinedInstanceGenerator requires a non-empty list of instances."
            )

        self.instances = list(instances)
        self.mode = mode
        self.num_jobs = num_jobs
        self.num_machines = num_machines

        self.idx = 0
        self.rng = random.Random()

        if reset_idx:
            self.idx = 0

    @staticmethod
    def from_csv(
        csv_path: str | Path,
        num_jobs: int | None = None,
        num_machines: int | None = None,
        mode: str = "sequential",
        **kwargs,
    ) -> PredefinedInstanceGenerator:
        """Create a generator from instances stored in a CSV file."""
        instances = PredefinedInstanceGenerator.load_instances_from_csv(csv_path)
        return PredefinedInstanceGenerator(
            instances=instances,
            num_jobs=num_jobs,
            num_machines=num_machines,
            mode=mode,
            **kwargs,
        )

    # Instance generation
    def generate(self, seed: int | None = None) -> JSSPInstance:
        """Return the next instance based on the chosen mode."""
        if seed is not None:
            self.rng.seed(seed)

        if self.mode == "sequential":
            instance = self.instances[self.idx]
            self.idx = (self.idx + 1) % len(self.instances)  # loop forever
        else:
            instance = self.rng.choice(self.instances)

        if isinstance(instance, JSSPInstance):
            return instance
        if isinstance(instance, str):
            return _parse_instance(instance)

        return instance  # Fallback

    def update(self, instances: list, mode: str | None = None, reset_idx: bool = True):
        """Update the underlying instance list in-place."""
        if not instances:
            raise ValueError("instances must be a non-empty list")

        self.instances = list(instances)
        if mode:
            if mode not in ("sequential", "random"):
                raise ValueError("mode must be 'sequential' or 'random'")
            self.mode = mode

        if reset_idx or self.idx >= len(self.instances):
            self.idx = 0
        else:
            self.idx = self.idx % len(self.instances)

    @staticmethod
    def load_instances_from_csv(csv_path: str | Path) -> list[str]:
        """Load JSSP instances from a CSV file (expects 'instances' column)."""
        csv_path = Path(csv_path)
        if not csv_path.exists():
            raise FileNotFoundError(f"Instance CSV file not found: {csv_path}")

        instances: list[str] = []
        with csv_path.open("r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            if "instances" not in reader.fieldnames:
                raise ValueError(
                    f"'instances' column not found in CSV. Found: {reader.fieldnames}"
                )

            for row in reader:
                text = row.get("instances")
                if text and text.strip():
                    instances.append(text.strip())

        if not instances:
            raise ValueError(f"No instances loaded from CSV file: {csv_path}")
        return instances

    # Iterator interface
    def __iter__(self):
        return self

    def __next__(self) -> JSSPInstance:
        return self.generate()


@INSTANCE_GENERATOR_REGISTRY.register("random_uniform", "uniform_random", "default")
@dataclass
class RandomInstanceGenerator:
    """
    Default generator that replicates the previous uniform random distribution.
    """

    num_jobs: int
    num_machines: int
    min_duration: int = 1
    max_duration: int = 10

    def generate(self, seed: int | None = None) -> JSSPInstance:
        return get_instance(
            {
                "type": "random",
                "num_jobs": self.num_jobs,
                "num_machines": self.num_machines,
                "min_duration": self.min_duration,
                "max_duration": self.max_duration,
                "seed": seed,
            }
        )

    def __iter__(self) -> RandomInstanceGenerator:
        return self

    def __next__(self) -> JSSPInstance:
        return self.generate()


@INSTANCE_GENERATOR_REGISTRY.register("truncated_normal")
@dataclass
class TruncatedNormalInstanceGenerator:
    """
    Default generator that replicates the previous uniform random distribution.
    """

    num_jobs: int
    num_machines: int
    min_duration: int = 1
    max_duration: int = 10
    interval: int = 10
    std: float = 5.0

    def generate(self, seed: int | None = None) -> JSSPInstance:
        return get_instance(
            {
                "type": "truncated_normal",
                "num_jobs": self.num_jobs,
                "num_machines": self.num_machines,
                "min_duration": self.min_duration,
                "max_duration": self.max_duration,
                "seed": seed,
                "interval": self.interval,
                "std": self.std,
            }
        )

    def __iter__(self) -> TruncatedNormalInstanceGenerator:
        return self

    def __next__(self) -> JSSPInstance:
        return self.generate()


@INSTANCE_GENERATOR_REGISTRY.register("file", "from_file")
class FileInstanceGenerator:
    """
    Generator that loads JSSP instances from files matching a pattern.
    """

    def __init__(
        self,
        file_pattern: str,
        num_jobs: int | None = None,
        num_machines: int | None = None,
        **kwargs,
    ):
        self.files = sorted(glob.glob(file_pattern))
        self._iterator = iter(self.files)

    def generate(self, seed: int | None = None) -> JSSPInstance:
        try:
            file_path = next(self._iterator)
        except StopIteration as err:
            raise StopIteration from err

        instance = JSSPInstance.from_file(file_path)
        instance.name = os.path.basename(file_path)
        return instance

    def __iter__(self) -> FileInstanceGenerator:
        self._iterator = iter(self.files)
        return self

    def __next__(self) -> JSSPInstance:
        return self.generate()


@INSTANCE_GENERATOR_REGISTRY.register("pickle", "from_pickle")
class PickleInstanceGenerator:
    """
    Generator that loads JSSP instances from a pickle file.
    """

    def __init__(
        self,
        file_path: str,
        num_jobs: int | None = None,
        num_machines: int | None = None,
        **kwargs,
    ):
        with open(file_path, "rb") as f:
            self.instances = pickle.load(f)

        if not isinstance(self.instances, list):
            raise ValueError(
                f"Expected a list of instances in {file_path}, got {type(self.instances)}"
            )

        self._iterator = iter(self.instances)

    def generate(self, seed: int | None = None) -> JSSPInstance:
        try:
            return next(self._iterator)
        except StopIteration as err:
            raise StopIteration from err

    def __iter__(self) -> PickleInstanceGenerator:
        self._iterator = iter(self.instances)
        return self

    def __next__(self) -> JSSPInstance:
        return self.generate()


class CallableInstanceGenerator:
    """Adapter that turns an arbitrary callable into an InstanceGenerator."""

    def __init__(self, fn: Callable[..., JSSPInstance]):
        if not callable(fn):
            raise TypeError("Instance generator must be callable")

        self._fn = fn
        signature = inspect.signature(fn)
        self._seed_param = None
        self._supports_var_kwargs = False

        for param in signature.parameters.values():
            if param.kind == inspect.Parameter.VAR_KEYWORD:
                self._supports_var_kwargs = True
            if param.name == "seed":
                self._seed_param = param
                break

    def generate(self, seed: int | None = None) -> JSSPInstance:
        if self._seed_param is not None:
            if self._seed_param.kind == inspect.Parameter.POSITIONAL_ONLY:
                return self._fn(seed)
            return self._fn(seed=seed)

        if self._supports_var_kwargs:
            return self._fn(seed=seed)

        return self._fn()

    def __iter__(self) -> CallableInstanceGenerator:
        return self

    def __next__(self) -> JSSPInstance:
        return self.generate()


@dataclass
class LimitedInstanceGenerator:
    """
    Wrapper that limits the number of instances generated.
    """

    generator: InstanceGenerator
    limit: int
    _count: int = 0

    def generate(self, seed: int | None = None) -> JSSPInstance:
        if self._count >= self.limit:
            raise StopIteration
        self._count += 1
        return self.generator.generate(seed)

    def __iter__(self) -> LimitedInstanceGenerator:
        return self

    def __next__(self) -> JSSPInstance:
        return self.generate()


def register_instance_generator(*names: str):
    """Decorator to register an instance generator class under one or more names."""
    return INSTANCE_GENERATOR_REGISTRY.register(*names)


def get_instance_generator(
    name: str, *, num_jobs: int, num_machines: int, **kwargs
) -> InstanceGenerator:
    """Instantiate a registered generator."""
    cls = INSTANCE_GENERATOR_REGISTRY.get(name)
    return cls(num_jobs=num_jobs, num_machines=num_machines, **kwargs)


def initialize_instance_generator(
    generator: InstanceGeneratorSpec | None,
    *,
    num_jobs: int,
    num_machines: int,
    generator_kwargs: dict | None = None,
    stop_count: int | None = None,
) -> InstanceGenerator:
    """
    Front door to obtain an InstanceGenerator from any supported input.

    Args:
        generator: Optional source provided by the caller (name, callable, or object).
        num_jobs: Number of jobs expected by the environment.
        num_machines: Number of machines expected by the environment.
        generator_kwargs: Additional kwargs compiled for generator construction.
        stop_count: Optional limit on the number of instances to generate.
    """
    kwargs = dict(generator_kwargs or {})

    gen: InstanceGenerator
    if generator is None:
        gen = RandomInstanceGenerator(
            num_jobs=num_jobs,
            num_machines=num_machines,
            min_duration=kwargs.pop("min_duration", 1),
            max_duration=kwargs.pop("max_duration", 10),
        )
    elif isinstance(generator, str):
        # Support PredefinedInstanceGenerator loading from file via kwargs
        if generator in ("predefined_instances", "curriculum_instance"):
            instances_file = kwargs.pop("instances_file", None)
            instances = kwargs.pop("instances", None)
            if instances_file:
                gen = PredefinedInstanceGenerator.from_csv(
                    instances_file,
                    num_jobs=num_jobs,
                    num_machines=num_machines,
                    **kwargs,
                )
            elif instances:
                gen = PredefinedInstanceGenerator(
                    instances=instances,
                    num_jobs=num_jobs,
                    num_machines=num_machines,
                    **kwargs,
                )
            else:
                raise ValueError(
                    f"Generator '{generator}' requires 'instances' or 'instances_file' in kwargs."
                )
        else:
            gen = get_instance_generator(
                generator, num_jobs=num_jobs, num_machines=num_machines, **kwargs
            )
    elif isinstance(generator, InstanceGenerator):
        gen = generator
    elif callable(generator):
        gen = CallableInstanceGenerator(generator)
    else:
        raise TypeError(
            "instance_generator must be an InstanceGenerator or callable returning a JSSPInstance"
        )

    if stop_count is not None:
        gen = LimitedInstanceGenerator(gen, stop_count)

    return gen


ensure_instance_generator = initialize_instance_generator


__all__ = [
    "CallableInstanceGenerator",
    "FileInstanceGenerator",
    "GeneratorFactory",
    "InstanceGenerator",
    "InstanceGeneratorLike",
    "InstanceGeneratorSpec",
    "LimitedInstanceGenerator",
    "PickleInstanceGenerator",
    "RandomInstanceGenerator",
    "get_instance_generator",
    "initialize_instance_generator",
    "register_instance_generator",
    "ensure_instance_generator",
    "TruncatedNormalInstanceGenerator",
    "PredefinedInstanceGenerator",
]

if __name__ == "__main__":
    gen = initialize_instance_generator(
        "truncated_normal",
        num_jobs=4,
        num_machines=6,
    )
    print(gen)
    for _ in range(1):
        instance = gen.generate()

        print(instance[0])
        print(instance[1])
