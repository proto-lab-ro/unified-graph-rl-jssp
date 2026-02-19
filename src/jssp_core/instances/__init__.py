"""
Project: JSSP GNN RL
File: src/jssp_core/instances/__init__.py

JSSP Instance handling module.
Provides functionality for loading, parsing, and generating JSSP instances.
"""

from pathlib import Path
from typing import Any

# Import base types and constants first (no dependencies)
from jssp_core.instances.base import F3X3_INSTANCE, FT06_INSTANCE, Job, Operation

# Import JSSP instance class and functions
from jssp_core.instances.jssp import (
    JSSPInstance,
    _load_instance,
    _parse_instance,
    generate_random_jssp_instance,
    generate_truncated_normal_jssp_instance,
    get_instance_info,
    read_yaml_specification_instance,
    read_yaml_specification_logistics,
    validate_instance,
)


###############################################################################
# Instance factory
###############################################################################


def get_instance(spec: Any) -> JSSPInstance:
    """
    Unified factory for creating/loading JSSP instances.

    Args:
        spec: Can be one of:
            - Benchmark name (e.g., "ft06", "la01") - checks jssp_instances/
            - File path string or Path object
            - Raw instance text string
            - JSSPInstance (returned as-is)
            - Dictionary specification (e.g., {"type": "random", ...})

    Returns:
        JSSPInstance: The loaded or generated instance.
    """
    # Already parsed
    if isinstance(spec, JSSPInstance):
        return spec

    # Dict-based specification
    if isinstance(spec, dict):
        spec_type = spec.get("type")
        if spec_type == "path":
            return _load_instance(str(spec["path"]))
        if spec_type == "text":
            return _parse_instance(spec["text"])
        if spec_type == "random":
            return generate_random_jssp_instance(
                spec["num_jobs"],
                spec["num_machines"],
                spec.get("min_duration", 1),
                spec.get("max_duration", 10),
            )
        if spec_type == "truncated_normal":
            return generate_truncated_normal_jssp_instance(
                spec["num_jobs"],
                spec["num_machines"],
                spec.get("min_duration", 1),
                spec.get("max_duration", 100),
                spec.get("interval", 10),
                spec.get("std", 5.0),
            )
        raise ValueError(f"Unknown instance spec type '{spec_type}' in {spec}")

    # Path or string-based resolution
    if isinstance(spec, (str, Path)):
        # 1. Check if it's a direct file path
        path = Path(spec)
        if path.exists() and path.is_file():
            return _load_instance(str(path))

        # 2. Check standard benchmark directory (jssp_instances/)
        benchmark_path = Path("jssp_instances") / str(spec)
        if benchmark_path.exists() and benchmark_path.is_file():
            return _load_instance(str(benchmark_path))

        # 3. Fallback for legacy built-in constants
        spec_str = str(spec)
        if spec_str == "ft06":
            return _parse_instance(FT06_INSTANCE)
        if spec_str == "f3x3":
            return _parse_instance(F3X3_INSTANCE)

        # 4. Treat as raw instance text if multiple tokens are present
        if " " in spec_str or "\n" in spec_str:
            return _parse_instance(spec_str)

    raise ValueError(f"Unsupported instance spec type: {spec}")


# Import generator helpers AFTER factory definition to avoid circular import
from jssp_core.instances.generators import (  # noqa: E402
    CallableInstanceGenerator,
    GeneratorFactory,
    InstanceGenerator,
    InstanceGeneratorLike,
    InstanceGeneratorSpec,
    RandomInstanceGenerator,
    ensure_instance_generator,
    get_instance_generator,
    register_instance_generator,
)


__all__ = [
    # Base types
    "Operation",
    "Job",
    # JSSP Instance class
    "JSSPInstance",
    "read_yaml_specification_instance",
    "read_yaml_specification_logistics",
    # Factory helpers
    "get_instance",
    # Instance utilities
    "get_instance_info",
    "validate_instance",
    # Generator interfaces
    "CallableInstanceGenerator",
    "GeneratorFactory",
    "InstanceGenerator",
    "InstanceGeneratorLike",
    "InstanceGeneratorSpec",
    "RandomInstanceGenerator",
    "ensure_instance_generator",
    "get_instance_generator",
    "register_instance_generator",
]
