import sys
import types
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from gymnasium import spaces
from omegaconf import OmegaConf

from jssp_core.domain.base_dispatcher import DispatcherBase
from jssp_core.domain.domains import ObservationType
from jssp_core.domain.observation import ObservationProvider
from jssp_core.instances import JSSPInstance
from jssp_core.schedule import Schedule
from jssp_core.solver.base import (
    Heuristic,
    JSSPSolverBase,
    SolveOutput,
    SolverType,
)
from tools.storage.base import (
    ModelStorageBackend,
    RemoteFileInfo,
    create_storage_backend,
)


def _build_jssp_instance() -> JSSPInstance:
    """Small helper to generate deterministic instances across tests."""
    return JSSPInstance(
        [
            [(0, 1), (1, 2)],
            [(1, 1), (0, 2)],
        ]
    )


class DummyObservationProvider(ObservationProvider):
    """Concrete provider to exercise the abstract ObservationProvider contract."""

    def __init__(self, schedule: Schedule):
        self._space: spaces.Dict | None = None
        super().__init__(schedule)

    @property
    def name(self) -> str:
        return "dummy"

    def _build_space(self) -> spaces.Dict:
        return spaces.Dict(
            {
                "jobs": spaces.Box(
                    low=0.0,
                    high=10.0,
                    shape=(self.num_jobs,),
                    dtype=np.float32,
                ),
                "machines": spaces.Box(
                    low=0.0,
                    high=10.0,
                    shape=(self.num_machines,),
                    dtype=np.float32,
                ),
            }
        )

    def get_observation_space(self) -> spaces.Space:
        if (
            self._space is None
            or self._space["jobs"].shape[0] != self.num_jobs
            or self._space["machines"].shape[0] != self.num_machines
        ):
            self._space = self._build_space()
        return self._space

    def get_observation(self, schedule: Schedule) -> dict[str, np.ndarray]:
        return {
            "jobs": np.full(self.num_jobs, schedule.num_jobs, dtype=np.float32),
            "machines": np.full(
                self.num_machines, schedule.num_machines, dtype=np.float32
            ),
        }

    def get_observation_space_trl(self):
        return {"shape": (self.num_jobs, self.num_machines)}

    @property
    def observation_type(self) -> ObservationType:
        return ObservationType.FLAT


def test_observation_provider_base_handles_schedule_variants():
    schedule = Schedule(_build_jssp_instance())
    provider = DummyObservationProvider(schedule)

    assert provider.num_jobs == 2
    assert provider.num_machines == 2
    assert provider.name == "dummy"

    obs = provider.get_observation(schedule)
    assert provider.validate_observation(obs)
    flattened = provider.get_observation_flattened(schedule)
    assert flattened.shape[0] == provider.num_jobs + provider.num_machines

    invalid_obs = obs.copy()
    invalid_obs["jobs"] = np.zeros(provider.num_jobs + 1, dtype=np.float32)
    assert provider.validate_observation(obs)
    assert not provider.validate_observation(invalid_obs)


class DummyDispatcher(DispatcherBase):
    """Minimal dispatcher to check orchestration logic in DispatcherBase."""

    def __init__(self, cfg):
        super().__init__(cfg)
        self.call_order: list[str] = []

    def setup_instance(self):
        self.call_order.append("instance")
        self.instance = "instance"
        return self.instance

    def setup_environment(self):
        self.call_order.append("environment")
        self.env = "env"
        return self.env

    def setup_test_environment(self):
        self.call_order.append("test_environment")
        return "test-env"

    def setup_models(self):
        self.call_order.append("models")
        self.policies = {"pi": object()}
        self.combined_policies = "combined"
        self.value_module = "value"
        self.shared_extractor = "shared"
        return (self.policies, self.combined_policies), self.value_module

    def setup_loss_modules(self):
        self.call_order.append("loss_modules")
        self.loss_modules = {"policy": "loss"}
        self.advantage_module = "adv"
        return self.loss_modules, self.advantage_module

    def setup_optimizers(self):
        self.call_order.append("optimizers")
        self.optimizers = {"policy": "opt"}
        return self.optimizers

    def setup_schedulers(self):
        self.call_order.append("schedulers")
        self.schedulers = {"policy": "sched"}
        return self.schedulers

    def setup_collector(self):
        self.call_order.append("collector")
        self.collector = "collector"
        return self.collector

    def setup_replay_buffer(self):
        self.call_order.append("replay_buffer")
        self.replay_buffers = {"policy": "buffer"}
        return self.replay_buffers


def test_dispatcher_base_setup_allows_introspection():
    cfg = OmegaConf.create({"seed": 7})
    dispatcher = DummyDispatcher(cfg)
    components = dispatcher.setup_all()

    assert dispatcher.call_order == [
        "instance",
        "environment",
        "test_environment",
        "models",
        "loss_modules",
        "optimizers",
        "schedulers",
        "collector",
        "replay_buffer",
    ]
    assert components["instance"] == "instance"
    assert components["env"] == "env"
    assert components["test_env"] == "test-env"
    assert components["loss_modules"] == {"policy": "loss"}
    assert components["advantage_module"] == "adv"
    assert components["shared_extractor"] == "shared"
    assert dispatcher.get_component("collector") == "collector"

    with pytest.raises(AttributeError):
        dispatcher.get_component("not_there")

    assert dispatcher.has_component("collector")
    assert not dispatcher.has_component("missing_component")
    assert "collector" in repr(dispatcher)


class DummySolver(JSSPSolverBase):
    """Concrete solver for testing base features."""

    def __init__(self):
        super().__init__()
        self.solve_calls = 0

    @classmethod
    def _create_from_package_root(
        cls,
        package_root: Path,
        manifest: dict[str, Any] | None,
        **solver_kwargs,
    ):
        # For tests we only need a bare solver instance; metadata assignment
        # happens in JSSPSolverBase.from_package().
        return cls(**solver_kwargs)

    def solve(self, instance: JSSPInstance) -> Schedule:
        self.solve_calls += 1
        schedule = Schedule(instance)
        while not schedule.is_complete():
            made_progress = False
            for job_id in range(schedule.num_jobs):
                if schedule.schedule_job(job_id):
                    made_progress = True
                    break
            if not made_progress:
                raise RuntimeError("Failed to schedule job")
        return schedule

    def get_type(self) -> SolverType:
        return SolverType.ML


def test_solver_base_batch_and_config_hash(tmp_path):
    solver = DummySolver()
    instances = [_build_jssp_instance(), _build_jssp_instance()]
    solutions = solver.solve_batch(instances)

    assert solver.solve_calls == 2
    assert all(solution.is_complete() for solution in solutions)

    solve_output = solver.solve_with_info(_build_jssp_instance())
    assert isinstance(solve_output, SolveOutput)
    assert solve_output.solution.is_complete()
    assert solve_output.info == {}

    cfg_path = tmp_path / "solver.yaml"
    cfg_path.write_text("value: 1\n")
    solver.load_config_from_yaml(cfg_path)
    first_hash = solver.get_config_hash()
    solver.cfg.value = 2
    updated_hash = solver.refresh_config_hash()
    assert first_hash != updated_hash

    info = solver.get_info()
    assert info["name"] == "DummySolver"
    assert info["type"] == SolverType.ML


class RoundRobinHeuristic(Heuristic):
    """Simple heuristic to ensure default solve() loop behaves as expected."""

    def __init__(self):
        self.step_calls: list[int] = []

    def step(self, current_schedule, *_) -> int:
        for job_id in range(current_schedule.num_jobs):
            if current_schedule.can_schedule_job(job_id):
                self.step_calls.append(job_id)
                return job_id
        raise RuntimeError("No schedulable job")


def test_heuristic_default_solve_schedules_all_jobs():
    heuristic = RoundRobinHeuristic()
    schedule = heuristic.solve(_build_jssp_instance())

    assert schedule.is_complete()
    assert heuristic.step_calls  # ensure step was invoked
    assert heuristic.get_config_hash() == "RoundRobinHeuristic"


class DummyBackend:
    """Runtime-checkable backend used to validate the Protocol contract."""

    def __init__(self):
        self.operations: list[str] = []

    def upload_file(self, local_path: Path, remote_path: str) -> None:
        self.operations.append(f"upload:{local_path}->{remote_path}")

    def download_file(self, remote_path: str, local_path: Path) -> None:
        self.operations.append(f"download:{remote_path}->{local_path}")

    def upload_directory(self, local_dir: Path, remote_prefix: str = "") -> None:
        self.operations.append(f"upload_dir:{local_dir}:{remote_prefix}")

    def download_directory(self, local_dir: Path, remote_prefix: str = "") -> None:
        self.operations.append(f"download_dir:{local_dir}:{remote_prefix}")

    def iter_files(self, remote_prefix: str = ""):
        yield RemoteFileInfo(path=f"{remote_prefix}/file", size=1)


def test_model_storage_backend_protocol_accepts_concrete_backend(tmp_path):
    backend = DummyBackend()
    assert isinstance(backend, ModelStorageBackend)

    local_file = tmp_path / "model.pt"
    local_file.write_text("weights")
    backend.upload_file(local_file, "model.pt")
    backend.download_file("model.pt", tmp_path / "copy.pt")
    backend.upload_directory(tmp_path, "checkpoints")
    backend.download_directory(tmp_path / "dl", "checkpoints")
    files = list(backend.iter_files("remote"))

    assert backend.operations[0].startswith("upload:")
    assert files == [RemoteFileInfo(path="remote/file", size=1)]


def test_create_storage_backend_uses_stubbed_azure(monkeypatch):
    created_kwargs = {}

    class FakeAzureBackend(DummyBackend):
        def __init__(self, **kwargs):
            super().__init__()
            created_kwargs.update(kwargs)

    fake_module = types.ModuleType("tools.storage.azure_blob")
    fake_module.AzureBlobStorage = FakeAzureBackend
    monkeypatch.setitem(sys.modules, "tools.storage.azure_blob", fake_module)

    backend = create_storage_backend(
        "azure",
        container="unit-test",
        connection_string="UseDevelopmentStorage=true",
    )

    assert isinstance(backend, FakeAzureBackend)
    assert created_kwargs["container"] == "unit-test"


def test_create_storage_backend_rejects_unknown_backend():
    with pytest.raises(ValueError):
        create_storage_backend("does-not-exist")
