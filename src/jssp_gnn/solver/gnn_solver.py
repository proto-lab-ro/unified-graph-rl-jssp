import shutil
import tarfile
from pathlib import Path

import torch
from omegaconf import DictConfig, OmegaConf
from tensordict import TensorDict
from torchrl.envs import Compose, RewardSum, StepCounter, TransformedEnv
from torchrl.envs.utils import ExplorationType, set_exploration_type
from torchrl.modules import ProbabilisticActor

from jssp_core.solver.base import JSSPSolverBase, SolveOutput, SolverType
from jssp_gnn.dispatcher.utils_matrix import create_models
from jssp_gnn.environments.jssp import GraphMatrixEnv
from jssp_gnn.utils import get_device


class RandomStepPolicy(torch.nn.Module):
    def __init__(self, policy: torch.nn.Module, n: int):
        super().__init__()
        self.policy = policy
        self.n = n
        self.step_count = 0

    def forward(self, td: TensorDict) -> TensorDict:
        self.step_count += 1
        if self.step_count % self.n == 0:
            return self.random_action(td)
        return self.policy(td)

    def random_action(self, td: TensorDict) -> TensorDict:
        mask = td.get("mask")
        if mask is None:
            # Fallback if no mask, just random ?
            # Assuming mask is present as per JSSP environment
            raise ValueError("Mask not found in observation for RandomStepPolicy")

        if mask.dim() == 1:
            valid_indices = torch.nonzero(mask).squeeze(1)
            if len(valid_indices) == 0:
                action = 0
            else:
                idx = torch.randint(0, len(valid_indices), (1,)).item()
                action = valid_indices[idx]
            td["action"] = torch.tensor(action, device=td.device)
            # Add dummy logits if expected?
            # ProbabilisticActor might not be used here, but env expects action
        elif mask.dim() == 2:
            # Batched case
            batch_size = mask.shape[0]
            actions = []
            for i in range(batch_size):
                m = mask[i]
                valid_indices = torch.nonzero(m).squeeze(1)
                if len(valid_indices) == 0:
                    a = 0
                else:
                    idx = torch.randint(0, len(valid_indices), (1,)).item()
                    a = valid_indices[idx]
                actions.append(a)
            td["action"] = torch.tensor(actions, device=td.device)

        return td

    def reset(self):
        self.step_count = 0


class GnnMatrixSolver(JSSPSolverBase):
    """
    Solver using a GNN-based policy trained with PPO.

    This solver loads a trained PyTorch model (either from a checkpoint file or a
    packaged .tar.gz archive) and uses it to solve JSSP instances.
    """

    def __init__(
        self,
        model_path: str | None,
        cfg: DictConfig | None = None,
        device: str | torch.device | None = None,
        max_steps: int = 50000,
        action_masking: bool = True,
        checkpoint_filename: str = "policy_module_final.pt",
        random_nth_step: int | None = None,
    ):
        super().__init__()
        self.model_path = model_path
        self.cfg = cfg
        self.device = device if device is not None else get_device()
        self.max_steps = max_steps
        self.action_masking = action_masking
        self.checkpoint_filename = checkpoint_filename
        self.random_nth_step = random_nth_step

        self.policy_module = None
        self._name: str | None = None

        self.config_hash: str | None = None

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str | Path,
        device: str | torch.device | None = None,
        max_steps: int = 5000,
        action_masking: bool = True,
        checkpoint_filename: str = "policy_module_final.pt",
        random_nth_step: int | None = None,
    ):
        """
        Create a solver instance from a checkpoint path (.pt file).

        This method automatically attempts to locate the associated config.yaml file
        by searching in the checkpoint's directory and parent directories.

        Args:
            checkpoint_path: Path to the .pt checkpoint file.
            device: Device to run the model on (e.g., "cpu", "cuda").
            max_steps: Maximum number of steps allowed for solving an instance.
            action_masking: Whether to use action masking during inference.
            checkpoint_filename: Name of the checkpoint file (for identification purposes).
            random_nth_step: If set, every nth step will be a random action.

        Returns:
            GnnMatrixSolver: An initialized solver instance.

        Raises:
            FileNotFoundError: If the checkpoint or config file cannot be found.
        """
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        # Try to find config
        # 1. Same directory as checkpoint
        config_path = checkpoint_path.parent / "config.yaml"

        # 2. .hydra directory in parent (standard Hydra run)
        if not config_path.exists():
            config_path = checkpoint_path.parent.parent / ".hydra" / "config.yaml"

        # 3. .hydra directory in same directory (sometimes happens)
        if not config_path.exists():
            config_path = checkpoint_path.parent / ".hydra" / "config.yaml"

        if config_path.exists():
            cfg = OmegaConf.load(config_path)
        else:
            # Fallback: try to find any config.yaml in the parent hierarchy
            print(
                f"Warning: Could not find config.yaml near {checkpoint_path}. Searching parents..."
            )
            current = checkpoint_path.parent
            found = False
            # Search up to 3 levels up
            for _ in range(3):
                candidate = current / "config.yaml"
                if candidate.exists():
                    config_path = candidate
                    found = True
                    break
                candidate = current / ".hydra" / "config.yaml"
                if candidate.exists():
                    config_path = candidate
                    found = True
                    break
                current = current.parent
                if current == current.parent:
                    break

            if found:
                cfg = OmegaConf.load(config_path)
            else:
                raise FileNotFoundError(
                    f"Could not find config.yaml for checkpoint {checkpoint_path}. "
                    "Please ensure config.yaml is in the same directory or in a .hydra folder."
                )

        return cls(
            model_path=str(checkpoint_path),
            cfg=cfg,
            device=device,
            max_steps=max_steps,
            action_masking=action_masking,
            checkpoint_filename=checkpoint_filename,
            random_nth_step=random_nth_step,
        )

    @classmethod
    def from_package(
        cls,
        package_path: str | Path,
        device: str | torch.device | None = None,
        max_steps: int = 50000,
        action_masking: bool = True,
        force_extract: bool = False,
        checkpoint_filename: str = "policy_module_final.pt",
        random_nth_step: int | None = None,
    ):
        """
        Create a solver instance from a .tar.gz package.

        This method extracts the package to a cache directory (`.extracted_models_cache`)
        and loads the model from the extracted contents.

        Args:
            package_path: Path to the .tar.gz package file.
            device: Device to run the model on.
            max_steps: Maximum number of steps allowed.
            action_masking: Whether to use action masking.
            force_extract: If True, re-extracts the package even if the cache exists.
            checkpoint_filename: Name of the checkpoint file to load (default: 'policy_module_final.pt').
            random_nth_step: If set, every nth step will be a random action.

        Returns:
            GnnMatrixSolver: An initialized solver instance.
        """
        package_path = Path(package_path)
        if not package_path.exists():
            raise FileNotFoundError(f"Package not found: {package_path}")

        # Create a cache directory in the current working directory
        cache_dir = Path(".extracted_models_cache")
        cache_dir.mkdir(exist_ok=True)

        # Create a unique folder name based on package name
        # We use the filename stem
        extract_dir = cache_dir / package_path.name.replace(".tar.gz", "").replace(
            ".tgz", ""
        )

        if force_extract or not extract_dir.exists():
            if extract_dir.exists():
                shutil.rmtree(extract_dir)
            extract_dir.mkdir()

            print(f"Extracting {package_path} to {extract_dir}...")
            with tarfile.open(package_path, "r:gz") as tar:
                tar.extractall(path=extract_dir)

        # Handle nested top-level folder if it exists (common in tar archives)
        # But be careful not to go too deep if there are multiple files
        contents = list(extract_dir.iterdir())
        if len(contents) == 1 and contents[0].is_dir():
            package_root = contents[0]
        else:
            package_root = extract_dir

        manifest = cls._load_manifest(package_root)
        solver = cls._create_from_package_root(
            package_root=package_root,
            manifest=manifest,
            device=device,
            max_steps=max_steps,
            action_masking=action_masking,
            checkpoint_filename=checkpoint_filename,
            random_nth_step=random_nth_step,
        )
        solver.package_root = package_root
        solver.package_manifest = manifest
        return solver

    @classmethod
    def from_policy(
        cls,
        policy_module: ProbabilisticActor,
        cfg: DictConfig | None = None,
        device: str | torch.device | None = None,
        max_steps: int = 50000,
        action_masking: bool = True,
        random_nth_step: int | None = None,
    ):
        """
        Create a solver instance directly from a policy module.

        Args:
            policy_module: The trained policy module (ProbabilisticActor).
            cfg: Configuration object.
            device: Device to run the model on.
            max_steps: Maximum number of steps allowed.
            action_masking: Whether to use action masking.
            random_nth_step: If set, every nth step will be a random action.

        Returns:
            GnnMatrixSolver: An initialized solver instance.
        """
        solver = cls(
            model_path=None,
            cfg=cfg,
            device=device,
            max_steps=50000,
            action_masking=action_masking,
            random_nth_step=random_nth_step,
        )
        solver.policy_module = policy_module
        if solver.device:
            solver.policy_module.to(solver.device)
        solver.policy_module.eval()
        return solver

    @classmethod
    def _create_from_package_root(
        cls,
        package_root: Path,
        manifest: dict | None,
        *,
        device: str | torch.device | None = None,
        max_steps: int = 50000,
        action_masking: bool = True,
        checkpoint_filename: str = "policy_module_final.pt",
        random_nth_step: int | None = None,
    ):
        cfg = cls._load_config_from_package(package_root)
        checkpoint_path = cls._locate_checkpoint(package_root, checkpoint_filename)
        solver = cls(
            model_path=str(checkpoint_path),
            cfg=cfg,
            device=device,
            max_steps=max_steps,
            action_masking=action_masking,
            checkpoint_filename=checkpoint_filename,
            random_nth_step=random_nth_step,
        )
        return solver

    def _load_model(self, env: TransformedEnv):
        """Load the trained policy model from file."""
        if self.policy_module is None:
            if self.model_path is None:
                raise ValueError(
                    "Cannot load model: model_path is None and policy_module is not set."
                )
            policy_module, _ = create_models(env, self.cfg)
            policy_module.load_state_dict(
                torch.load(self.model_path, map_location=self.device)
            )
            policy_module.to(self.device)
            policy_module.eval()

            if self.random_nth_step is not None and self.random_nth_step > 0:
                print(
                    f"Wrapping policy with RandomStepPolicy (n={self.random_nth_step})"
                )
                policy_module = RandomStepPolicy(policy_module, self.random_nth_step)

            self.policy_module = policy_module

    def get_name(self) -> str:
        if self._name is not None:
            return self._name

        if self.model_path is None:
            return "GnnSolver_from_policy"

        from pathlib import Path

        path = Path(self.model_path)
        name = path.name
        if name.endswith(".tar.gz"):
            base_name = name[:-7]
        elif name.endswith(".tgz"):
            base_name = name[:-4]
        elif name in [
            "best_model.pt",
            "policy_module.pt",
            "policy_module_final.pt",
            "model.pt",
        ]:
            # Walk up directories to find a non-generic name
            current = path.parent
            generic_names = {
                "checkpoints",
                "checkpoint",
                "artifacts",
                "artifact",
                "models",
                "model",
                "outputs",
                ".hydra",
            }

            # Try to find a meaningful parent
            for _ in range(4):
                if current.name not in generic_names:
                    base_name = current.name
                    break
                if current.parent == current:  # Root reached
                    break
                current = current.parent
            else:
                base_name = path.parent.name
        else:
            base_name = name

        # Append checkpoint filename to make it unique
        checkpoint_suffix = self.checkpoint_filename.replace(".pt", "")
        return f"{base_name}__{checkpoint_suffix}"

    def get_type(self) -> SolverType:
        return SolverType.ML

    def solve(self, instance):
        """
        Solve a JSSP instance using the trained GNN policy.

        Args:
            instance: JSSPInstance to solve

        Returns:
            Schedule: The solution found by the policy
        """
        env = self._create_not_random_env(instance)
        self._load_model(env)
        solution = self._run_inference(env)
        if not solution.is_complete():
            # Warn if solution is incomplete
            print(
                "Warning: Incomplete solution generated by GNN solver. Increase max_steps?"
            )
        return solution

    def solve_with_info(self, instance):
        env = self._create_not_random_env(instance)
        self._load_model(env)
        solution = self._run_inference_with_info(env=env)

        if not solution.solution.is_complete():
            # Warn if solution is incomplete
            print(
                "Warning: Incomplete solution generated by GNN solver. Increase max_steps?"
            )

        return solution

    def _create_not_random_env(self, instance):
        """Create environment for the given instance."""
        env_kwargs = {
            "max_episode_steps": 50000,
            "random_instance": False,
            "reward_function": "sparse_makespan",
            "reward_kwargs": self.cfg.env.get("reward_kwargs", {}),
            "observation_provider": self.cfg.env.observation_provider,
            "observation_kwargs": self.cfg.env.get("observation_kwargs", {}),
            "job_selector_type": self.cfg.env.get("job_selector_type", "job"),
        }
        env = GraphMatrixEnv(
            instance=instance,
            env_kwargs=env_kwargs,
            device=self.device,
        )
        env = TransformedEnv(
            env,
            Compose(
                StepCounter(max_steps=self.max_steps),
                RewardSum(),
            ),
        )
        return env

    def _run_inference(self, env: TransformedEnv):
        """Run inference using the trained policy in the environment."""
        if isinstance(self.policy_module, RandomStepPolicy):
            self.policy_module.reset()

        with set_exploration_type(ExplorationType.DETERMINISTIC), torch.no_grad():
            rollout = env.rollout(
                self.max_steps,  # Single episode
                self.policy_module,
            )
        solution = env.env.schedule
        return solution

    def _run_inference_with_info(self, env: TransformedEnv):
        """Run inference using the trained policy in the environment."""
        if isinstance(self.policy_module, RandomStepPolicy):
            self.policy_module.reset()

        with set_exploration_type(ExplorationType.DETERMINISTIC), torch.no_grad():
            rollout = env.rollout(
                self.max_steps,  # Single episode
                self.policy_module,
            )
        solution = env.env.schedule
        total_return = rollout["next", "episode_reward"][-1].item()

        logits = rollout.get("logits")
        info = {"total_return": total_return}

        if logits is not None:
            info["logits"] = logits
            info["logits_mean"] = logits.mean().item()
            info["logits_std"] = logits.std().item()

        return SolveOutput(solution=solution, info=info)

    def get_action(self, observation: dict) -> int:
        """
        Get an action for a single observation.

        Args:
            observation: Dictionary containing observation data with keys:
                - 'node_feats': Node features tensor
                - 'edge_index': Edge indices tensor
                - 'mask': Action mask tensor

        Returns:
            int: The selected action index
        """
        if self.policy_module is None:
            raise RuntimeError(
                "Model not loaded. Call solve() first or manually load the model."
            )

        # Convert observation dict to TensorDict format expected by policy
        from tensordict import TensorDict

        # Prepare observation in the expected nested format
        td = TensorDict(
            {
                "observation": {
                    "node_feats": torch.as_tensor(
                        observation["node_feats"], device=self.device
                    ),
                    "edge_index": torch.as_tensor(
                        observation["edge_index"], device=self.device
                    ),
                },
                "mask": torch.as_tensor(observation["mask"], device=self.device),
            },
            batch_size=[],
            device=self.device,
        )

        # Get action from policy (deterministic mode)
        with set_exploration_type(ExplorationType.DETERMINISTIC), torch.no_grad():
            td = self.policy_module(td)

        # Extract and return the action
        action = td["action"].item()
        return action

    @staticmethod
    def _locate_checkpoint(
        package_root: Path, checkpoint_filename: str = "policy_module_final.pt"
    ) -> Path:
        """
        Locate the model checkpoint file (.pt) within the package directory.

        Only looks for the explicitly specified checkpoint file, no fallback logic.

        Args:
            package_root: Root directory of the package
            checkpoint_filename: Exact name of the checkpoint file to load

        Returns:
            Path to the checkpoint file

        Raises:
            FileNotFoundError: If the checkpoint file is not found
        """
        checkpoints_dir = package_root / "artifacts" / "checkpoints"
        if not checkpoints_dir.exists():
            raise FileNotFoundError(
                f"No checkpoints folder found inside package: {package_root}"
            )

        checkpoint_path = checkpoints_dir / checkpoint_filename
        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"Checkpoint file '{checkpoint_filename}' not found in {checkpoints_dir}. "
                f"Please specify the exact checkpoint filename."
            )

        return checkpoint_path

    @staticmethod
    def _load_config_from_package(package_root: Path) -> DictConfig:
        candidates = [
            package_root / "artifacts" / "checkpoints" / "config.yaml",
            package_root / "configs" / "hydra" / "config.yaml",
        ]
        for candidate in candidates:
            if candidate.exists():
                cfg = OmegaConf.load(candidate)
                if not isinstance(cfg, DictConfig):
                    cfg = OmegaConf.create(cfg)
                return cfg
        raise FileNotFoundError(
            f"No config.yaml found in packaged run at {package_root}"
        )
