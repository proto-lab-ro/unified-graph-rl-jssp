"""
Base Dispatcher Class

This module provides the abstract base class for all dispatcher implementations.
Dispatchers are responsible for setting up and coordinating all components needed
for training multi-agent reinforcement learning models on JSSP with AGV transport.

All concrete dispatchers (MLP, GNN, Advantage-based variants) inherit from this base
and implement the abstract methods according to their specific requirements.
"""

from abc import ABC, abstractmethod
from typing import Any

from omegaconf import DictConfig


class DispatcherBase(ABC):
    """
    Abstract base class for all dispatcher implementations.

    A dispatcher orchestrates the setup of all training components in the correct order:
    1. Problem instance loading
    2. Environment creation
    3. Model creation (policies and critics)
    4. Loss module creation
    5. Optimizer creation
    6. Scheduler creation
    7. Data collector creation
    8. Replay buffer creation

    Attributes:
        cfg: Hydra configuration object
        instance: Problem instance (JSSP with AGV transport)
        env: Training environment
        test_env: Optional test environment for evaluation
        policies: Dictionary of individual agent policies
        combined_policies: Combined policy module for all agents
        value_module: Critic/value function module
        loss_modules: Dictionary of loss modules for each agent
        advantage_module: Optional separate advantage module (for Adv variants)
        shared_extractor: Optional shared feature extractor (for GNN variants)
        optimizers: Dictionary of optimizers for each agent
        schedulers: Dictionary of learning rate schedulers
        collector: Data collector for experience gathering
        replay_buffers: Dictionary of replay buffers for each agent
    """

    def __init__(self, cfg: DictConfig):
        """
        Initialize the dispatcher with configuration.

        Args:
            cfg: Hydra configuration object containing all hyperparameters
        """
        self.cfg = cfg

        # Core components - initialized by setup methods
        self.instance = None
        self.env = None
        self.test_env = None

        # Model components
        self.policies = None
        self.combined_policies = None
        self.value_module = None
        self.shared_extractor = None  # Used by GNN variants

        # Training components
        self.loss_modules = None
        self.advantage_module = None  # Used by Advantage-based variants
        self.optimizers = None
        self.schedulers = None

        # Data collection components
        self.collector = None
        self.replay_buffers = None

    @abstractmethod
    def setup_instance(self):
        """
        Load the problem instance.

        Returns:
            Problem instance object
        """
        pass

    @abstractmethod
    def setup_environment(self):
        """
        Create the training environment.

        Returns:
            Training environment
        """
        pass

    def setup_test_environment(self):
        """
        Create a separate test environment for evaluation.

        Optional method - not all dispatchers need a separate test environment.

        Returns:
            Test environment (or None if not implemented)
        """
        return None

    @abstractmethod
    def setup_models(self) -> tuple[tuple[Any, Any], Any]:
        """
        Create policies and critic models.

        Returns:
            Tuple of ((policies, combined_policies), value_module)
        """
        pass

    @abstractmethod
    def setup_loss_modules(self) -> tuple[Any, Any | None]:
        """
        Create loss modules for training.

        Returns:
            Tuple of (loss_modules, advantage_module)
            - loss_modules: Dictionary of loss modules for each agent
            - advantage_module: Optional advantage module (None for standard GAE)
        """
        pass

    @abstractmethod
    def setup_optimizers(self):
        """
        Create optimizers for each agent.

        Returns:
            Dictionary of optimizers
        """
        pass

    @abstractmethod
    def setup_schedulers(self):
        """
        Create learning rate schedulers.

        Returns:
            Dictionary of schedulers
        """
        pass

    @abstractmethod
    def setup_collector(self):
        """
        Create data collector for experience gathering.

        Returns:
            Data collector
        """
        pass

    @abstractmethod
    def setup_replay_buffer(self):
        """
        Create replay buffers for each agent group.

        Returns:
            Dictionary of replay buffers
        """
        pass

    def setup_all(self) -> dict[str, Any]:
        """
        Setup all components in the correct order.

        This method calls all setup methods in sequence and returns a dictionary
        containing all initialized components. This is the main entry point for
        dispatcher usage.

        Returns:
            Dictionary containing all initialized components:
            - instance: Problem instance
            - env: Training environment
            - test_env: Test environment (if created)
            - policies: Individual agent policies
            - combined_policies: Combined policy module
            - value_module: Critic/value function
            - loss_modules: Loss modules
            - advantage_module: Advantage module (if applicable)
            - shared_extractor: Shared feature extractor (if applicable)
            - optimizers: Optimizers
            - schedulers: Learning rate schedulers
            - collector: Data collector
            - replay_buffers: Replay buffers
        """
        # Setup in correct order
        self.setup_instance()
        self.setup_environment()

        # Optional: setup test environment if needed
        test_env = self.setup_test_environment()
        if test_env is not None:
            self.test_env = test_env

        self.setup_models()
        self.setup_loss_modules()
        self.setup_optimizers()
        self.setup_schedulers()
        self.setup_collector()
        self.setup_replay_buffer()

        # Build result dictionary with all components
        result = {
            "instance": self.instance,
            "env": self.env,
            "policies": self.policies,
            "combined_policies": self.combined_policies,
            "value_module": self.value_module,
            "loss_modules": self.loss_modules,
            "optimizers": self.optimizers,
            "schedulers": self.schedulers,
            "collector": self.collector,
            "replay_buffers": self.replay_buffers,
        }

        # Add optional components if they exist
        if self.test_env is not None:
            result["test_env"] = self.test_env

        if self.advantage_module is not None:
            result["advantage_module"] = self.advantage_module

        if self.shared_extractor is not None:
            result["shared_extractor"] = self.shared_extractor

        return result

    def get_component(self, component_name: str) -> Any:
        """
        Get a specific component by name.

        Args:
            component_name: Name of the component to retrieve

        Returns:
            The requested component

        Raises:
            AttributeError: If component doesn't exist
        """
        if not hasattr(self, component_name):
            raise AttributeError(
                f"Component '{component_name}' not found in dispatcher"
            )
        return getattr(self, component_name)

    def has_component(self, component_name: str) -> bool:
        """
        Check if a component exists and is initialized.

        Args:
            component_name: Name of the component to check

        Returns:
            True if component exists and is not None, False otherwise
        """
        return (
            hasattr(self, component_name) and getattr(self, component_name) is not None
        )

    def __repr__(self) -> str:
        """String representation of the dispatcher."""
        class_name = self.__class__.__name__
        components = []

        for attr in [
            "instance",
            "env",
            "policies",
            "value_module",
            "loss_modules",
            "optimizers",
            "collector",
            "replay_buffers",
        ]:
            if self.has_component(attr):
                components.append(attr)

        return f"{class_name}(initialized_components={components})"
