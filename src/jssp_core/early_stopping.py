"""Early stopping base class and implementations for training."""

from abc import ABC, abstractmethod


class EarlyStoppingBase(ABC):
    """Base class for early stopping strategies.

    This class provides a framework for implementing early stopping during training.
    Child classes must implement state initialization and update logic.

    Args:
        patience: Number of epochs to wait before stopping if no improvement
        mode: Either 'min' or 'max' - whether lower or higher values are better
        target: Target threshold for stopping. Can be:
            - float: Static threshold value
            - str: Dynamic threshold key to look up in state
            - None: No threshold, only use patience
        state: Dictionary to track early stopping state. Must contain 'streak' key.
        gap: Minimum improvement gap to consider as improvement (default: 0.0)
    """

    def __init__(
        self,
        patience: int,
        mode: str,
        target: float | str | None,
        gap: float = 0.0,
    ) -> None:
        """Initialize early stopping base class."""
        if mode not in ("min", "max"):
            raise ValueError(f"mode must be 'min' or 'max', got {mode}")

        if patience < 1:
            raise ValueError(f"patience must be >= 1, got {patience}")

        self.patience = patience
        self.mode = mode
        self.target = target
        self.state = {}
        self.gap = gap

        # Initialize state to 0 streak
        self.state["streak"] = 0

        # Allow child classes to initialize additional state
        self._initialize_state()

    @abstractmethod
    def _initialize_state(self) -> None:
        """Initialize state variables specific to the child class.

        This method is called during __init__ and should set up any additional
        state variables needed by the specific early stopping implementation.
        """
        pass

    @abstractmethod
    def _update_state(self, current_value: float) -> None:
        """Update state variables based on the current value.

        This method is called during check_stop and should update any state
        variables (except streak, which is handled by the base class) based
        on the current metric value.

        Args:
            current_value: The current metric value to evaluate
        """
        pass

    def check_stop(self, current_value: float) -> bool:
        """Check if training should stop based on current value.

        Args:
            current_value: Current metric value to evaluate

        Returns:
            True if training should stop, False otherwise
        """

        # Check target threshold if specified
        self.improvement = self._check_target_threshold(current_value)
        self._update_state(current_value)
        # Check patience-based stopping

        return self.state["streak"] >= self.patience

    def _check_target_threshold(self, current_value: float) -> bool:
        """Check if target threshold has been reached.

        Args:
            current_value: Current metric value

        Returns:
            True if target threshold reached, False otherwise
        """
        if self.target is None:
            return False

        # Get threshold value
        if isinstance(self.target, str):
            # Dynamic threshold from state
            threshold = self.state.get(self.target)
            if threshold is None:
                return False
        else:
            # Static threshold
            threshold = self.target

        # Check if threshold met based on mode
        if self.mode == "min":
            return current_value <= threshold - self.gap
        elif self.mode == "max":  # mode == "max"
            return current_value >= threshold + self.gap
        else:
            raise ValueError(f"Unknown mode '{self.mode}', expected 'min' or 'max'.")

    def reset(self) -> None:
        """Reset the early stopping state."""
        self.state["streak"] = 0
        self._initialize_state()


class NoEarlyStopping(EarlyStoppingBase):
    """Early stopping implementation that never stops training.

    This class is used when no early stopping is desired. It always returns
    False from check_stop, allowing training to continue indefinitely.
    """

    def __init__(self) -> None:
        """Initialize NoEarlyStopping with dummy values."""
        # Use dummy values since they won't be used
        super().__init__(
            patience=1,
            mode="min",
            target=None,
            gap=0.0,
        )

    def _initialize_state(self) -> None:
        """No additional state needed for NoEarlyStopping."""
        pass

    def _update_state(self, current_value: float) -> None:
        """No state updates needed for NoEarlyStopping.

        Args:
            current_value: The current metric value (ignored)
        """
        pass

    def check_stop(self, current_value: float) -> bool:
        """Always return False to never stop training.

        Args:
            current_value: The current metric value (ignored)

        Returns:
            Always False
        """
        return False


# NOT EVALUATED YET
class StaticEarlyStopping(EarlyStoppingBase):
    """Early stopping based on a static target threshold.

    Stops training when the metric meets the target threshold (with optional gap)
    for a specified number of consecutive evaluations (patience).

    Example:
        >>> # Stop when loss <= 0.1 for 5 consecutive evaluations
        >>> stopper = StaticEarlyStopping(patience=5, mode="min", target=0.1)
        >>> stopper.check_stop(0.15)  # False, above target
        >>> stopper.check_stop(0.09)  # False, streak=1/5
        >>> # ... after 5 consecutive hits
        >>> stopper.check_stop(0.08)  # True, streak=5/5
    """

    def __init__(
        self,
        patience: int,
        mode: str,
        target: float,
        gap: float = 0.0,
    ) -> None:
        """Initialize static early stopping.

        Args:
            patience: Number of consecutive hits required to stop
            mode: Either 'min' or 'max' - optimization direction
            target: Static threshold value to compare against
            gap: Additional margin required beyond target (default: 0.0)
                For mode='min': must reach target - gap
                For mode='max': must reach target + gap
        """
        super().__init__(
            patience=patience,
            mode=mode,
            target=target,
            gap=gap,
        )

    def _initialize_state(self) -> None:
        """No additional state needed beyond streak."""
        pass

    def _update_state(self, current_value: float) -> None:
        """Update streak based on whether target was hit.

        Args:
            current_value: The current metric value
        """
        # self.improvement is set by base class _check_target_threshold
        if self.improvement:
            self.state["streak"] = self.state.get("streak", 0) + 1
        else:
            self.state["streak"] = 0

        print(
            f"Early stopping streak: {self.state['streak']} / {self.patience} "
            f"(hit: {self.improvement})"
        )


# NOT EVALUATED YET
class LastValueEarlyStopping(EarlyStoppingBase):
    """Early stopping based on lack of improvement from previous value.

    Stops training when the metric fails to improve from the previous value
    for a specified number of consecutive evaluations (patience). This detects
    when training has stagnated.

    Example:
        >>> # Stop when no improvement for 5 consecutive evaluations
        >>> stopper = DynamicEarlyStopping(patience=5, mode="min", gap=0.01)
        >>> stopper.check_stop(0.5)  # False, first value
        >>> stopper.check_stop(0.4)  # False, improved, streak=0
        >>> stopper.check_stop(0.41)  # False, no improvement, streak=1
        >>> stopper.check_stop(0.42)  # False, no improvement, streak=2
        >>> # ... after 5 consecutive non-improvements
        >>> stopper.check_stop(0.43)  # True, streak=5/5
    """

    def __init__(
        self,
        patience: int,
        mode: str,
        gap: float = 0.0,
    ) -> None:
        """Initialize dynamic early stopping.

        Args:
            patience: Number of consecutive non-improvements before stopping
            mode: Either 'min' or 'max' - optimization direction
            gap: Minimum improvement required to count as improvement (default: 0.0)
                For mode='min': must improve by at least gap (value <= prev - gap)
                For mode='max': must improve by at least gap (value >= prev + gap)
        """
        # Use "prev_value" as dynamic target key
        super().__init__(
            patience=patience,
            mode=mode,
            target="prev_value",
            gap=gap,
        )

    def _initialize_state(self) -> None:
        """Initialize prev_value to None for first comparison."""
        self.state["prev_value"] = None

    def _update_state(self, current_value: float) -> None:
        """Update state based on improvement from previous value.

        Args:
            current_value: The current metric value
        """
        # Get previous value before updating it
        prev_value = self.state.get("prev_value")

        # Store current value for next iteration
        self.state["prev_value"] = current_value

        # If no previous value (first call), don't update streak
        if prev_value is None:
            self.state["streak"] = 0
            improved = False
        else:
            # self.improvement is set by base class _check_target_threshold
            # It's True if current value is better than previous value
            improved = self.improvement

            # Increment streak if NOT improved, reset if improved
            if not improved:
                self.state["streak"] = self.state.get("streak", 0) + 1
            else:
                self.state["streak"] = 0

        print(
            f"Early stopping streak: {self.state['streak']} / {self.patience} "
            f"(improved: {improved})"
        )


class IntervalEarlyStopping(EarlyStoppingBase):
    """Early stopping based on improvement over best value ever seen.

    Stops training when the metric fails to improve over the best value
    for a specified number of consecutive evaluations (patience). This is
    the classic early stopping pattern used in most ML frameworks.

    Example:
        >>> # Stop when no improvement over best for 5 consecutive evaluations
        >>> stopper = IntervalEarlyStopping(patience=5, mode="min", gap=0.01)
        >>> stopper.check_stop(0.5)  # False, first value (best=0.5), streak=0
        >>> stopper.check_stop(0.4)  # False, improved (best=0.4), streak=0
        >>> stopper.check_stop(0.45)  # False, no improvement, streak=1
        >>> stopper.check_stop(0.44)  # False, no improvement, streak=2
        >>> stopper.check_stop(0.35)  # False, improved (best=0.35), streak=0
        >>> stopper.check_stop(0.36)  # False, no improvement, streak=1
        >>> # ... after 5 consecutive non-improvements over best
        >>> stopper.check_stop(0.37)  # True, streak=5/5
    """

    def __init__(
        self,
        patience: int,
        mode: str,
        gap: float = 0.0,
    ) -> None:
        """Initialize interval early stopping.

        Args:
            patience: Number of consecutive non-improvements before stopping
            mode: Either 'min' or 'max' - optimization direction
            gap: Minimum improvement required over best value (default: 0.0)
                For mode='min': must be at least gap better (value <= best - gap)
                For mode='max': must be at least gap better (value >= best + gap)
        """
        # Use "best_value" as dynamic target key
        super().__init__(
            patience=patience,
            mode=mode,
            target="best_value",
            gap=gap,
        )

    def _initialize_state(self) -> None:
        """Initialize best_value to None for first comparison."""
        self.state["best_value"] = None

    def _update_state(self, current_value: float) -> None:
        """Update state based on improvement over best value ever seen.

        Args:
            current_value: The current metric value
        """
        # Get best value before potentially updating it
        best_value = self.state.get("best_value")

        # If no best value (first call), initialize it
        if best_value is None:
            self.state["best_value"] = current_value
            self.state["streak"] = 0
            improved = True  # First value establishes baseline
        else:
            # self.improvement is set by base class _check_target_threshold
            # It's True if current value is better than best value
            improved = self.improvement

            # Update best value and reset streak if improved
            if improved:
                self.state["best_value"] = current_value
                self.state["streak"] = 0
            else:
                # No improvement over best, increment streak
                self.state["streak"] = self.state.get("streak", 0) + 1

        print(
            f"Early stopping streak: {self.state['streak']} / {self.patience} "
            f"(improved: {improved}, best: {self.state['best_value']:.6f})"
        )


def create_early_stopping(config: dict | None = None) -> EarlyStoppingBase:
    """Factory function to create early stopping instances.

    Args:
        config: Configuration dictionary with initialization parameters.
            If None or empty, returns NoEarlyStopping.
            Must contain:
                - name: str - Strategy name ("none", "static", "last_value", "interval", "best_value")
            For StaticEarlyStopping, should also contain:
                - patience: int
                - mode: str ("min" or "max")
                - target: float
                - gap: float (optional, default 0.0)
            For LastValueEarlyStopping, should also contain:
                - patience: int
                - mode: str ("min" or "max")
                - gap: float (optional, default 0.0)
            For IntervalEarlyStopping, should also contain:
                - patience: int
                - mode: str ("min" or "max")
                - gap: float (optional, default 0.0)

    Returns:
        An instance of the requested early stopping class

    Raises:
        ValueError: If the name is not recognized or required config is missing

    Examples:
        >>> # Empty config returns NoEarlyStopping
        >>> stopper = create_early_stopping()
        >>> stopper = create_early_stopping({})
        >>> # Explicit "none" strategy
        >>> stopper = create_early_stopping({"name": "none"})
        >>> # Static threshold
        >>> stopper = create_early_stopping(
        ...     {
        ...         "name": "static",
        ...         "patience": 5,
        ...         "mode": "min",
        ...         "target": 0.1,
        ...         "gap": 0.01,
        ...     }
        ... )
        >>> # Last value comparison
        >>> stopper = create_early_stopping(
        ...     {"name": "last_value", "patience": 10, "mode": "min", "gap": 0.001}
        ... )
        >>> # Best value tracking (interval)
        >>> stopper = create_early_stopping(
        ...     {"name": "interval", "patience": 15, "mode": "min", "gap": 0.01}
        ... )
    """
    # If config is None or empty, return NoEarlyStopping
    if not config:
        return NoEarlyStopping()

    # Extract name from config, default to "none"
    name = config.get("name", "none")
    name_lower = name.lower()

    if name_lower in ("none", "no_early_stopping"):
        return NoEarlyStopping()

    if name_lower == "static":
        # Extract required parameters
        try:
            patience = config["patience"]
            mode = config["mode"]
            target = config["target"]
            gap = config.get("gap", 0.0)
        except KeyError as e:
            raise ValueError(
                f"Missing required config parameter for StaticEarlyStopping: {e}"
            ) from e

        return StaticEarlyStopping(
            patience=patience,
            mode=mode,
            target=target,
            gap=gap,
        )

    if name_lower == "last_value":
        # Extract required parameters
        try:
            patience = config["patience"]
            mode = config["mode"]
            gap = config.get("gap", 0.0)
        except KeyError as e:
            raise ValueError(
                f"Missing required config parameter for LastValueEarlyStopping: {e}"
            ) from e

        return LastValueEarlyStopping(
            patience=patience,
            mode=mode,
            gap=gap,
        )

    if name_lower in ("interval", "best_value"):
        # Extract required parameters
        try:
            patience = config["patience"]
            mode = config["mode"]
            gap = config.get("gap", 0.0)
        except KeyError as e:
            raise ValueError(
                f"Missing required config parameter for IntervalEarlyStopping: {e}"
            ) from e

        return IntervalEarlyStopping(
            patience=patience,
            mode=mode,
            gap=gap,
        )

    raise ValueError(
        f"Unknown early stopping strategy: {name}. "
        f"Supported strategies: 'none', 'no_early_stopping', 'static', 'last_value', 'interval', 'best_value'"
    )


# Example functions demonstrating usage
def example_no_early_stopping() -> None:
    """Example demonstrating NoEarlyStopping usage.

    This function shows that NoEarlyStopping never stops training,
    regardless of the metric values.
    """
    print("\n" + "=" * 60)
    print("Example: NoEarlyStopping")
    print("=" * 60)

    # Create NoEarlyStopping instance using factory
    stopper = create_early_stopping()  # Empty config returns NoEarlyStopping
    print(f"Created: {stopper.__class__.__name__}")

    # Alternative: Create directly
    # stopper = NoEarlyStopping()

    # Test with various metric values
    test_values = [0.5, 0.4, 0.45, 0.3, 0.35, 0.25, 0.26, 0.27]

    print("\nTesting with metric values (simulating loss):")
    for i, value in enumerate(test_values, 1):
        should_stop = stopper.check_stop(value)
        print(f"  Epoch {i}: value={value:.3f}, should_stop={should_stop}")

    print("\n[SUCCESS] NoEarlyStopping never stops training")
    print("=" * 60 + "\n")


def example_interval_early_stopping() -> None:
    """Example demonstrating IntervalEarlyStopping usage.

    This function shows how IntervalEarlyStopping tracks the best value
    and stops when no improvement occurs for consecutive evaluations.
    """
    print("\n" + "=" * 60)
    print("Example: IntervalEarlyStopping")
    print("=" * 60)

    # Configuration
    patience = 3
    mode = "min"  # Lower values are better (e.g., loss)
    gap = 0.01  # Require at least 0.01 improvement

    # Create IntervalEarlyStopping using factory
    config = {"name": "interval", "patience": patience, "mode": mode, "gap": gap}
    stopper = create_early_stopping(config)

    # Alternative: Create directly
    # stopper = IntervalEarlyStopping(patience=patience, mode=mode, gap=gap)

    print(f"Created: {stopper.__class__.__name__}")
    print(f"Config: patience={patience}, mode={mode}, gap={gap}")

    # Test with metric values simulating training progress
    # Scenario: Loss decreases, then plateaus, triggering early stopping
    test_values = [
        0.500,  # Initial value (best=0.500)
        0.450,  # Improved (best=0.450, streak=0)
        0.420,  # Improved (best=0.420, streak=0)
        0.425,  # No improvement (streak=1)
        0.423,  # No improvement (streak=2)
        0.422,  # No improvement (streak=3) → STOP
    ]

    print("\nTesting with metric values (simulating loss):")
    for i, value in enumerate(test_values, 1):
        should_stop = stopper.check_stop(value)
        print(f"  Epoch {i}: value={value:.3f}, should_stop={should_stop}")

        if should_stop:
            print(f"\n[SUCCESS] Early stopping triggered at epoch {i}")
            print(f"  Best value: {stopper.state['best_value']:.3f}")
            print(f"  No improvement for {patience} consecutive epochs")
            break
    else:
        print(
            f"\n[INFO] Early stopping not triggered (streak: {stopper.state['streak']})"
        )

    print("=" * 60 + "\n")

    # Additional example: Training with improvement resuming
    print("\n" + "=" * 60)
    print("Example: IntervalEarlyStopping with resumed improvement")
    print("=" * 60)

    # Reset and create new stopper
    config2 = {"name": "interval", "patience": 5, "mode": "min", "gap": 0.005}
    stopper2 = create_early_stopping(config2)
    print("Config: patience=5, mode=min, gap=0.005")

    # Scenario: Improvement, plateau, then improvement again
    test_values2 = [
        0.500,  # Initial (best=0.500)
        0.450,  # Improved (best=0.450, streak=0)
        0.451,  # No improvement (streak=1)
        0.452,  # No improvement (streak=2)
        0.453,  # No improvement (streak=3)
        0.440,  # Improved! (best=0.440, streak=0) ← Reset streak
        0.441,  # No improvement (streak=1)
        0.442,  # No improvement (streak=2)
    ]

    print("\nTesting with metric values:")
    for i, value in enumerate(test_values2, 1):
        should_stop = stopper2.check_stop(value)
        print(f"  Epoch {i}: value={value:.3f}, should_stop={should_stop}")

        if should_stop:
            print(f"\n[SUCCESS] Early stopping triggered at epoch {i}")
            break
    else:
        print(
            "\n[SUCCESS] Training continues (streak reset after improvement at epoch 6)"
        )
        print(f"  Current best: {stopper2.state['best_value']:.3f}")
        print(f"  Current streak: {stopper2.state['streak']}")

    print("=" * 60 + "\n")


# Run examples if this file is executed directly
if __name__ == "__main__":
    example_no_early_stopping()
    example_interval_early_stopping()
