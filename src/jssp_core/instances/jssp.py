import random

from jssp_core.instances.base import F3X3_INSTANCE, FT06_INSTANCE


class JSSPInstance(list):
    """
    Represents a Job Shop Scheduling Problem (JSSP) instance.

    Inherits from list, where each element is a job (list of operations).
    An operation is a tuple of (machine_id, duration).

    Example:
        instance = JSSPInstance([
            [(0, 3), (1, 2)],  # Job 0: (M0, 3) -> (M1, 2)
            [(1, 1), (0, 4)]   # Job 1: (M1, 1) -> (M0, 4)
        ])
    """

    def __repr__(self):
        """
        Return a string representation of the instance in standard format.

        Returns:
            str: Multi-line string with dimensions and job details.
        """
        lines = []
        for job in self:
            line = " ".join(f"{m} {d}" for m, d in job)
            lines.append(line)
        return (
            f"JSSPInstance(num_jobs={len(self)}, num_machines={self.num_machines()}):\n"
            + "\n".join(lines)
        )

    @classmethod
    def from_file(cls, file_path: str) -> "JSSPInstance":
        """
        Create a JSSPInstance by loading data from a file.

        Args:
            file_path: Path to the instance file in standard format.

        Returns:
            JSSPInstance: The loaded instance.
        """
        return cls(_load_instance(file_path))

    def num_jobs(self) -> int:
        """
        Get the number of jobs in the instance.

        Returns:
            int: Number of jobs.
        """
        return len(self)

    def num_machines(self) -> int:
        """
        Get the number of unique machines used across all jobs.

        Returns:
            int: Total number of machines.
        """
        return len(set(m for job in self for m, _ in job))

    def get_lower_bound(self) -> int:
        """
        Calculate a theoretical lower bound for the makespan.

        The lower bound is the maximum of:
        1. Maximum total processing time of any single job.
        2. Maximum total workload assigned to any single machine.

        Returns:
            int: The calculated lower bound.
        """
        job_durations = [sum(duration for _, duration in job) for job in self]
        machine_workloads = [0] * self.num_machines()
        for job in self:
            for machine, duration in job:
                machine_workloads[machine] += duration
        return max(max(job_durations), max(machine_workloads))

    def get_hash(self) -> str:
        """
        Generate a unique MD5 hash for the instance content.

        Returns:
            str: MD5 hexadecimal hash string.
        """
        import hashlib

        # Flatten jobs to a consistent format
        flat = "|".join(",".join(f"{m}:{d}" for m, d in job) for job in self)
        return hashlib.md5(flat.encode("utf-8")).hexdigest()


def _load_instance(file_path: str) -> JSSPInstance:
    """
    Load instance from file.

    Args:
        file_path: Path to the instance file

    Returns:
        Parsed JSSP instance
    """
    with open(file_path) as f:
        instance_text = f.read()
    return _parse_instance(instance_text)


def save_instance(
    instance: JSSPInstance,
    file_path: str,
    instance_name: str | None = None,
    comment: str | None = None,
) -> None:
    """
    Save JSSP instance to a file in standard format.

    Args:
        instance: JSSP instance to save
        file_path: Path where the instance file will be saved
        instance_name: Optional name for the instance (used in header)
        comment: Optional comment to include in header

    Example output format:
        #++++++++++++++++++++++++++
        # instance ft07
        #++++++++++++++++++++++++++
        # Fisher and Thompson-style 7x7 instance
        7 7
        0  3  1  8  2  5  3  6  4  7  5  4  6  9
        1 10  2  5  3  7  4  6  5  4  6  9  0  8
        ...
    """
    num_jobs = instance.num_jobs()
    num_machines = instance.num_machines()

    with open(file_path, "w") as f:
        # Write header with instance name
        if instance_name or comment:
            f.write("#" + "+" * 27 + "\n")
            if instance_name:
                f.write(f"# instance {instance_name}\n")
            f.write("#" + "+" * 27 + "\n")
            if comment:
                f.write(f"# {comment}\n")

        # Write dimensions
        f.write(f"{num_jobs} {num_machines}\n")

        # Write each job as a line of alternating machine and duration values
        for job in instance:
            job_line = " ".join(
                f"{machine:2d} {duration:2d}" for machine, duration in job
            )
            f.write(job_line + "\n")


def _parse_instance(instance_text: str) -> JSSPInstance:
    """
    Parse JSSP instance from text format.

    Format: Each line represents a job with alternating machine and duration values.
    Comments (lines starting with #) and empty lines are ignored.

    Args:
        instance_text: Text representation of the instance

    Returns:
        Parsed JSSP instance as list of jobs
    """
    lines = instance_text.strip().split("\n")

    # Remove comments and empty lines
    lines = [
        line.strip()
        for line in lines
        if line.strip() and not line.strip().startswith("#")
    ]

    if not lines:
        raise ValueError("No valid lines found in instance")

    # Check if first line contains metadata (different length), skip if so
    if len(lines) > 1:
        first_line_tokens = lines[0].split()
        second_line_tokens = lines[1].split()

        if len(first_line_tokens) != len(second_line_tokens):
            lines = lines[1:]  # Skip first line (likely metadata)

    # Parse each job line
    instance = []
    for line in lines:
        tokens = list(map(int, line.split()))

        if len(tokens) % 2 != 0:
            raise ValueError(f"Invalid line format (odd number of tokens): {line}")

        # Convert to (machine, duration) pairs
        job = [(tokens[i], tokens[i + 1]) for i in range(0, len(tokens), 2)]
        instance.append(job)

    return JSSPInstance(instance)


def generate_random_jssp_instance(
    num_jobs: int,
    num_machines: int,
    min_duration: int = 1,
    max_duration: int = 10,
) -> JSSPInstance:
    """
    Generate a random JSSP instance.

    Args:
        num_jobs: Number of jobs
        num_machines: Number of machines
        min_duration: Minimum operation duration
        max_duration: Maximum operation duration

    Returns:
        Random JSSP instance
    """
    instance = []

    for job_id in range(num_jobs):
        # Create random permutation of machines for this job
        machines = list(range(num_machines))
        random.shuffle(machines)

        job = []
        for machine in machines:
            duration = random.randint(min_duration, max_duration)
            job.append((machine, duration))

        instance.append(job)

    return JSSPInstance(instance)


def validate_instance(instance: JSSPInstance) -> tuple[bool, list[str]]:
    """
    Validate that an instance is well-formed.

    Args:
        instance: JSSP instance to validate

    Returns:
        Tuple of (is_valid, list_of_issues)
    """
    issues = []

    if not instance:
        issues.append("Instance is empty")
        return False, issues

    num_machines = set()

    for job_id, job in enumerate(instance):
        if not job:
            issues.append(f"Job {job_id} is empty")
            continue

        job_machines = set()

        for op_id, (machine, duration) in enumerate(job):
            if not isinstance(machine, int) or machine < 0:
                issues.append(
                    f"Job {job_id}, operation {op_id}: invalid machine {machine}"
                )

            if not isinstance(duration, int) or duration <= 0:
                issues.append(
                    f"Job {job_id}, operation {op_id}: invalid duration {duration}"
                )

            if machine in job_machines:
                issues.append(f"Job {job_id}: machine {machine} appears multiple times")

            job_machines.add(machine)
            num_machines.add(machine)

    # Check if machine IDs are contiguous starting from 0
    if num_machines:
        max_machine = max(num_machines)
        expected_machines = set(range(max_machine + 1))
        if num_machines != expected_machines:
            missing = expected_machines - num_machines
            if missing:
                issues.append(f"Missing machine IDs: {sorted(missing)}")

    return len(issues) == 0, issues


def read_yaml_specification_logistics(logistics_text: str) -> list[tuple]:
    instance_list_str = logistics_text.split("\n")[1:-1]
    instance = []

    for machine in instance_list_str:
        transports = (
            machine.split("|")[1].replace("(", "").replace(")", "").replace(",", "")
        )
        transport_times = tuple(map(int, transports.split()))
        instance.append(transport_times)
    # print(instance)

    return instance


def read_yaml_specification_instance(instance_text: str) -> list[list[tuple[int, int]]]:
    instance_list_str = instance_text.split("\n")[1:-1]
    instance = []

    for job in instance_list_str:
        operations = (
            job.split("|")[1].replace("(", "").replace(")", "")  # .replace(",", "")
        )
        tokens = list(operations.split(" "))
        job = []
        for token in tokens:
            m_d = token.split(",")
            m = int(m_d[0])
            d = int(m_d[1])
            job.append((m, d))

        instance.append(job)

    return instance


def get_instance_info(instance: JSSPInstance) -> dict:
    """
    Get summary information about an instance.

    Args:
        instance: JSSP instance

    Returns:
        Dictionary with instance statistics
    """
    if not instance:
        return {"num_jobs": 0, "num_machines": 0, "num_operations": 0}

    num_jobs = len(instance)
    num_operations = sum(len(job) for job in instance)

    all_machines = set()
    durations = []

    for job in instance:
        for machine, duration in job:
            all_machines.add(machine)
            durations.append(duration)

    num_machines = len(all_machines)

    info = {
        "num_jobs": num_jobs,
        "num_machines": num_machines,
        "num_operations": num_operations,
        "min_duration": min(durations) if durations else 0,
        "max_duration": max(durations) if durations else 0,
        "avg_duration": sum(durations) / len(durations) if durations else 0,
        "machines_used": sorted(all_machines),
    }

    return info


from scipy import stats


def generate_truncated_normal_jssp_instance(
    num_jobs: int,
    num_machines: int,
    min_duration: int = 1,
    max_duration: int = 100,
    interval: int = 10,
    std: float = 5.0,
) -> JSSPInstance:
    # Default mean: 20% from min toward max (favors easier instances)

    dist = _get_truncated_dist(
        max_duration=max_duration,
        min_duration=min_duration,
        interval=interval,
        std=std,
    )
    durations = dist.rvs(size=num_jobs * num_machines).astype(int)
    instance = []

    for job_id in range(num_jobs):
        # Create random permutation of machines for this job
        machines = list(range(num_machines))
        random.shuffle(machines)

        job = []
        for machine in machines:
            duration = int(random.choice(durations))
            job.append((machine, duration))

        instance.append(job)

    return JSSPInstance(instance)


def _get_truncated_dist(
    max_duration: int, min_duration: int, interval: int, std: float = 5.0
):
    # range = max_duration - min_duration
    next_min = min_duration + interval
    # mean = next_min if next_min < max_duration else max_duration
    mean = (next_min + min_duration) / 2 if next_min < max_duration else max_duration
    # Create truncated normal distribution using scipy
    # Standardize the bounds
    a_std = (min_duration - mean) / std
    b_std = (max_duration - mean) / std
    dist = stats.truncnorm(a_std, b_std, loc=mean, scale=std)
    return dist


def generate_norm_truncated_normal_jssp_instance(
    num_jobs: int,
    num_machines: int,
    min_duration: int = 1,
    max_duration: int = 100,
    interval: int = 10,
    std: float = 5.0,
) -> JSSPInstance:
    # Default mean: 20% from min toward max (favors easier instances)

    dist = _get_norm_truncated_dist(
        max_duration=max_duration,
        min_duration=min_duration,
        interval=interval,
        std=std,
    )
    durations = dist.rvs(size=num_jobs * num_machines).astype(int)
    instance = []

    for job_id in range(num_jobs):
        # Create random permutation of machines for this job
        machines = list(range(num_machines))
        random.shuffle(machines)

        job = []
        for machine in machines:
            duration = int(random.choice(durations))
            job.append((machine, duration))

        instance.append(job)

    return JSSPInstance(instance)


def _get_norm_truncated_dist(
    max_duration: int, min_duration: int, interval: int, std: float = 5.0
):
    # range = max_duration - min_duration
    next_min = min_duration + interval
    # mean = next_min if next_min < max_duration else max_duration
    mean = (next_min + min_duration) / 2 if next_min < max_duration else max_duration
    # Create truncated normal distribution using scipy
    # Standardize the bounds
    a_std = (min_duration - mean) / std
    b_std = (
        (next_min - mean) / std
        if next_min < max_duration
        else (max_duration - mean) / std
    )
    dist = stats.truncnorm(a_std, b_std, loc=mean, scale=std)
    return dist


# Convenience functions for standard instances
def get_ft06_instance() -> JSSPInstance:
    """Get the FT06 benchmark instance."""
    return _parse_instance(FT06_INSTANCE)


def get_f3x3_instance() -> JSSPInstance:
    """Get the small 3x3 test instance."""
    return _parse_instance(F3X3_INSTANCE)


if __name__ == "__main__":
    instance = generate_random_jssp_instance(8, 8, 1, 10)
    save_instance(
        instance,
        "ft08.txt",
        instance_name="ft08",
        comment="Fisher and Thompson-style 8x8 instance",
    )
