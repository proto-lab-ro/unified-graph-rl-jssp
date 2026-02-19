from dataclasses import dataclass


@dataclass
class OperationInfo:
    """Represents a single operation in a JSSP"""

    job_id: int
    op_id: int
    machine: int
    duration: int

    def __repr__(self):
        return f"J{self.job_id}O{self.op_id}(M{self.machine},{self.duration})"


@dataclass
class ScheduledOperation:
    """Represents a scheduled operation with timing information"""

    operation: OperationInfo
    start_time: int
    end_time: int

    def __repr__(self):
        return f"{self.operation}[{self.start_time}-{self.end_time}]"
