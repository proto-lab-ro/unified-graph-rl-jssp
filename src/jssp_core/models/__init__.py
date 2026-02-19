"""
JSSP Core Models

This module contains data models for representing Job Shop Scheduling Problems
and their solutions.

Classes:
    OperationInfo: Represents a single operation in a JSSP
    ScheduledOperation: Represents an operation with timing information
"""

from jssp_core.models.operation import OperationInfo, ScheduledOperation


__all__ = [
    "OperationInfo",
    "ScheduledOperation",
]
