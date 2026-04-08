"""
Civil Command Center — Tasks Module
=====================================
Exports task configs for the 3 graded difficulty levels
plus 2 demo modes for judges and quick testing.
"""

from tasks.task_easy import TASK_CONFIG as EASY_CONFIG
from tasks.task_medium import TASK_CONFIG as MEDIUM_CONFIG
from tasks.task_hard import TASK_CONFIG as HARD_CONFIG

ALL_TASKS = {
    EASY_CONFIG["id"]: EASY_CONFIG,
    MEDIUM_CONFIG["id"]: MEDIUM_CONFIG,
    HARD_CONFIG["id"]: HARD_CONFIG,
}

__all__ = ["ALL_TASKS", "EASY_CONFIG", "MEDIUM_CONFIG", "HARD_CONFIG"]
