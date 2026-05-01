"""Notion property names and option values for the tasks database.

Values are case-sensitive — they must match the database in Notion exactly.
The agent only reads/writes the subset of properties listed here; everything
else (formulas, rollups, system-managed) is left untouched.
"""

from enum import StrEnum


class Status(StrEnum):
    TO_DO = "To Do"
    DOING = "Doing"
    DONE = "Done"


class Priority(StrEnum):
    LOW = "Low"
    MEDIUM = "Medium"
    HIGH = "High"


class TaskProperty:
    NAME = "Name"
    STATUS = "Status"
    PRIORITY = "Priority"
    DUE = "Due"
    PROJECT = "Project"
    DESCRIPTION = "Description"
    LABELS = "Labels"
    MY_DAY = "My Day"
