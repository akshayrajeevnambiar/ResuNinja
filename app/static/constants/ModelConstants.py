"""
TextProcessingConstants.py

- This module defines constants used throughout the project.

Usage:
- Import this module into your Python script or program to access the constants.
"""


class ModelConstants:

    # Overrides the object method to restrict value modification.
    def __setattr__(self, name, value):
        raise AttributeError("Write Error: Cannot re-write constants values ")

    # Array group number
    TEXT_GROUP = 1

    # Common word count
    COMMON_WORD_COUNT = 50
