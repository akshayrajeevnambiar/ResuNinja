"""
textProcessingConstants.py

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

    # Test size split percent
    TEST_SIZE = 0.2

    # Random state value
    RANDOM_STATE = 42

    # Max number of iteration for the model
    MAX_ITERS = 500

    # Number of estimators
    NUMBER_OF_ESTIMATORS = 100
