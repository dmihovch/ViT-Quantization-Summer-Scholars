"""Tests for the custom exception hierarchy defined in :mod:`src.exceptions`."""

from __future__ import annotations

from src.exceptions import DataDirectoryError


def test_data_directory_error_is_file_not_found_error() -> None:
    """DataDirectoryError must be catchable as FileNotFoundError."""
    assert issubclass(DataDirectoryError, FileNotFoundError)