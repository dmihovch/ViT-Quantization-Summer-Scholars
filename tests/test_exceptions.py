"""Tests for the custom exception hierarchy defined in :mod:`src.exceptions`."""

from __future__ import annotations

from src.exceptions import DataDirectoryError, HookRegistrationError, ShapeMismatchError


def test_data_directory_error_is_file_not_found_error() -> None:
    """DataDirectoryError must be catchable as FileNotFoundError."""
    assert issubclass(DataDirectoryError, FileNotFoundError)


def test_hook_registration_error_is_runtime_error() -> None:
    """HookRegistrationError must be catchable as RuntimeError."""
    assert issubclass(HookRegistrationError, RuntimeError)


def test_shape_mismatch_error_is_value_error() -> None:
    """ShapeMismatchError must be catchable as ValueError."""
    assert issubclass(ShapeMismatchError, ValueError)
