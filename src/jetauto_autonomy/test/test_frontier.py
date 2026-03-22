"""Basic tests for frontier explorer logic."""
import pytest


def test_import():
    """Verify the module imports without error."""
    from jetauto_autonomy.frontier_explorer import FrontierExplorer  # noqa: F401


def test_safety_import():
    """Verify safety monitor imports without error."""
    from jetauto_autonomy.safety_monitor import SafetyMonitor  # noqa: F401
