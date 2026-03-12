"""Basic import tests for jetauto_dashboard."""

import pytest


def test_import_dashboard_node():
    """Verify the dashboard module is importable (minus rclpy)."""
    # We can't import the full node without rclpy, but we can check
    # the file exists and has no syntax errors
    import py_compile
    import os
    node_path = os.path.join(
        os.path.dirname(__file__), '..', 'jetauto_dashboard', 'dashboard_node.py')
    py_compile.compile(node_path, doraise=True)
