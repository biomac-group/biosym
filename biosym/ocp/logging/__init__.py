from biosym.ocp.logging.iteration_logger import IterationLogger

__all__ = [
    "IterationLogger",
    "create_dashboard_app",
    "visualize_convergence",
]

_DASH_ATTRS = {"create_dashboard_app", "visualize_convergence"}


def __getattr__(name):
    # dash_logger.py depends on the optional dash/plotly packages. Importing it
    # only on first access (rather than eagerly here) lets any code that just
    # needs IterationLogger -- e.g. collocation.py, and everything that imports
    # it -- work without those packages installed.
    if name in _DASH_ATTRS:
        from biosym.ocp.logging import dash_logger

        return getattr(dash_logger, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
