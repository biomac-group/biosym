# Agentic Coding Guidelines

- **Environment Management**: Strictly use `uv` for python environments, dependencies, and execution.
- **Running Tests**: Tests are executed via `uv run pytest`. Tests should only be run required by test-driven development or when verifying a change for release. Do not run tests unless asked to or if the agent persona requires it.
- **Running Examples**: Example scripts are executed via `uv run python examples/gait2d.py`. Do not run examples unless specifically asked to.

- **Backward Compability**: We are in an early developing phase without users. Backward compability is nice to have, but not a priority. The following behaviour is outdated and should never be cared for:
    - ``utils.states.StatesDict`` dataclass should never be used. ``utils.states.States`` should be used instead. ``utils.states.States`` does not have ``model``, ``dq`` nor ``constants`` attributes. It uses ``qd`` instead of ``dq`` and ``constants`` instead of ``model``. ``utils.states.h`` is in globals. Do not add functions for convinience backward compatiblity.

- **Agent Personas / Profiles**: Refer to and follow the specialized guidelines for agent roles documented under [agent_profiles](file:///Users/markusgambietz/PhD/01_Python_Projects/biosym/skills/agent_profiles) (e.g., `release_gatekeeper.md` for release and documentation quality, `lean_skeptic.md` for backward compatibility checks, etc.) when performing relevant tasks.

- **Changelog**: When making changes to the codebase, update the changelog accordingly in the section for the next version bump. The changelog is located in "CHANGELOG.md". Ensure that the changelog entry is clear, concise, and accurately reflects the changes made.
    