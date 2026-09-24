# Sets the shell to be used for executing recipes.
set shell := ["bash", "-e", "-u", "-o", "pipefail", "-c"]

# Default command: lists all available commands in this Justfile.
default:
    just --list

# Sets up the project by creating a virtual environment and installing dependencies.
setup:
    echo "Creating virtual environment using uv..."
    uv venv
    echo "Syncing dependencies from pyproject.toml..."
    uv sync --all-groups
    echo "Setup complete. Activate the environment with 'source .venv/bin/activate'"

# Runs the PPO training script.
train:
    echo "Starting PPO training..."
    uv run pirl-train

# Renders a video of a trained agent's performance.
# Usage: just video <run_id>
video run_id:
    echo "Rendering video for run ID: {{run_id}}..."
    uv run pirl-video --runid {{run_id}}

# Runs the unit test suite.
test:
    uv run pytest

# Runs the unit test suite with a coverage report.
test-cov:
    uv run pytest --cov --cov-report=term-missing

# Lints the codebase with ruff.
lint:
    uv run ruff check .

# Lints the codebase with ruff and applies automatic fixes.
lint-fix:
    uv run ruff check --fix .

# Formats the codebase with ruff.
format:
    uv run ruff format .

# Checks that the codebase is formatted, without modifying files.
format-check:
    uv run ruff format --check .

# Runs every check the CI pipeline runs (lint, format check, tests).
check: lint format-check test

# Cleans up the project directory by removing generated files and the virtual environment.
clean:
    echo "Cleaning up generated directories and the .venv..."
    rm -rf ppo_checkpoints
    rm -rf ppo_videos
    rm -rf .pytest_cache
    rm -rf .ruff_cache
    rm -rf .coverage htmlcov
    rm -rf .venv