#!/bin/bash
# VSCode debugger launcher — loads the required module before invoking Python.
# Configured as the "python" executable in .vscode/launch.json.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
VENV_PYTHON="$PROJECT_DIR/.ditunet_venv/bin/python3"

# Initialize lmod if not already done
if ! command -v module &>/dev/null; then
    source /usr/share/lmod/lmod/init/bash
fi

# Load the Python module
module load Python/3.11.3-GCCcore-12.3.0

# Run the venv python with all arguments passed by VSCode/debugpy
exec "$VENV_PYTHON" "$@"
