#!/bin/bash
# --- OneFileLLM Launcher (Linux/macOS) ---

# Get the directory where the script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

# IMPORTANT: Configure this path if your setup differs!
# Option 1: If venv is in the project directory (same as script_dir):
VENV_PATH="$SCRIPT_DIR/.venv"
# Option 2: If venv is elsewhere (example):
# VENV_PATH="/path/to/your/venvs/onefilellm_env"

# Path to the onefilellm.py script
SCRIPT_FILE="$SCRIPT_DIR/onefilellm.py"

# --- Activation and Execution ---
echo "Activating virtual environment..."
if [ -f "$VENV_PATH/bin/activate" ]; then
    source "$VENV_PATH/bin/activate"
else
    echo "[WARNING] Virtual environment activation script not found at '$VENV_PATH/bin/activate'."
    echo "Attempting to run directly. Ensure Python and dependencies are in your system PATH or environment."
fi

echo "Running OneFileLLM..."
python "$SCRIPT_FILE" "$@"

echo ""
echo "OneFileLLM finished."

# Deactivate if the 'deactivate' function exists (common in venvs)
if command -v deactivate &> /dev/null
then
    # echo "Deactivating virtual environment..." # Optional message
    deactivate
fi