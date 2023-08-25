#!/bin/bash

# Check if conda is in the PATH
if ! command -v conda &> /dev/null; then
    echo "Conda not found in PATH. Please install Conda and make sure it's in the PATH." >&2
    exit 1
fi

# Function to determine the architecture
get_architecture() {
    echo $(uname -m)
}

# Detect the architecture
architecture=$(get_architecture)

echo "Updating conda..."
echo

# update conda
conda update conda
echo

# Check if architecture is Intel
if [ "$architecture" == "x86_64" ]; then
    echo "Detected Intel architecture. Creating Intel conda environment."
    echo
    conda env create -n mario_venv_intelx64 -f ./environments/mario_venv_x64.yaml
    conda activate mario_venv_intel64
    echo
    echo "'mario_venv_intel64' successfully created."
    echo
    echo "You can now activate the environment by running 'conda activate mario_venv_intel64'"
    echo
# Check if architecture is Apple Silicon
elif [ "$architecture" == "arm64" ]; then
    echo "Detected Apple Silicon architecture. Creating Apple Silicon conda environment."
    echo
    conda env create -n mario_venv_arm64 -f ./environments/mario_venv_arm64.yaml
    conda activate mario_venv_arm64
    echo
    echo "'mario_venv_arm64' successfully created and activated."
    echo
    echo "You can now activate the environment by running 'conda activate mario_venv_arm64'"
    echo
else
    echo "Unknown architecture: $architecture"
    echo
fi
