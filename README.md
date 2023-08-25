# SuperMario-AI-Strategies
CITS3001 Agents, Algorithms and Artificial Intelligence Group Project

## Project Overview
In this project, we will develop AI agents to control the iconic character Mario in the classic game Super Mario Bros using the gym-super-mario-bros environment. The main objective is to implement at two distinct AI algorithms/methods (Reinforcement Learning: Q-learning and Monte Carlo Tree Search (MCTS)) and compare their performance, strengths, and weaknesses in the context of playing the game.

## Members
Edison Foo (23121022) Charles So (23199336)

## Conda Set up
Conda is an open source package management system and environment management system that runs on Windows, macOS, and Linux. Conda quickly installs, runs and updates packages and their dependencies.

Note: Conda is often distributed with the Anaconda distribution, which includes additional packages and tools. However, for a lightweight installation, you can use Miniconda, which only includes Conda itself and a few essential packages.

#### Download Miniconda Installer: https://docs.conda.io/en/latest/miniconda.html
### Method one (`install.sh`):
This script will automatically detect, create and activate conda environment based on your system architecture.
```bash
   # give permission to execute
   chmod +x ./environments/install.sh
   # execute the script
   ./environments/install.sh
```
### Method two (set up manually):
#### Step 1: Check conda is installed and in your PATH
```bash
    conda --version
```
#### Step 2: Check conda is up to date
```bash
    conda update conda
```
#### Step 3: Create a virtual environment for the project
```bash
    # set up for intel
    conda env create -n mario_venv_intelx64 -f ./environments/mario_venv_x64.yaml
    # set up for apple-sillicon
    conda env create -n mario_venv_arm64 -f ./environments/mario_venv_arm64.yaml
```
#### Step 4: Activate your virtual environment
```bash
    # intel
    conda activate mario_venv_intel64
    # apple-sillicon
    conda activate mario_venv_arm64
```
### Basic Commands
install packages
```bash
    conda install [package]
    # or use pip if package doesn't exit in conda repository
    pip install [package]
```

export environment
```bash
    # intel
    conda env export --no-builds > ./environments/mario_venv_x64.yaml
    # apple-sillicon
    conda env export --no-builds > ./environments/mario_venv_arm64.yaml
```
remove the conda environment
```bash
    conda remove -n mario --all
```
