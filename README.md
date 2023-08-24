# SuperMario-AI-Strategies
CITS3001 Agents, Algorithms and Artificial Intelligence Group Project

## Project Overview
In this project, we will develop AI agents to control the iconic character Mario in the classic game Super Mario Bros using the gym-super-mario-bros environment. The main objective is to implement at two distinct AI algorithms/methods (Reinforcement Learning: Q-learning and Monte Carlo Tree Search (MCTS)) and compare their performance, strengths, and weaknesses in the context of playing the game.

## Members
Edison Foo (23121022) Charles So (23199336)

## Conda Set up
Conda is an open source package management system and environment management system that runs on Windows, macOS, and Linux. Conda quickly installs, runs and updates packages and their dependencies.

Note: Conda is often distributed with the Anaconda distribution, which includes additional packages and tools. However, for a lightweight installation, you can use Miniconda, which only includes Conda itself and a few essential packages.

### Installation
#### Step 1: Download Miniconda Installer: https://docs.conda.io/en/latest/miniconda.html

#### Step 2: Check conda is installed and in your PATH
```bash
    conda --version
```
#### Step 3: Check conda is up to date
```bash
    conda update conda
```
#### Step 4: Create a virtual environment for the project
```bash
    # set up for intel
    conda env create -n mario_venv -f mario_venv_x64.yaml
    # set up for apple-sillicon
    conda env create -n mario_venv -f mario_venv_arm64.yaml
```
#### Step 5: Activate your virtual environment
```bash
    conda activate mario_venv
```
### Basic Commands
To install packages
```bash
    conda install [package]
    # or use pip if package doesn't exit in conda repository
    pip install [package]
```

To export environment
```bash
    # for intel
    conda env export > mario_venv_x64.yaml
    # for apple-sillicon
    conda env export > mario_venv_arm64.yaml
```
To remove the conda environment
```bash
    conda remove -n mario --all
```
