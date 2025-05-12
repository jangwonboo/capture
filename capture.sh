#!/bin/bash
# Wrapper script to run the e-book capture tool with Python 3.11

# Activate the conda environment
eval "$(conda shell.bash hook)"
conda activate py311

# Run the application with all arguments passed to this script
python run.py "$@"

# Deactivate the environment when done
conda deactivate 