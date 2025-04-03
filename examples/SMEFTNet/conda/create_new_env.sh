#!/bin/bash
#
# Create new GPU environment for weaver
#
# Usage:
#   create_new_env.sh [environment name]


ENV_NAME=${1:-"weaver"}
# CMD="conda"
CMD="mamba"
CHANNELS="-c pytorch -c nvidia -c conda-forge -c pyg -c ostrokach-forge"


# allow conda in shell
eval $(conda shell.bash hook)

set -ex

# install root first to avoid conda.link error
$CMD create -y -n "$ENV_NAME" $CHANNELS python=3.10 root=6.28.0 
$CMD install -y -n "$ENV_NAME" $CHANNELS --file environment.txt

# save environment
conda list -n "$ENV_NAME" --export > environment-list.txt

conda activate "$ENV_NAME"
if [ $? -ne 0 ]; then
    echo "cannot activate environment $ENV_NAME" >&2
    exit 1
else
    # these packages are not available in conda-forge
    pip install -r requirements.txt
fi
conda install pytorch-cluster -c pyg
