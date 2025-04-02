#!/bin/bash
#
# Setup GPU based environment for weaver from frozen environment
#
# Usage:
#   create_env.sh [environment name]
#

ENV_NAME=${1:-"weaver"}
CMD="conda"
# CMD="mamba"
CHANNELS="-c pytorch -c nvidia -c conda-forge"

eval $(conda shell.bash hook)

set -ex

# install root first to avoid conda.link error
$CMD create  -y -n "$ENV_NAME" $CHANNELS python=3.10 root=6.28.0
$CMD install -y -n "$ENV_NAME" $CHANNELS --file environment-list.txt

conda activate "$ENV_NAME"
pip install -r requirements.txt
