#!/bin/bash


nvidia-smi

#python src/main.py params/ds1-pion_bitnet.yaml -c
python src/main.py params/ds1-pion_bitnet.yaml -c --generate -d /users/claudius.krause/HEP-BitNet/hep-bitnet/examples/results-CaloINN/ds1-pions_bitnet_158b/ -its _last
