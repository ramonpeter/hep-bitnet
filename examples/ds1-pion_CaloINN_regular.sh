#!/bin/bash


nvidia-smi

#python src/main.py params/ds1-pion_regularnet.yaml -c
python src/main.py params/ds1-pion_regularnet.yaml -c --generate -d /users/claudius.krause/HEP-BitNet/hep-bitnet/examples/results-CaloINN/ds1-pions_regular/ -its 450
