#!/bin/bash


nvidia-smi

#python src/main.py params/ds2_regularnet.yaml -c
python src/main.py params/ds2_regularnet.yaml -c --generate -d /users/claudius.krause/HEP-BitNet/hep-bitnet/examples/results-CaloINN/ds2_regular -its _last --nsamples 100100
