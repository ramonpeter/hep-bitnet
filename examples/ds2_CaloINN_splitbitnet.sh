#!/bin/bash


nvidia-smi

#python src/main.py params/ds2_splitbitnet.yaml -c
python src/main.py params/ds2_splitbitnet.yaml -c --generate -d /users/claudius.krause/HEP-BitNet/hep-bitnet/examples/results-CaloINN/ds2_splitbitnet_158b -its _last --nsamples 100100
