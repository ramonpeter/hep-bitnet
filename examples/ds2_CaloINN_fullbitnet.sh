#!/bin/bash


nvidia-smi

#python src/main.py params/ds2_fullbitnet.yaml -c
python src/main.py params/ds2_fullbitnet.yaml -c --generate -d /users/claudius.krause/HEP-BitNet/hep-bitnet/examples/results-CaloINN/ds2_fullbitnet_158b -its _last --nsamples 105000
