#!/bin/bash


nvidia-smi

#python src/main.py params/ds2_splitregularnet.yaml -c
python src/main.py params/ds2_splitregularnet.yaml -c --generate -d /users/claudius.krause/HEP-BitNet/hep-bitnet/examples/results-CaloINN/ds2_splitregularnet_158b -its _last --nsamples 100100
