#!/bin/bash


nvidia-smi

#python src/main.py params/ds1-pion_splitregularnet.yaml -c
python src/main.py params/ds1-pion_splitregularnet.yaml -c --generate -d /users/claudius.krause/HEP-BitNet/hep-bitnet/examples/results-CaloINN/ds1-pions_splitregularnet_158b/ -its _last
