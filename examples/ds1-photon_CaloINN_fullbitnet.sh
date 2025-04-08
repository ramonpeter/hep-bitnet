#!/bin/bash


nvidia-smi

#python src/main.py params/ds1-photon_fullbitnet.yaml -c
python src/main.py params/ds1-photon_fullbitnet.yaml -c --generate -d /users/claudius.krause/HEP-BitNet/hep-bitnet/examples/results-CaloINN/ds1-photons_fullbitnet_158b/ -its _last
