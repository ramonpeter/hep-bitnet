#!/bin/bash


nvidia-smi

echo "sample from trained model"
python sample.py --model trained_shape_model/ --energy_model trained_energy_model/ --batch_size 1000 

RES=$(ls results)
echo "ls results folder: $RES"

