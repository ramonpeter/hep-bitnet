#!/bin/bash


nvidia-smi

echo "train energy model"
python src/main.py configs/claudius_params_d2_AR.yaml -c

#echo "train shape model"
# adjust energy_model to point to training above
#python src/main.py configs/claudius_d2_shape_model_submission.yaml --use_cuda

RES=$(ls results)
echo "ls results folder: $RES"

#echo "now plot and evaluate model WIP"
#python src/main.py --use_cuda --plot --model_dir results/
