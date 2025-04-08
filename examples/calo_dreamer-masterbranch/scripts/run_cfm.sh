#!/bin/bash
#PBS -l walltime=40:00:00
#PBS -l nodes=1:ppn=1:gpus=1:a30
#PBS -q a30

#module load cuda/11.4
source /remote/gpu05/palacios/venv/bin/activate
mydev=`cat $PBS_GPUFILE | sed s/.*-gpu// `
export CUDA_VISIBLE_DEVICES=$mydev
cd /remote/gpu05/palacios/calo_dreamer/
#python3 src/main.py /remote/gpu05/palacios/calo_dreamer//configs/d2_shape_model.yaml -c
python3 src/main.py /remote/gpu05/palacios/calo_dreamer/configs/d2_energy_model.yaml -c #-p --model_dir="/remote/gpu05/palacios/calo_dreamer/results/20231214_151426_d2_2000_epoch_shape_model_5e3lr_"