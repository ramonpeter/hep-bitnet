#!/usr/bin/env bash


#SBATCH --job-name=SM
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:V100:1
#SBATCH --partition=g
#SBATCH --time=2-00:00:00
#SBATCH --qos=medium
#SBATCH --output=result.stdout

ml build-env/f2022
ml root/6.26.10-foss-2022b
ml cuda/12.0.0

export TF_ENABLE_ONEDNN_OPTS=0

python3 /users/daohan.wang/SMEFTNet/train_regression.py --overwrite --epochs 300 --prefix eft_wz_regressed   --config regress_wz_v0 --dRN 0.4 --conv_params "( (0.0, [20, 20]), (0.0, [20, 20]) )" --readout_params "(0.0, [32, 32])"

