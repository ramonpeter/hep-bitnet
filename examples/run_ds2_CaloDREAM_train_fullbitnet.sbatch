#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=25G
#SBATCH --time=0-02:30:00     # train energy model: 2h
#SBATCH --array=0
#SBATCH --output=CaloDREAM_fullbitnet_%A_%a.out
#SBATCH --error=CaloDREAM_fullbitnet_%A_%a.err
#SBATCH --partition=g
#SBATCH --gres=gpu
#SB ATCH --qos=medium
#SBATCH --nodelist=clip-g2-[0-3],clip-g3-[0-9],clip-g4-[0-11]

mkdir -p /users/claudius.krause/HEP-BitNet/hep-bitnet/examples/results-CaloDREAM/"$SLURM_ARRAY_JOB_ID"_fullbitnet
mkdir -p /scratch-cbe/users/claudius.krause/hep-bitnet/"$SLURM_ARRAY_JOB_ID"_"$SLURM_JOBID"
cp -r /users/claudius.krause/HEP-BitNet/hep-bitnet/examples/calo_dreamer-masterbranch/* /scratch-cbe/users/claudius.krause/hep-bitnet/"$SLURM_ARRAY_JOB_ID"_"$SLURM_JOBID"/.
cp -r /users/claudius.krause/HEP-BitNet/hep-bitnet/examples/calo_dreamer_params/params_d2_AR_fullbitnet.yaml /scratch-cbe/users/claudius.krause/hep-bitnet/"$SLURM_ARRAY_JOB_ID"_"$SLURM_JOBID"/configs/claudius_params_d2_AR.yaml

cp -r /users/claudius.krause/HEP-BitNet/hep-bitnet/examples/ds2_CaloDREAM_regular_train.sh /scratch-cbe/users/claudius.krause/hep-bitnet/"$SLURM_ARRAY_JOB_ID"_"$SLURM_JOBID"/.

cd /scratch-cbe/users/claudius.krause/hep-bitnet/"$SLURM_ARRAY_JOB_ID"_"$SLURM_JOBID"/
echo "SLURM_JOBID: " $SLURM_JOBID
echo "SLURM_ARRAY_TASK_ID: " $SLURM_ARRAY_TASK_ID
echo "SLURM_ARRAY_JOB_ID: " $SLURM_ARRAY_JOB_ID

srun ./ds2_CaloDREAM_regular_train.sh

mv /users/claudius.krause/HEP-BitNet/hep-bitnet/examples/CaloDREAM_fullbitnet_"$SLURM_ARRAY_JOB_ID"_"$SLURM_ARRAY_TASK_ID".* /users/claudius.krause/HEP-BitNet/hep-bitnet/examples/results-CaloDREAM/"$SLURM_ARRAY_JOB_ID"_fullbitnet/.

