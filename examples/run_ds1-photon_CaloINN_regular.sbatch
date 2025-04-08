#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=24G
#SBATCH --time=0-01:00:00     # train 8+ gen 1
#SBATCH --array=0
#SBATCH --output=CaloINN_regular_%A_%a.out
#SBATCH --error=CaloINN_regular_%A_%a.err
#SBATCH --partition=g
#SBATCH --gres=gpu
#SBATCH --nodelist=clip-g2-[0-3],clip-g3-[0-9],clip-g4-[0-11]

mkdir -p /users/claudius.krause/HEP-BitNet/hep-bitnet/examples/results-CaloINN/"$SLURM_ARRAY_JOB_ID"_regular
mkdir -p /scratch-cbe/users/claudius.krause/hep-bitnet/"$SLURM_ARRAY_JOB_ID"_"$SLURM_JOBID"
cp -r /users/claudius.krause/HEP-BitNet/hep-bitnet/examples/CaloINN-calochallenge-generatebranch/* /scratch-cbe/users/claudius.krause/hep-bitnet/"$SLURM_ARRAY_JOB_ID"_"$SLURM_JOBID"/.
cp -r /users/claudius.krause/HEP-BitNet/hep-bitnet/examples/ds1-photon_CaloINN_regular.sh /scratch-cbe/users/claudius.krause/hep-bitnet/"$SLURM_ARRAY_JOB_ID"_"$SLURM_JOBID"/.

cd /scratch-cbe/users/claudius.krause/hep-bitnet/"$SLURM_ARRAY_JOB_ID"_"$SLURM_JOBID"/
echo "SLURM_JOBID: " $SLURM_JOBID
echo "SLURM_ARRAY_TASK_ID: " $SLURM_ARRAY_TASK_ID
echo "SLURM_ARRAY_JOB_ID: " $SLURM_ARRAY_JOB_ID

srun ./ds1-photon_CaloINN_regular.sh

mv /users/claudius.krause/HEP-BitNet/hep-bitnet/examples/CaloINN_regular_"$SLURM_ARRAY_JOB_ID"_"$SLURM_ARRAY_TASK_ID".* /users/claudius.krause/HEP-BitNet/hep-bitnet/examples/results-CaloINN/"$SLURM_ARRAY_JOB_ID"_regular/.

