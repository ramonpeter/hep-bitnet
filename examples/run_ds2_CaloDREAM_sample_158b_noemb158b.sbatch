#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=10G
#SBATCH --time=0-03:00:00     # train: 2d, sample 3h on A100, 90m on g3, 
#SBATCH --array=0
#SBATCH --output=CaloDREAM_158b_noemb158b_%A_%a.out
#SBATCH --error=CaloDREAM_158b_noemb158b_%A_%a.err
#SBATCH --partition=g
#SBATCH --gres=gpu
#S BATCH --qos=medium
#SBATCH --nodelist=clip-g2-[0-3],clip-g3-[0-9],clip-g4-[0-11]

mkdir -p /users/claudius.krause/HEP-BitNet/hep-bitnet/examples/results-CaloDREAM/"$SLURM_ARRAY_JOB_ID"_158b_noemb158b
mkdir -p /scratch-cbe/users/claudius.krause/hep-bitnet/"$SLURM_ARRAY_JOB_ID"_"$SLURM_JOBID"
cp -r /users/claudius.krause/HEP-BitNet/hep-bitnet/examples/results-CaloDREAM/bitnet_158b/trained_energy_model /scratch-cbe/users/claudius.krause/hep-bitnet/"$SLURM_ARRAY_JOB_ID"_"$SLURM_JOBID"/.
cp -r /users/claudius.krause/HEP-BitNet/hep-bitnet/examples/results-CaloDREAM/bitnet_158b/trained_shape_model_noemb /scratch-cbe/users/claudius.krause/hep-bitnet/"$SLURM_ARRAY_JOB_ID"_"$SLURM_JOBID"/trained_shape_model

cp -r /users/claudius.krause/HEP-BitNet/hep-bitnet/examples/calo_dreamer-submissionbranch/* /scratch-cbe/users/claudius.krause/hep-bitnet/"$SLURM_ARRAY_JOB_ID"_"$SLURM_JOBID"/.
cp -r /users/claudius.krause/HEP-BitNet/hep-bitnet/examples/ds2_CaloDREAM_regular_sample.sh /scratch-cbe/users/claudius.krause/hep-bitnet/"$SLURM_ARRAY_JOB_ID"_"$SLURM_JOBID"/.

cd /scratch-cbe/users/claudius.krause/hep-bitnet/"$SLURM_ARRAY_JOB_ID"_"$SLURM_JOBID"/
echo "SLURM_JOBID: " $SLURM_JOBID
echo "SLURM_ARRAY_TASK_ID: " $SLURM_ARRAY_TASK_ID
echo "SLURM_ARRAY_JOB_ID: " $SLURM_ARRAY_JOB_ID

srun ./ds2_CaloDREAM_regular_sample.sh

mv /users/claudius.krause/HEP-BitNet/hep-bitnet/examples/CaloDREAM_158b_noemb158b_"$SLURM_ARRAY_JOB_ID"_"$SLURM_ARRAY_TASK_ID".* /users/claudius.krause/HEP-BitNet/hep-bitnet/examples/results-CaloDREAM/"$SLURM_ARRAY_JOB_ID"_158b_noemb158b/.

