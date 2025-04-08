####Dirs####
MAINDIR="/remote/gpu06/favaro/calo_dreamer/"
DATADIR="/remote/gpu06/favaro/datasets/calo_challenge/"
OUTDIR="/remote/gpu06/favaro/calo_dreamer/results/"
XMLFILE="${DATADIR}binning_dataset_1_photons.xml"

####Parameters####
RUN_NAME="test_calo_dreamer"
PTYPE="photon"
NOISE=5.0e-6
W_DECAY=0.01
NBLOCKS=1
BLOCK_SZ=256
BLOCK_LY=3
ALPHA=1.0e-8

####Training####
EPS=1
CYCLE=50
SAVE_I=100

####Model####
BATCH_SZ=256

#Code to create yml file#
ARGSDIR="${MAINDIR}/configs/"
YMLFILE="${ARGSDIR}cfm_base.yaml"

mkdir $ARGSDIR

cat << EOF > $YMLFILE
run_name: ${RUN_NAME}
p_type: ${PTYPE}
dtype: float32
# Data
hdf5_file: ${DATADIR}gamma_data_1.hdf5
xml_filename: ${XMLFILE}
width_noise: ${NOISE}
val_frac: 0.2
eps: 1.0e-10
particle_type: "photon"
eval_dataset: "1-photons"
transforms: {
 NormalizeByElayer: {ptype: ${XMLFILE}, xml_file: ${PTYPE}},
 AddNoise: {noise_width: ${NOISE}}
}

# Training
lr: 1.e-5
max_lr: 1.e-4
batch_size: ${BATCH_SZ}
validate_every: 100
use_scheduler: True
lr_scheduler: one_cycle_lr
weight_decay: ${W_DECAY}
betas: [0.9, 0.999]
n_epochs: ${EPS}
cycle_epochs: ${CYCLE}
save_interval: ${SAVE_I}

# Sampling
sample_periodically: True
sample_every: 100
sample_every_n_samples: 100

# ResNet block
intermediate_dim: ${BLOCK_SZ}
n_blocks: ${NBLOCKS}
dim: 373
n_con: 1
layers_per_block: ${BLOCK_LY}
conditional: True

# Additional params
alpha: ${ALPHA}
alpha_logit: 1.0e-6
EOF
chmod +x $YMLFILE

####Build and Sub####

SUB_MODEL="${MAINDIR}/scripts/run_cfm.sh"

cat << EOF > $SUB_MODEL
#!/bin/bash
#PBS -N test_calo_dreamer
#PBS -l walltime=40:00:00
#PBS -l nodes=1:ppn=1:gpus=1:gshort
#PBS -q gshort

module load cuda/11.4
source activate /remote/gpu06/favaro/.conda/pytorch
mydev=\`cat \$PBS_GPUFILE | sed s/.*-gpu// \`
export CUDA_VISIBLE_DEVICES=\$mydev
cd ${MAINDIR}
python3 src/main.py ${YMLFILE} -c
EOF
chmod +x $SUB_MODEL

qsub $SUB_MODEL
echo "[INFO] Submitted job: config ${YMLFILE}"

