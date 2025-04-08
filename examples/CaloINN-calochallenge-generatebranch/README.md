# CaloINN
INN for Calorimeter Shower Generation

Code used for "Normalizing Flows for High-Dimensional Detector Simulations" (arxiv:2312:09290) by 
Ernst F., Favaro L., Krause C., Plehn T., and Shih D.

The samples used in the paper are publicly available on Zenodo. [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.14178546.svg)](https://doi.org/10.5281/zenodo.14178546)

Fast calorimeter generation for CaloGAN dataset and Fast Calorimeter Challenge.

This is the main repository for the full-space CaloINN network. For the CaloGAN data results check the branch "calogan_data",
while for our latent model see "VAE+INN".

## Train a model

Start a training:
```
python src/main.py params/<param_card>.yaml -c
```
This creates a subfolder in the `results` folder named `yyyymmdd_hhmmss_run_name` where the
prefix is the date and time and `run_name` is specified in the param card.

Example param card used for pions in `params/pions.yaml`

## Generate samples from an existing model
Assuming the model is located in `<dir>` and saved model with name `model_<name>.pt`, to generate a new sample in this directory run:
```
python3 src/main.py <dir>/params.yaml -d <dir> -its <name> --generate -c 
```
Additionaly `--nsamples` provides a way to specify the number of samples generated for dataset 2.


To only run the plotting from the CaloChallenge pipeline using the previously generated samples switch
`--generate` with `--plot`.

## Parameters

This is a list of the parameters that can be used in yaml parameter files. Many have default
values, such that not all parameters have to be specified.

### Run parameters

Parameter               | Explanation
----------------------- | --------------------------------------------------------------------
run\_name               | Name for the output folder

### Data parameters

Parameter               | Explanation
----------------------- | --------------------------------------------------------------------
data\_path              | Name of the hdf5 file containing the data set
val\_data\_path         | Name of the hdf5 file containing the validation data set
xml\_path               | Name of the XML file containing the calorimeter binning
val\_frac               | Fraction of the data used for validation
width\_noise            | Higher end of the uniform noise to be added to the data
dtype                   | float16, float32 or float64; Higher precision makes training and generating slower
single\_energy          | Train on a single incident energy, only for dataset 1
xml\_ptype              | Specifics for the XML file: "photon", "pion", or "electron"
eval\_dataset           | Needed for the validation used in the CaloChallenge pipeline: "1-photons", "1-pions", "2", or "3"

### Training parameters

Parameter               | Explanation
----------------------- | --------------------------------------------------------------------
lr                      | Learning rate
lr\_sched\_mode         | Type of LR scheduling: "reduce\_on\_plateau", "step" or "one\_cycle"
lr\_decay\_epochs       | Only step scheduler: decay interval in epochs
lr\_decay\_factor       | Only step scheduler: decay factor
batch\_size             | Batch size
weight\_decay           | L2 weight decay
betas                   | List of the two Adam beta parameters
eps                     | Adam eps parameter for numerical stability
n\_epochs               | Number of training epochs
save\_interval          | Interval in epochs for saving
grad\_clip              | If given, a gradient clip with the given value is applied


### Architecture parameters

Parameter               | Explanation
----------------------- | --------------------------------------------------------------------
n\_blocks               | Number of coupling blocks
internal\_size          | Internal size of the coupling block subnetworks
layers\_per\_block      | Number of layers in each coupling block subnetwork
dropout                 | Dropout fraction for the subnetworks
permute\_soft           | If True, uses random rotation matrices instead of permutations
coupling\_type          | Type of coupling block: "affine", "cubic", "rational\_quadratic" or "MADE"
clamping                | Only affine blocks: clamping parameter
num\_bins               | Only spline blocks: number of bins
bounds\_init            | Only spline blocks: bounds of the splines
bayesian                | True to enable Bayesian training
sub\_layers             | A list for partial Bayesian networks, e.g. \[linear, linear, VBLinear\]
prior\_prec             | Only Bayesian: Inverse of the prior standard deviation for the Bayesian layers
std\_init               | Only Bayesian: ln of the initial standard deviation of the weight distributions
layer\_act              | Activation function in the subnetwork
norm                    | Apply ActNorm after preprocessing

### Preprocessing parameters
Parameter               | Explanation
----------------------- | --------------------------------------------------------------------
use\_extra\_dim         | If true an extra dimension is added to the data containing the ratio between parton and detector level energy. This value is used to renormalize generated data.
use\_extra\_dims        | Adds as extra dimensions the energy variables u_i
use_norm                | If true samples are normalized to the incident energy. Do not use in combination with use\_extra\_dim or use\_extra\_dims 
log\_cond               | If true use the logarithm of the incident energy as condition
alpha                   | Constant value to add on the data before taking the logarithm 
