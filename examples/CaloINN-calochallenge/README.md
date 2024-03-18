# caloinn
INN for Calorimeter Shower Generation

Code used for "Detector Flows" (arxiv:2312:09290) by 
Ernst F., Favaro L., Krause C., Plehn T., and Shih D.

Fast calorimeter generation for CaloGAN dataset and Fast Calorimeter Challenge

## Usage

Running a training:
```
python src/main.py params/<param_card>.yaml -c
```
This creates a subfolder in the `results` folder named `yyyymmdd_hhmmss_run_name` where the
prefix is the date and time and `run_name` is specified in the param card.

Example param card used for pions in `params/pions.yaml`
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
train\_split            | Fraction of the data set used as training data
width\_noise            | Width of the noise to be added to the data
mask                    | None or 0: no mask applied; 1: onely punch throughs; 2:without punch throughs
calo\_layer             | If given, only this calorimeter layer is used.
dtype                   | float16, float32 or float64; Higher precision makes training and generating slower.

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
prior\_prec             | Only Bayesian: Inverse of the prior standard deviation for the Bayesian layers
std\_init               | Only Bayesian: ln of the initial standard deviation of the weight distributions

### Preprocessing parameters
Parameter               | Explanation
----------------------- | --------------------------------------------------------------------
use\_extra\_dim         | If true an extra dimension is added to the data containing the ratio between parton and detector level energy. This value is used to renormalize generated data.
use\_extra\_dims        | Same as use_extra_dim onely now u1, u2 and u3 are getting stored in three extra dimensions.
use_norm                | If true samples are normalized to the incident energy. Do not use in combination with use\_extra\_dim or use\_extra\_dims 
log\_cond               | If true use the logarithm of the incident energy as condition
alpha                   | Constant value to add on the data before taking the logarithm 
