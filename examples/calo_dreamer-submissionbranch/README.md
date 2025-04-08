# CaloDREAM

A repository for fast detector simulation using Conditional Flow Matching
on the CaloChallenge datasets

## Usage

Example 1 (minimal):

Run the dataset 2 model with default settings
```
python3 sample.py models/d2
```

Options:

The following arguments can optionally be specified
Flag		| Usage
------------------------| ----------------------------------------
`energy_model`  | Directory containing config and checkpoint for the energy model \[default: `models/energy`\]
`sample_size`   | The number of samples to generate \[default: `100_000`\]
`batch_size`    | The batch size used for sampling \[default `5_000`\]
`use_cpu`       | A flag indicating that the cpu should be used
`which_cuda`    | Index of the cuda device to use \[default `0`\]

Example 2:

Run the dataset 3 model on cpu for specific sample and batch sizes
```
python3 sample.py models/d3 --sample_size 10000 --batch_size 100 --use_cpu
```
## Outputs
After runnning `sample.py`, a `HDF5` dataset will be created in the model directory, with keys `showers` and `incident_energies`.
