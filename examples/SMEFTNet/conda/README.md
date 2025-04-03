# Create Conda environment

Scripts to create environment for weaver with GPU.

## Installations

The environment can be installed using [miniconda](https://docs.conda.io/en/latest/miniconda.html) or 
[mambaforge](https://github.com/conda-forge/miniforge#mambaforge). 

The environment is relatively large and can require relatively long time to build.

[Mamba](https://mamba.readthedocs.io/en/latest/) is a drop-in replacement for conda, written in C++, that 
can handle large environments in a more efficient way. Mamba can be enables inside the scripts.

Some dependencies are not available in  conda and are additionally installed from [pypi](http://pypi)

## Versions

* python 3.10
* root 6.28.0
* cuda 11.7.1

## Scripts

    create_new_environment.sh [env name]

creates a new environment by resolving dependencies.
For reproducibility the resolved dependencies are frozen
in ```environment-list.txt```.

    create_environment.sh [env name]
     
cerates a new environment from the frozen configuration.

## Scripts

After the environment is created and activated do
```
pip install torch_geometric
pip install torch_cluster
```


