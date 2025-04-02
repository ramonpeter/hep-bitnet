# Higgs ML Uncertainty Challenge -- HEPHY

## Hardware

CPU model: Intel(R) Xeon(R) Gold 6138 CPU @ 2.00GHz
Architecture: x86\_64
number of CPU cores: 1
memory: 20GB
GPU: not needed

Minimum requirements:
number of CPU cores: 1
memory: 20GB
GPU: not needed

## OS

CentOS Linux 7

## 3rd-party software and environment setup

TensorFlow: `pip install tensorflow`
imunuit: `pip install iminuit`

To ensure a consistent environment, we recommend using **Conda**. You can create the required environment using:

```bash
conda env create -f environment.yml
conda activate uncertainty_challenge_new
```


## ML models

There are two models: 
- TensorFlow multiclassifier (TFMC)
- Parametric neural network (PNN)

### Training

The models are pre-trained, so no training needed.


### Inference

The TFMC and PNN are used together to infer the interval of the signal strength. The evaluation is performed in the `model.py`, with the input of a config file `configs/config_submission.yaml`.

#### Config files

The config file `configs/config_submission.yaml` is hardcoded in `model.py`. The important sections are the following:

- `Tasks`: This specifies which tasks to run, including multiclassifier and the parametric neural network for all the processes. This section should not be changed.
- `Selections`: The framework applies selections (defined in `common/selections.py`) on events to categorize them into different regions. This specifies which regions to use in the signal strength inference.
- `CSI`: This sets whether to use cubic spline interpolation for inclusive cross section. It should not be changed.
- `MultiClassifier/htautau/ztautau/ttbar/diboson`: The ML architechture and model paths for each regions are set in these sections. Different files are provided for different tasks. For multiclassifier, `model_path` and `calibration` are provided, the `calibration` is optional. For PNN in different processes, the inclusive cross section parametrization file `icp_file` and `model_path` are provided.

#### Trained models

The trained models are stored in `models/*Task*/*selection*/*specifics*/`. The `*Task*`, `*selection*`, and `*specifics*` are explained below:
- `Task`: The `Tasks` in the config file, includes `MultiClassifier`, `htautau`, `ztautau`, `ttbar`, and `diboson`.
- `selection`: The `Selections` in the config file, includes `lowMT_VBFJet`, `lowMT_noVBFJet_ptH100`, and `highMT_VBFJet`.
- `specifics`: The stored files includes the trained model path (`model_path`), calibration files for multiclassifier (`calibration`), and inclusive cross section parametrization file (`icp_file`).

In addition, the `CSI` files for training data is used as well in the prediction. Those files are saved in `data/tmp_data/`.

## Side effects

- Running `predict.py` produces a `results.json` under the `SUBMISSION_DIR` provided in the arguments. The json file will be overwritten if the file already exists.

## Key assumptions

- The framework applies selections on the data before processing it. The selections are defined in `common/selections.py`. A set of selections can be used when infering the signal strength, as specified in `configs/config_submission.yaml`. When running the framework, it assumes non-zero events from all selections.
- The script `predict.py` should be run under the main directory.
