# Experiments

We perform different experiments:

1. **PDAT**: A classification network to perform quak-gluon discrimination which is based on [arXiv:2307.04723](https://arxiv.org/abs/2307.04723)
2. **SMEFTNet**: A regression network to infer SMEFT parameters based on [arXiv:2401.10323](https://arxiv.org/abs/2401.10323) and taken from [SMEFTNet](https://github.com/HephyAnalysisSW/SMEFTNet/tree/master/)
3. Detector Simulation:
    - **CaloINN**: Normalizing flow based calorimeter simulation as proposed in [arXiv:2312.09290](https://arxiv.org/abs/2312.09290) adapted from [[code]](https://github.com/heidelberg-hepml/CaloINN) at commit `3384038`. 
      The submission scripts for training and sampling follow the following naming scheme:

      | script name | setup name in paper |
      | :---------: | :-----------------: |
      | regular     | Default             |
      | splitregularnet | Exchange Permutation | 
      | bitnet | NNCentral |
      | splitbitnet | BlockCentral |
      | fullbitnet | All |

    - **CaloDREAM**: Flow-matching and transformer based calorimeter simulation as proposed in [arXiv:2405.09629](https://arxiv.org/abs/2405.09629). The code here is split in two parts, one for training (adapted from [[code]](https://github.com/luigifvr/calo_dreamer/tree/master) in the folder `calo_dreamer-masterbranch`) and one for sampling (adapted from [[code]](https://github.com/luigifvr/calo_dreamer/tree/submission) in the folder `calo_dreamer-submissionbranch`). 
      The individual parameter cards for CaloDREAM are in `calo_dreamer_params`. The submission scripts for training follow the naming scheme:

      | script name | setup name in paper |
      | :---------: | :-----------------: |
      | train_regular | train regular energy net |
      | train_fullbitnet | train quantized energy net |
      | train_shape_regular | train regular shape net |
      | train_shape_noembbitnet | train no embedding shape net |
      | train_shape_aggressivebitnet | train full shape net |

    In generation, we use the files `*_sample_EN_SH.sbatch`, where `EN` refers to the configuration of the energy net (`reg` for regular, `158b` for quantized) and `SH` refers to the configuration of the shape net (`reg` for regular, `noemb158b` or `aggressive158b` for the quantized setups defined before).

    Evaluation of the generated samples is based on the [[code]](https://github.com/CaloChallenge/homepage/blob/more_evaluations/code/evaluate.py) used in the [CaloChallenge](https://calochallenge.github.io/homepage/) with more details at [arXiv:2410.21611](https://arxiv.org/abs/2410.21611).

More details are given in each sub-folder.
