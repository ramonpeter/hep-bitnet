# ELSA: Enhanced Latent SpAces for improved collider simulations


[![arXiv](http://img.shields.io/badge/arXiv-2305.xxxx-B31B1B.svg)](https://arxiv.org/abs/23??)

## Introduction

This repo contains the code for the **E**nhanced **L**atent **S**p**A**ces (ELSA) framework
for neural network improved collider simulations. It is based on the LaSeR protocol [[1]](#laser) and
further employs augmented flows [[2]](#survae). The code is provided in PyTorch. 

## Installation

### Dependencies

**Package**     | **Version**
----------------|-------------------------------------------------
Python          | >= 3.7
Torch           | >= 1.8
Numpy           | >= 1.20.0 


### Download + Install

```sh
# clone the repository
git clone https://github.com/ramonpeter/elsa.git
# then install in dev mode
cd elsa
python setup.py develop
```

### Prepare datasets

In order to download the datasets

```bash
./get_datasets.sh
```
   
This prepares and/or downloads the datasets into the **datasets** folder.



## References 

<a name="laser">[1]</a> Latent Space Refinement: [2106.00792](https://arxiv.org/abs/2106.00792)
   - Code: [https://github.com/ramonpeter/LaSeR](https://github.com/ramonpeter/LaSeR)

<a name="survae">[2]</a> SurVAE Flows: [2007.02731](https://arxiv.org/abs/2007.02731)
   - Code [https://github.com/didriknielsen/survae_flows](https://github.com/didriknielsen/survae_flows)
