# BitHEP — The Limits of Low-Precision ML in HEP

<p align="center">
<a href="https://arxiv.org/abs/2504.????"><img alt="Arxiv" src="https://img.shields.io/badge/arXiv-2504.12345-b31b1b.svg"></a>
<a href="https://github.com/psf/black"><img alt="Code style: black" src="https://img.shields.io/badge/code%20style-black-000000.svg"></a>
<a href="https://pytorch.org"><img alt="pytorch" src="https://img.shields.io/badge/PyTorch-2.0-DC583A.svg?style=flat&logo=pytorch"></a>
</p>

> Official code for [BitHEP — The Limits of Low-Precision ML in HEP](https://arxiv.org/abs/2504.????)  
by Claudius Krause, Daohan Wang, Ramon Winterhalder.

This repo contains a PyTorch implementation of the BitLinear layer as proposed in these papers:

- [BitNet: Scaling 1-bit Transformers for Large Language Models](2310.11453)
- [The Era of 1-bit LLMs: All Large Language Models are in 1.58 Bits](https://arxiv.org/abs/2402.17764)

It also contains all examples shown in our paper.

## Installation

You can install by cloning the repository and do pip install

```sh
# clone the repository
git clone https://github.com/ramonpeter/hep-bitnet
# then install (add '-e' for dev mode)
cd bitnet
pip install (-e) .
```

## Experiments

Details about the performed experiments can be found in the subfolder `examples`