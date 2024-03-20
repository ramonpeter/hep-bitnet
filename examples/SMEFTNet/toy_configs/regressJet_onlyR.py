import warnings
warnings.filterwarnings("ignore")

import os
import math
import numpy as np
import torch
import pickle
import glob
import itertools
import operator
import functools


# Let us import toy generator 
from toy_models.JetModel import JetModel

# Main class
import SMEFTNet

# Add yourself in this file
import tools.user as user

# Configure message passing like in PNet. Here, we just use one MLP with two hidden layers (20, 20) and no dropout
conv_params     = ( [(0.0, [20, 20])] )
readout_params  = (0.0, [32, 32])
# This important hyperparameter defines the dR distance when computing neighbours
dRN = 0.4

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Instanciate a data model that can generate "events". Let us generate "jets" drawn from two Gaussians separated by a distance R between 0 and 1.5.
# The variance of each Gaussian is 0.3, so these are well separated 
data_model = JetModel(minR=0, maxR=1.5, minGamma=0, maxGamma=0, events_per_parampoint=1)

from SMEFTNet import SMEFTNet
model = SMEFTNet(
    dRN=dRN,
    conv_params=conv_params,
    readout_params=readout_params,
    learn_from_gamma=True,
    num_classes = 1,
    regression=True,
    ).to(device)

def loss( out, truth, weights=None ):
    return ( (out[:,0] - truth[:,0])**2 ).sum() 
