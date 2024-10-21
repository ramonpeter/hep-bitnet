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

import models.delphesTTbar as model
data_model = model.delphesTTbarModel(min_pt = 1, padding = 100, 
           features = ["eflow_cosTheta_k"], 
           scalar_features = ["delphesJet_lep_cosTheta_n", "delphesJet_lep_cosTheta_r", "delphesJet_lep_cosTheta_k"],
           train_with_truth = False ,
            )

import SMEFTNet
#import sys
#sys.path.insert(0, '..')
#sys.path.insert(0, '../..')
#sys.path.insert(0, '../../..')
import tools.user as user

#conv_params     = ( [(0.0, [20, 20])] )
#readout_params  = (0.0, [32, 32])
#dRN = 0.4

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

from SMEFTNet import SMEFTNet

def get_model( dRN, conv_params, readout_params):
    return SMEFTNet(
        dRN=dRN,
        conv_params     = conv_params,
        readout_params  =readout_params,
        learn_from_gamma=True if len(conv_params)>0 else False, 
        num_classes     = 1,
        num_features    = 1,
        include_features_in_radius = [0],
        num_scalar_features = 3,
        scalar_batch_norm = False,
        readout_batch_norm = True, #
        negative_slope = 0.3,
        regression=True ).to(device)

def MSE( out, truth, weights=None):
    return (weights*( (out[:,0] - truth)**2)).sum() 
def CrossEntropy( out, truth, weights=None):
    return -(weights*( 0.5*torch.log(1./(1.+out[:,0])**2) + 0.5*truth*torch.log((out[:,0]/(1.+out[:,0]))**2)  )).sum()
 
loss = MSE
