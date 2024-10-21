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
import ROOT 
import tools.helpers as helpers
ROOT.gROOT.SetBatch(True)

from models.WZandDYModel import WZandDYModel
import matplotlib.pyplot as plt 

import SMEFTNet
#import sys
#sys.path.insert(0, '..')
#sys.path.insert(0, '../..')
#sys.path.insert(0, '../../..')
import tools.user as user

conv_params     = ( [(0.0, [20, 20])] )
readout_params  = (0.0, [32, 32])
dRN = 0.4

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#scalar_features = ["genJet_pt",'parton_lepV_pt']
scalar_features = []

#data_model = WZModel(what='VV', scalar_features = scalar_features)
data_model = WZandDYModel(what='lab')

from SMEFTNet import SMEFTNet

def get_model( dRN, conv_params, readout_params):
    return SMEFTNet(
        dRN=dRN,
        conv_params     = conv_params,
        readout_params  =readout_params,
        learn_from_gamma=True if len(conv_params)>0 else False,
        num_classes     = 1,
        #num_features    = 1,
        #include_features_in_radius = [0],
        #num_scalar_features = 3,
        scalar_batch_norm = False,
        readout_batch_norm = True, #
        negative_slope = 0.3,
        regression=False ).to(device)

# def loss( out, truth, weights=None):
#     weight_sm = weights[:,0]
#     target    = weights[:,1]
#     return torch.mean( weight_sm*torch.abs( out[:,0] - target ))

#loss = torch.nn.CrossEntropyLoss()

def loss( out, truth, weights=None):
    #weight_sm = weights[:,0]
    #target    = weights[:,1]
    #print (out)
    #print( truth)
    return - torch.sum( truth*torch.log( out ) +(1-truth)*torch.log(1-out) )
