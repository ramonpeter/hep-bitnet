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

from models.WZModel import WZModel
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

scalar_features = ["genJet_pt"]
#scalar_features = []

data_model = WZModel(what='VV', scalar_features = scalar_features)
#data_model = WZModel(what='lab')

from SMEFTNet import SMEFTNet
model = SMEFTNet(
    dRN=dRN,
    conv_params=conv_params,
    readout_params=readout_params,
    learn_from_gamma=True,
    num_scalar_features=len(scalar_features),
    num_classes = 1,
    regression=True,
   ).to(device)

def get_model(dRN, conv_params, readout_params):
    model = SMEFTNet(
        dRN=dRN,
        conv_params=conv_params,
        readout_params=readout_params,
        learn_from_gamma=True,
        num_scalar_features=len(scalar_features),
        num_classes=1,
        regression=True,
    )
    return model.to(device)

def loss(out, truth, weights=None):
    #return torch.min( (out[:,0] - torch.sin(truth[:,1]))**2 + (out[:,1] - torch.cos(truth[:,1]))**2, (out[:,0] + torch.sin(truth[:,1]))**2 + (out[:,1] + torch.cos(truth[:,1]))**2 ).sum() 
    #return ( (out[:,0] - torch.sin(truth[:,1]))**2 + (out[:,1] - torch.cos(truth[:,1]))**2).sum() 
    mu = out[:,0]
    logsigma2 = out[:,1]
    alpha = 0.2
#    logsigma2 = out[:, 1]
    dPhi = mu-truth[:,0]
    sin_dPhi = torch.sin(dPhi)
    out = torch.pow(sin_dPhi, 2) / (2 * logsigma2.exp()) + alpha / 2.0 * logsigma2
    #return torch.abs( dPhi/(math.pi) - torch.floor( dPhi/(math.pi) + 0.5 ) ).sum() 
    return torch.mean(out)

def plot( out, truth, weights, model_directory, epoch):
    
    plt.hist2d( out[:,0].numpy(),  truth.numpy() , bins=30)
    plt.savefig( os.path.join( model_directory, f'test_cov_{epoch}.png'))
    plt.clf()
