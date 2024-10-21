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

scalar_features = ["genJet_pt",'parton_lepV_pt']
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

# def loss( out, truth, weights=None):
#     weight_sm = weights[:,0]
#     target    = weights[:,1]
#     return torch.mean( weight_sm*torch.abs( out[:,0] - target ))

def loss( out, truth, weights=None):
    weight_sm = weights[:,0]
    target    = weights[:,1]
    return torch.mean( weight_sm*( out[:,0] - target )**2)

truth_var_names = ['phi', 'hadV_pt', 'lepV_pt']
ranges = [(-math.pi, math.pi), (500, 1000), (0,1000)]
def plot( out, truth, weights, model_directory, epoch):
    weight_sm = weights[:,0]
    target    = weights[:,1]


    for var, (varname, (xlow,xhigh)) in enumerate(zip(truth_var_names,ranges)):
        score = ROOT.TH1F("score_%s"%varname, "score", 50, xlow, xhigh)
        pred  = ROOT.TH1F("pred_%s"%varname , "pred" , 50, xlow, xhigh)
        norm  = ROOT.TH1F("norm_%s"%varname , "norm" , 50, xlow, xhigh)

        
        score.Add( helpers.make_TH1F( np.histogram(truth[:,var], np.linspace(xlow, xhigh, 50+1), weights=weight_sm*target) ))
        pred .Add( helpers.make_TH1F( np.histogram(truth[:,var], np.linspace(xlow, xhigh, 50+1), weights=weight_sm*out[:,0]) ))
        norm .Add( helpers.make_TH1F( np.histogram(truth[:,var], np.linspace(xlow, xhigh, 50+1), weights=weight_sm)) )
        
        score.Divide(norm)
        pred.Divide(norm)

        c1 = ROOT.TCanvas()
        score.SetLineStyle(2)
        score.GetXaxis().SetTitle(varname)
        score.Draw()
        pred.Draw("same")
        c1.Print( 'closure_%s_%03i.png'%(varname,epoch))
