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

from toy_models.EFTModel import EFTModel

import SMEFTNet
import LikleihoodFreeInference

#import sys
#sys.path.insert(0, '..')
#sys.path.insert(0, '../..')
import tools.user as user

conv_params     = ( [(0.0, [20, 20])] )
#conv_params     = ( [(0.0, [10, 10]), (0.0, [20, 20])] ) NOT better
readout_params  = (0.0, [32, 32])
dRN = 0.6

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

coefficients = ['thetaG']
#coefficients = ['thetaR']#, 'thetaG']
first_derivatives  = sorted(list(itertools.combinations_with_replacement(coefficients,1))) 
second_derivatives = sorted(list(itertools.combinations_with_replacement(coefficients,2))) 
derivatives        = [tuple()] + first_derivatives + second_derivatives

base_points = [1, 2]
#for comb in list(itertools.combinations_with_replacement(coefficients,1))+list(itertools.combinations_with_replacement(coefficients,2)):
#    base_points.append( {c:comb.count(c) for c in coefficients} )

model = SMEFTNet.SMEFTNet(
    dRN=dRN,
    conv_params=conv_params,
    readout_params=readout_params,
    learn_from_gamma=True,
    num_classes = len(derivatives)-1,
    regression=True,
    ).to(device)

data_model = EFTModel(0,0, events_per_parampoint=1)

loss = LikleihoodFreeInference.MSELoss(base_points=base_points)#, derivatives=[('thetaR',), ('thetaR', 'thetaR')])

## precoumputed base_point_const
#base_point_const = np.array([[ functools.reduce(operator.mul, [point[coeff] if (coeff in point) else 0 for coeff in der ], 1) for der in derivatives] for point in base_points]).astype('float')
#for i_der, der in enumerate(derivatives):
#   if not (len(der)==2 and der[0]==der[1]): continue
#   for i_point in range(len(base_points)):
#       base_point_const[i_point][i_der]/=2.
#
#assert np.linalg.matrix_rank(base_point_const) == base_point_const.shape[0], \
#          "Base points not linearly independent! Found rank %i for %i base_points" %( np.linalg.matrix_rank(base_point_const), base_point_const.shape[0])

#            if self.cfg['loss'] == 'MSE':
#                neg_loss_gains = np.sum(np.dot( sorted_weight_sums, self.base_point_const.transpose())**2,axis=1)/sorted_weight_sums[:,0]
#                neg_loss_gains+= np.sum(np.dot( sorted_weight_sums_right, self.base_point_const.transpose())**2,axis=1)/sorted_weight_sums_right[:,0]
#            elif self.cfg['loss'] == 'CrossEntropy':
#                with np.errstate(divide='ignore', invalid='ignore'):
#                    r       = np.dot( sorted_weight_sums, self.base_point_const.transpose())/sorted_weight_sums[:,0].reshape(-1,1)
#                    r_right = np.dot( sorted_weight_sums_right, self.base_point_const.transpose())/sorted_weight_sums_right[:,0].reshape(-1,1)
#                    #neg_loss_gains  = sorted_weight_sums[:,0]*np.sum( ( r*0.5*np.log(r**2) + (1.-r)*0.5*np.log((1.-r)**2) ), axis=1)
#                    #neg_loss_gains += sorted_weight_sums_right[:,0]*np.sum( ( r_right*0.5*np.log(r_right**2) + (1.-r_right)*0.5*np.log((1.-r_right)**2) ), axis=1)
#                    neg_loss_gains  = sorted_weight_sums[:,0]*      np.sum( ( 0.5*np.log((1./(1.+r))**2) + r*0.5*np.log((r/(1.+r))**2) ), axis=1)
#                    neg_loss_gains += sorted_weight_sums_right[:,0]*np.sum( ( 0.5*np.log((1./(1.+r_right))**2) + r_right*0.5*np.log((r_right/(1.+r_right))**2) ), axis=1)
#                    neg_loss_gains -= min(neg_loss_gains)# make loss positive
