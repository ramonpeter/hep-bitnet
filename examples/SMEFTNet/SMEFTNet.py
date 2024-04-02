import warnings
warnings.filterwarnings("ignore")

import os
import math
import numpy as np
import torch
import pickle
import glob
import copy

from torch_geometric.nn import MessagePassing

import sys

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

from torch_geometric.nn import MLP

### alternative
#from collections import OrderedDict
#class MLP(torch.nn.Module):
#    def __init__(self, layers, dropout=0, batch_norm=True, act="LeakyRelu"):
#        super(MLP, self).__init__()
#        layers_ = []
#
#        if dropout > 0:
#            layers_.append( ('dropout', torch.nn.Dropout(p=dropout)) )
#
#        if batch_norm:
#            layers_.append( ('batch_norm', torch.nn.BatchNorm1d(layers[0])) )
#
#        for i_layer in range(len(layers)-1):
#            if act.lower()=="LeakyRelu".lower():
#                act_ = torch.nn.LeakyReLU(negative_slope=0.3)
#                layers_.append( ('act'+str(i_layer), act_) )
#            elif act is None:
#                pass
#            else:
#                raise NotImplementedError
#
#            layers_.append( ('dense'+str(i_layer), torch.nn.Linear(layers[i_layer], layers[i_layer+1])) )
#
#        self.model = torch.nn.Sequential(OrderedDict(layers_))
# 
#    # forward propagate input
#    def forward(self, x):
#        return self.model(x)

class EdgeConv(MessagePassing):
    def __init__(self, mlp):
        super().__init__(aggr="sum") 
        self.mlp = mlp

        # log messages
        self.message_logging = False
        self.message_dict    = {}
 
    def forward(self, x, edge_index):
        with torch.no_grad():
            if self.message_logging:
                self.message_dict["edge_index"] = edge_index 
        return self.propagate(edge_index, x=x)

    def message(self, x_i, x_j):

        #print( x_i[:,-2:] )
        #print( x_i[:,-2:].shape )
        #print( x_i[:,-2:].stride() )
        #print( torch.view_as_complex(x_i[:,-2:]) )

        #print( "an i", x_i[:,-2:] )
        #print( "an j", x_j[:,-2:] )
        # compute sin and cos(gamma_i-gamma_j) -> Here both rho and gamma drop out, only delta_gamma survives

        #angles_ij  = torch.view_as_real(torch.view_as_complex(x_i[:,-2:].contiguous())/torch.view_as_complex(x_j[:,-2:].contiguous()))
        norm = torch.sqrt((x_i[:,-2]**2+ x_i[:,-1]**2 )*(x_j[:,-2]**2+ x_j[:,-1]**2))
        cos_ij, sin_ij = ( (x_i[:,-2]*x_j[:,-2] + x_i[:,-1]*x_j[:,-1])/norm, (x_i[:,-1]*x_j[:,-2]-x_i[:,-2]*x_j[:,-1])/norm )
        mlp = self.mlp(torch.cat(  # .. and mlp output as fkt of 
                [x_i[:,1:-2], # features of node i  (1x nf(l))
                 x_j[:,1:-2], # features of node j  (1x nf(l))
                 x_j[:,1:-2]-x_i[:,1:-2], # differences (1x nf(l))
                 cos_ij.view(-1,1), # and two angular coordinates, in / cos(gamma_i - gamma_j) 
                 sin_ij.view(-1,1), # -> 3 nf(l) + 2
                ], dim=1)) 
        if torch.any(torch.isnan(mlp)):
            print ("Warning! Found nan in message passing MLP output. Set to zero (likely zero variance).")
            mlp = torch.nan_to_num(mlp) 
        return torch.cat(  
            ( x_i[:,:1], #return pt of node 'i' .. (1 column)
              mlp, 
              x_i[:,-2:], # ... and finally the angles of xi  (2 columns)
             ), dim=1 ) 

    def aggregate( self, inputs, index):#,  **kwargs):
        ##remember: propagate calls message, aggregate, and update
        # inputs : ( pt, MLP[nf], angles[2]) 
        # where MLP[-1] is the gamma
        # and MLP[:-1] are the features 
        # -> 1 + nf(l+1) + 1 + 2 = nf(l+1) + 4
        
        # we accumulate the pt according to the index and normalize (IRC safe pooling of messages)

        pt = inputs[:,0]
        #print ("inputs", inputs.shape, inputs[:20])
        #print ("index", index.shape)
        #print ("pt", pt.shape)

        # The index must mean which particle the message is going to.
        # Therefore, the following computes the sum of pts of all the particles going to the same place.
        # wj is then pt/sums[index]
        
        #sums = torch.zeros_like(index.unique(),dtype=torch.float).index_add_(0, index, pt)
        
        wj = pt/( torch.zeros_like(index.unique(),dtype=torch.float).index_add_(0, index, pt)[index])
        if torch.any( torch.isnan(wj)):
            print ("Warning! Found nan in pt weighted message passing (aggregation). There is a particle with only pt=0 particles in its neighbourhood. Replace with zero.")
            wj = torch.nan_to_num(wj)

        # first, weight ALL inputs
        result = torch.zeros((len(index.unique()),inputs.shape[1]),dtype=torch.float).to(device).index_add_(0, index, wj.view(-1,1)*inputs)
        # second, we take gamma=MLP[-1]=inputs[-3] and equivariantly rotate the angles in what is now results[-2]. 
        # gamma is not returned -> 1 + nf(l+1) + 2 -> nf(l+1) + 3
        #print ("EC out", result.shape)            
        result = torch.cat( (
            result[:,:-3], 
            torch.view_as_real( torch.exp( 2*torch.pi*1j*result[:,-3])*torch.view_as_complex(result[:,-2:].contiguous()) ),
            ), dim=1 )
        #if True:
        with torch.no_grad():
            if self.message_logging:
                self.message_dict["message"] = torch.sqrt( torch.square(inputs[:, 1:-3]).sum(dim=-1)).numpy() 
            
        #print ("result EC", result.shape, result)
        #print ("index", index)
        #print ("index", index.unique())
        return result

from torch_geometric.nn.pool import radius

class EIRCGNN(EdgeConv):
    def __init__(self, mlp, dRN=0.4, include_features_in_radius=()):
        super().__init__(mlp=mlp)
        # distance
        self.dRN = dRN
        # tuple index of features that should be included in computing the distance
        self.include_features_in_radius = include_features_in_radius

    def forward(self, x, batch):

        # ( pt[1], features, angles[2] )
        #pt  = x[:,0]
        # NOTE! The first feature is rho, which is always there. What we call "features" are extra features that start after 1. 
        #features = x[:,1:-2]
        #angles = x[:,-2:]

        #print ("pt",pt.shape, pt)
        #print ("(rho,features)", features.shape, features)
        #print ("angles", angles.shape, angles)
        #print ("self.include_features_in_radius",self.include_features_in_radius)

        max_num_neighbors = max(batch.unique(return_counts=True)[1]).item()
        if self.include_features_in_radius is not None and len(self.include_features_in_radius)>0:
            features = x[:,2:-2] # We start taking from "2:" because the first feature in the first EIRCGNN is rho
            x_radius = torch.stack( [x[:,-2], x[:,-1]] + [features[:, pos] for pos in self.include_features_in_radius] ).transpose(0,1)
        else:
            x_radius = x[:,-2:]
        #print ("x_radius", x_radius.shape, x_radius)
        edge_index = radius(x_radius, x_radius, r=self.dRN, batch_x=batch, batch_y=batch, max_num_neighbors=max_num_neighbors)
        return super().forward(x, edge_index=edge_index)

class SMEFTNet(torch.nn.Module):
    def __init__(self, 
            num_classes  = 1, 
            num_features = 0,
            include_features_in_radius = (), 
            num_scalar_features = 0,
            scalar_batch_norm = True, 
            conv_params=( (0.0, [10, 10]), (0.0, [10, 10]) ), 
            dRN=0.4, 
            readout_params=(0.0, [32, 32]), 
            readout_batch_norm="batch_norm",
            negative_slope = 0.01,
            learn_from_gamma=False, regression=False):
        super().__init__()

        self.learn_from_gamma = learn_from_gamma
        self.regression          = regression
        self.num_classes         = num_classes
        self.num_features        = num_features
        self.num_scalar_features = num_scalar_features
        self.scalar_batch_norm   = torch.nn.BatchNorm1d(num_scalar_features) if (num_scalar_features>0 and scalar_batch_norm) else None
        # tuple index of features that should be included in computing the distance
        self.include_features_in_radius = include_features_in_radius
        self.readout_batch_norm = readout_batch_norm 
        if self.readout_batch_norm and self.scalar_batch_norm: print ("Warning! Two batch norms for scalar features!")

        self.EC = torch.nn.ModuleList()

        for l, (dropout, hidden_layers) in enumerate(conv_params):
            hidden_layers_ = copy.deepcopy(hidden_layers)
            hidden_layers_[-1]+=1 # separate output for gamma-coordinate 
            if l==0:
                _mlp = MLP([3*(1+num_features) + 2 ]+hidden_layers_, dropout=dropout, act="LeakyRelu")
                _mlp.act.negative_slope = negative_slope
                # only include features in radius in the first layer
                self.EC.append( EIRCGNN( _mlp, dRN=dRN, include_features_in_radius=include_features_in_radius ) )
            else:
                _mlp = MLP([3*conv_params[l-1][1][-1]+2]+hidden_layers_,dropout=dropout, act="LeakyRelu")
                _mlp.act.negative_slope = negative_slope
                self.EC.append( EIRCGNN( _mlp, dRN=dRN ) ) 

        if len(self.EC)>0:
            # output features + cos/sin gamma
            EC_out_chn = hidden_layers[-1]
            # whether we're going to feed cos/sin gamma
            if self.learn_from_gamma:
                EC_out_chn += 2
        else:
            # the case where we do not have a gNN
            EC_out_chn = 0

        self.mlp = MLP( [EC_out_chn+self.num_scalar_features]+readout_params[1]+[num_classes], dropout=readout_params[0], act="LeakyRelu",batch_norm=self.readout_batch_norm)
        self.mlp.act.negative_slope = negative_slope

        if not self.regression:
            self.out = torch.nn.Sigmoid()

    @classmethod
    def load(cls, directory, epoch=None):
        if epoch is None:
            load_file_name = 'best_state.pt'
        else:
            load_file_name = 'epoch-%d_state.pt'%epoch
        load_file_name = os.path.join( directory, load_file_name)
        cfg_dict = pickle.load(open(load_file_name.replace('_state.pt', '_cfg_dict.pkl'),'rb'))
        model = cls( num_classes=cfg_dict['num_classes'] if "num_classes" in cfg_dict else 1, 
                     conv_params=eval(cfg_dict['conv_params']), dRN=cfg_dict['dRN'], readout_params=eval(cfg_dict['readout_params']), 
                     learn_from_gamma=cfg_dict['learn_from_gamma'] if 'learn_from_gamma' in cfg_dict else cfg_dict['learn_from_phi'])
        model_state = torch.load(load_file_name, map_location=device)
        model.load_state_dict(model_state)
        model.cfg_dict = cfg_dict
        model.eval()
        return model

    def forward(self, pt, angles, features=None, scalar_features=None, message_logging=False, return_EIRCGNN_output=False):

        if len(self.EC)>0:
            # for IRC tests we actually low zero pt. Zero abs angles define the mask
            mask = (pt != 0)
            batch= (torch.arange(len(mask)).to(device).view(-1,1)*mask.int())[mask]

            # we feed pt in col. 0, rho (as feature) in col. 1, then the features, and finally the angles in col. 2,3
            if features is not None:
                assert features.shape[2]==self.num_features, "Got %i features but was expecting %i."%( features.shape[2], self.num_features)
                x = torch.cat( (pt[mask].view(-1,1), torch.view_as_complex( angles[mask] ).abs().view(-1,1), features[mask], angles[mask]), dim=1)
            else: 
                x = torch.cat( (pt[mask].view(-1,1), torch.view_as_complex( angles[mask] ).abs().view(-1,1), angles[mask]), dim=1)

            for l, EC in enumerate(self.EC):
                EC.message_logging = message_logging
                x = EC(x, batch)

            # global IRC safe message pooling
            pt = x[:,0] 
            wj = pt/( torch.zeros_like(batch.unique(),dtype=torch.float).index_add_(0, batch, pt))[batch]
            if torch.any( torch.isnan(wj)):
                print ("Warning! Found nan in pt weighted readout. Are there no particles with pt>0?. Replace with zero.")
                wj = torch.nan_to_num(wj)

            # disregard first column (pt, keep the last two ones: cos/sin gamma)
            x = torch.zeros((len(batch.unique()),x[:,1:].shape[1]),dtype=torch.float).to(device).index_add_(0, batch, wj.view(-1,1)*x[:,1:])

            # Return only the pooled message, for plotting etc. 
            if return_EIRCGNN_output:
                if self.learn_from_gamma == True:
                    return x 
        # THIS is the default case -> we pass the pooled message through the output MLP & the 'out' layer (except for regression where we don't use the 'out' layer)

        if scalar_features is not None:
            y = self.scalar_batch_norm(scalar_features) if self.scalar_batch_norm is not None else scalar_features
            if len(self.EC)>0:
                # prepend scalar_features to feed into MLP
                x = torch.cat( (y, x), 1)
            else:
                # we only have scalar features, no gNN poresent
                x = y

        if len(self.EC)>0:
            if self.learn_from_gamma == True:
                if self.regression: 
                    return torch.cat( (self.mlp( x ), x[:, -2:]), dim=1)
                else:
                    return torch.cat( (self.out(self.mlp( x )), x[:, -2:]), dim=1)
            else:
                if self.regression: 
                    return torch.cat( (self.mlp( x[:, :-2] ), x[:, -2:]), dim=1)
                else:
                    return torch.cat( (self.out(self.mlp( x[:, :-2] )), x[:, -2:]), dim=1)
        else:
            if self.learn_from_gamma == True:
                raise RuntimeError( "No EC layer, can't learn from gamma!" )           
            else:
                if self.regression: 
                    return self.mlp( x ).view(-1,1)
                else:
                    return self.out(self.mlp( x[:, :-2] )).view(-1,1)

    # intercept EIRCGNN output
    def EIRCGNN_output( self, pt, angles, message_logging=False):
        return self.forward( pt=pt, angles=angles, message_logging=message_logging, return_EIRCGNN_output=True)

if __name__=="__main__":

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--overwrite', action='store_true', default=False, help="restart training?")
    parser.add_argument('--prefix',    action='store', default='v1', help="Prefix for training?")
    parser.add_argument('--learning_rate', '--lr',    action='store', default=0.001, help="Learning rate")
    parser.add_argument('--learn_from_gamma', action='store_true',  help="SMEFTNet parameter")
    parser.add_argument('--epochs', action='store', default=100, type=int, help="Number of epochs.")
    parser.add_argument('--nTraining', action='store', default=1000, type=int, help="Number of epochs.")
    args = parser.parse_args()

    if args.learn_from_gamma:
        args.prefix+="_LFP"

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # reproducibility
    torch.manual_seed(0)
    import numpy as np
    np.random.seed(0)


    ################ Micro MC Toy Data #####################
    import MicroMC
    from sklearn.model_selection import train_test_split
    ##################### TESTS  ##########################
    signal     = MicroMC.make_model( R=1, gamma=0, var=0.3 )
    background = MicroMC.make_model( R=0, gamma=0, var=0.3 )
    
    pt_sig, angles_sig = MicroMC.sample(signal, 100)
    pt_bkg, angles_bkg = MicroMC.sample(background, 100)
    
    label_sig = np.ones(  len(pt_sig) )
    label_bkg = np.zeros( len(pt_bkg) )
    
    pt_train, pt_test, angles_train, angles_test, labels_train, labels_test = train_test_split( 
            np.concatenate( (pt_sig, pt_bkg) ), 
            np.concatenate( (angles_sig, angles_bkg) ), 
            np.concatenate( (label_sig, label_bkg) )
        )
    maxN = 1
    pt_train     = torch.Tensor(pt_train[:maxN]).to(device)
    angles_train = torch.Tensor(angles_train[:maxN]).to(device)
    labels_train = torch.Tensor(labels_train[:maxN]).to(device)
    # model instance
    model  = SMEFTNet(learn_from_gamma=False).to(device)
    model.eval()
    #with torch.no_grad():
    #    result = model( pt=pt_train, angles=angles_train)
    ##################### EQUIVARIANCE #######################
        #for i in range(101):
        #    gamma = 2*math.pi*i/100
        #    R   = torch.Tensor( [[math.cos(gamma), math.sin(gamma)],[-math.sin(gamma), math.cos(gamma)]] )
        #    angles_train_ = torch.matmul( angles_train, R)

        #    #rho = torch.view_as_complex( angles_train ).abs()
        #    result = model( pt=pt_train, angles=angles_train_)
        #    classifier, angles = result[:,:-2], result[:,-2:]
        #    print ("classifier", classifier.item(), "angle", torch.atan2(angles[:,1], angles[:,0]).item()/(math.pi) )
    #################### IR safety #######################
        ### add a bunch of soft particles
        #pt_train     = torch.Tensor([[1.]]).to(device)
        #angles_train = torch.Tensor([[[.5, .5]]]).to(device)
        #result = model( pt=pt_train, angles=angles_train)
        #classifier, angles = result[:,:-2], result[:,-2:]
        #print ("orig classifier", classifier.item(), "angle", torch.atan2(angles[:,1], angles[:,0]).item()/(math.pi) )

        #for i in range(-3,9):
        #    pt_soft = 10**(-i)
        #    pt_train     = torch.Tensor([[1., pt_soft, pt_soft, pt_soft]]).to(device)
        #    angles_train = torch.Tensor([[[.5, .5], [1., 0.], [-.3,.4], [5,-.6]]]).to(device)
        #    result = model( pt=pt_train, angles=angles_train)
        #    classifier, angles = result[:,:-2], result[:,-2:]
        #    print ("classifier", classifier.item(), "angle", torch.atan2(angles[:,1], angles[:,0]).item()/(math.pi) )
    #################### Collinear safety #######################
        #pt_train     = torch.Tensor([[1., 2.]]).to(device)
        #angles_train = torch.Tensor([[[.5, .5], [-.3,.3]]]).to(device)
        #result = model( pt=pt_train, angles=angles_train)
        #classifier, angles = result[:,:-2], result[:,-2:]
        #print ("orig classifier", classifier.item(), "angle", torch.atan2(angles[:,1], angles[:,0]).item()/(math.pi) )
    
        #for i in range(0,11):
        #    l = i/10. 
        #    pt_train     = torch.Tensor([[1., 2*l, 2*(1-l)]]).to(device)
        #    angles_train = torch.Tensor([[[.5, .5], [-.3, .3], [-.3, .3]]]).to(device)
        #    result = model( pt=pt_train, angles=angles_train)
        #    classifier, angles = result[:,:-2], result[:,-2:]
        #    print ("i",i,"classifier", classifier.item(), "angle", torch.atan2(angles[:,1], angles[:,0]).item()/(math.pi) )
    
    ########################## directories ###########################
    import tools.user as user
    model_directory = os.path.dirname( os.path.join( user.model_directory, 'EIRCGNN', args.prefix ))
    os.makedirs( model_directory , exist_ok=True)

    ################ Loading previous state ###########################
    epoch_min = 0
    if not args.overwrite:
        files = glob.glob( os.path.join( user.model_directory, 'EIRCGNN', args.prefix + '_epoch-*_state.pt') )
        if len(files)>0:
            load_file_name = max( files, key = lambda f: int(f.split('-')[-1].split('_')[0]))
            load_epoch = int(load_file_name.split('-')[-1].split('_')[0])
        else:
            load_epoch = None
        if load_epoch is not None:
            print('Resume training from %s' % load_file_name)
            model_state = torch.load(load_file_name, map_location=device)
            model.load_state_dict(model_state)
            opt_state_file = load_file_name.replace('_state.pt', '_optimizer.pt') 
            if os.path.exists(opt_state_file):
                opt_state = torch.load(opt_state_file, map_location=device)
                optimizer.load_state_dict(opt_state)
            else:
                print('Optimizer state file %s NOT found!' % opt_state_file)
            epoch_min=load_epoch+1

    ####################  Training loop ##########################
    signal     = MicroMC.make_model( R=1, gamma=0, var=0.3 )
    background = MicroMC.make_model( R=1, gamma=math.pi/4, var=0.3 )

    def getEvents( nTraining=args.nTraining ):

        pt_sig, angles_sig = MicroMC.sample(signal, nTraining)
        pt_bkg, angles_bkg = MicroMC.sample(background, nTraining)

        label_sig = torch.ones(  len(pt_sig) )
        label_bkg = torch.zeros( len(pt_bkg) )
        return train_test_split( 
            torch.Tensor(np.concatenate( (pt_sig, pt_bkg) )).to(device), 
            torch.Tensor(np.concatenate( (angles_sig, angles_bkg) )).to(device), 
            torch.Tensor(np.concatenate( (label_sig, label_bkg) )).to(device)
        )
    
    model  = SMEFTNet(learn_from_gamma=args.learn_from_gamma).to(device)
    model.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=1./20)

    criterion = torch.nn.BCELoss()

    pt_train, pt_test, angles_train, angles_test, labels_train, labels_test = getEvents(args.nTraining) 
    for epoch in range(epoch_min, args.epochs):

        optimizer.zero_grad()

        out  = model(pt=pt_train, angles=angles_train)
        loss = criterion(out[:,0], labels_train )
        n_samples = len(pt_train) 

        loss.backward()
        optimizer.step()

        if args.prefix:
            torch.save( model.state_dict(), os.path.join( user.model_directory, 'EIRCGNN', args.prefix + '_epoch-%d_state.pt' % epoch))
            torch.save( optimizer.state_dict(), os.path.join( user.model_directory, 'EIRCGNN', args.prefix + '_epoch-%d_optimizer.pt' % epoch))

        with torch.no_grad():
            out_test  = model(pt=pt_test, angles=angles_test)
            loss_test = criterion(out_test[:,0], labels_test )

        print(f'Epoch {epoch:03d} with N={n_samples:03d}, Loss(train): {loss:.4f} Loss(test): {loss_test:.4f}')
