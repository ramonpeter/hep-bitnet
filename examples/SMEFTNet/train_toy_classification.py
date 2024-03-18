import warnings
warnings.filterwarnings("ignore")

import os
import math
import numpy as np
import torch
import pickle
import glob

import sys
sys.path.insert(0, '..')
import tools.user as user

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--overwrite', action='store_true', default=False, help="restart training?")
parser.add_argument('--prefix',    action='store', default='v1', help="Prefix for training?")
parser.add_argument('--data_model', action='store', default='R1dGamma', help="Prefix for training?")
parser.add_argument('--learning_rate', '--lr',    action='store', default=0.001, help="Learning rate")
parser.add_argument('--learn_from_gamma', action='store_true',  help="SMEFTNet parameter")
parser.add_argument('--epochs', action='store', default=100, type=int, help="Number of epochs.")
parser.add_argument('--nTraining', action='store', default=1000, type=int, help="Number of epochs.")
parser.add_argument('--dRN',  action='store', type=float, default=0.4)
parser.add_argument('--conv_params',  action='store', default="( (0.0, [20, 20]), )", help="Conv params")
parser.add_argument('--readout_params',  action='store', default="(0.0, [32, 32])", help="Conv params")

args = parser.parse_args()

if args.learn_from_gamma:
    args.prefix+="_LFP"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# reproducibility
torch.manual_seed(0)
import numpy as np
np.random.seed(0)

from SMEFTNet import SMEFTNet
model = SMEFTNet(
    dRN=args.dRN,
    conv_params=eval(args.conv_params),
    readout_params=eval(args.readout_params),
    learn_from_gamma=args.learn_from_gamma,
    ).to(device)

###################### Micro MC Toy Data #########################
import MicroMC

########################## directories ###########################
model_directory = os.path.join( user.model_directory, 'SMEFTNet',  args.data_model, args.prefix)
os.makedirs( model_directory, exist_ok=True)
print ("Using model directory", model_directory)

################### model, scheduler, loss #######################
model.train()
optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=1./20)
criterion = torch.nn.BCELoss()

#################### Loading previous state #####################
model.cfg_dict = {'best_loss_test':float('inf')}
model.cfg_dict.update( {key:getattr(args, key) for key in ['prefix', 'learning_rate', 'learn_from_gamma', 'epochs', 'nTraining', 'conv_params', 'readout_params', 'dRN']} )

epoch_min = 0
if not args.overwrite:
    files = glob.glob( os.path.join( model_directory, 'epoch-*_state.pt') )
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
        model.cfg_dict = pickle.load( open( load_file_name.replace('_state.pt', '_cfg_dict.pkl'), 'rb') )
        # FIXME should add warning if keys change

########################  Training loop ##########################

for epoch in range(epoch_min, args.epochs):

    # new data every 10 epochs
    if epoch%10==0 or epoch==epoch_min:
        pt_train, pt_test, angles_train, angles_test, labels_train, labels_test = MicroMC.getEvents(*MicroMC.getModels(args.data_model), args.nTraining)
        print ("New training and test dataset.")
    optimizer.zero_grad()

    out  = model(pt=pt_train, angles=angles_train)
    loss = criterion(out[:,0], labels_train )
    n_samples = len(pt_train) 

    loss.backward()
    optimizer.step()

    with torch.no_grad():
        out_test  = model(pt=pt_test, angles=angles_test)
        loss_test = criterion(out_test[:,0], labels_test )
        if not "test_losses" in model.cfg_dict:
            model.cfg_dict["train_losses"] = []
            model.cfg_dict["test_losses"] = []
        model.cfg_dict["train_losses"].append( loss.item() )
        model.cfg_dict["test_losses"].append(  loss_test.item() )

    print(f'Epoch {epoch:03d} with N={n_samples:03d}, Loss(train): {loss:.4f} Loss(test): {loss_test:.4f}')

    model.cfg_dict['epoch']       = epoch
    model.cfg_dict['num_classes'] = 1#model.num_classes
    if args.prefix:
        if loss_test.item()<model.cfg_dict['best_loss_test']:
            model.cfg_dict['best_loss_test'] = loss_test.item()
            torch.save(  model.state_dict(),     os.path.join( model_directory, 'best_state.pt'))
            torch.save(  optimizer.state_dict(), os.path.join( model_directory, 'best_optimizer.pt'))
            pickle.dump( model.cfg_dict,          open(os.path.join( model_directory, 'best_cfg_dict.pkl'),'wb'))
            
        torch.save(  model.state_dict(),     os.path.join( model_directory, 'epoch-%d_state.pt' % epoch))
        torch.save(  optimizer.state_dict(), os.path.join( model_directory, 'epoch-%d_optimizer.pt' % epoch))
        pickle.dump( model.cfg_dict,          open(os.path.join( model_directory, 'epoch-%d_cfg_dict.pkl' % epoch),'wb'))
