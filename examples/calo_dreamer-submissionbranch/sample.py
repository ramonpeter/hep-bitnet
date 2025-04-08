#!/usr/bin/env python3

import h5py
import os
import time
import torch
import yaml

import calodream

from argparse import ArgumentParser


parser = ArgumentParser()
parser.add_argument(
    '--model' ,
    help='Directory containing model config and checkpoint'
    )
parser.add_argument(
    '--energy_model',
    default='models/energy',
    help='Directory containing config and checkpoint for the energy model'
    )
parser.add_argument(
    '--sample_size',
    type=int,
    default=10**5,
    help='The number of samples to generate'
    )
parser.add_argument(
    '--batch_size',
    type=int,
    default=10**4,
    help='The batch size used for sampling'
    )
parser.add_argument(
    '--use_cpu',
    action='store_true',
    help='Whether to run on cpu'
    )
parser.add_argument(
    '--which_cuda',
    default=0,
    help='Index of the cuda device to run on'
    )
args = parser.parse_args()

def sample(args):

    config_path = os.path.join(args.model, 'params.yaml')
    chkpnt_path = os.path.join(args.model, 'model.pt')

    # choose device
    device = 'cpu' if args.use_cpu else f'cuda:{args.which_cuda}'
    print(f"[sample.py] Using device '{device}'")
    
    # read config file
    with open(config_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    # set path to energy model
    config['energy_model'] = args.energy_model

    # choose dtype
    dtype = config.get('dtype', '')
    if   dtype=='float32': torch.set_default_dtype(torch.float32)    
    elif dtype=='float64': torch.set_default_dtype(torch.float64)
    elif dtype=='float16': torch.set_default_dtype(torch.float16)

    # load model
    model = config.get('model', 'TBD')
    try:
        model = getattr(calodream, model)(config, device, args.model)
    except AttributeError:
        raise NotImplementedError(f'build_model: Model class {model} not recognised')
    model.load(chkpnt_path)

    # sample model
    print(f'[sample.py] Start sampling model...')
    t_0 = time.time()
    showers, energies = model.sample(args.sample_size, args.batch_size) 
    t_1 = time.time()
    print(f"[sample.py]: Finished generating {len(showers)} samples in {t_1 - t_0:.2f} s.")

    # implement voxel energy cutoff
    showers *= showers >= config['eval_cut']

    # save samples
    savepath = os.path.join(args.model, 'samples.hdf5')
    with h5py.File(savepath, 'w') as f:
        f.create_dataset('showers', data=showers)
        f.create_dataset('incident_energies', data=energies)
    print(f'[sample.py] Saved samples to {savepath}.')

if __name__ == '__main__':
    sample(args)

