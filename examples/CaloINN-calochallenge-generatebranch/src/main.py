import shutil
import argparse
import os

import yaml
import torch

from documenter import Documenter
from trainer import Trainer

def main():
    parser = argparse.ArgumentParser(description='train network')
    parser.add_argument('param_file', help='yaml file location with all the parameters')
    parser.add_argument('-c', '--use_cuda', action='store_true', default=False,
        help='whether cuda should be used')
    parser.add_argument('-p', '--plot', action='store_true', default=False,
        help='only run the evaluation script')
    parser.add_argument('-g', '--generate', action='store_true', default=False,
        help='generate and save a new sample from a trained model')
    parser.add_argument('-n', '--nsamples', type=int, default=100000,
        help='number of samples, only used for ds2')
    parser.add_argument('-d', '--model_dir', default=None,
        help='directory used to load a model')
    parser.add_argument('-its', '--model_name', default='_last',
        help='name of the model used to generate the new sample')
    args = parser.parse_args()

    with open(args.param_file) as f:
        params = yaml.load(f, Loader=yaml.FullLoader)
    use_cuda = torch.cuda.is_available() and args.use_cuda
    device = 'cuda:0' if use_cuda else 'cpu'
    
    if args.plot or args.generate:
        doc = Documenter(params['run_name'], existing_run=args.model_dir)
    else:
        doc = Documenter(params['run_name'])
    
    # disabled by Claudius
    #try:
    #    shutil.copy(args.param_file, doc.get_file('params.yaml'))
    #except shutil.SameFileError:
    #    pass
    print('device: ', device)

    dtype = params.get('dtype', '')
    if dtype=='float64':
        torch.set_default_dtype(torch.float64)
    elif dtype=='float16':
        torch.set_default_dtype(torch.float16)
    elif dtype=='float32':
        torch.set_default_dtype(torch.float32)

    trainer = Trainer(params, device, doc)
    if args.generate:
        trainer.load(args.model_name)
        trainer.generate(args.nsamples)
        #trainer.plot_default_from_caloch(
        #        sample_name='samples.hdf5', eval_name='final', cut=1.515e-3
        #        )
    elif args.plot:
        trainer.plot_default_from_caloch(
                sample_name='samples.hdf5', eval_name='final', cut=1.515e-3
                ) 
    else:
        trainer.train()

if __name__=='__main__':
    main()
