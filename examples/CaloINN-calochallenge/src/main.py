import shutil
import argparse
import os

import yaml
import torch

from documenter import Documenter
from trainer import Trainer

def main():
    parser = argparse.ArgumentParser(description='train network')
    parser.add_argument('param_file', help='where to find the parameters')
    parser.add_argument('-c', '--use_cuda', action='store_true', default=False,
        help='whether cuda should be used')
    parser.add_argument('-p', '--plot', action='store_true', default=False,
        help='make only plots for traint modle')
    parser.add_argument('-d', '--model_dir', default=None,
        help='model directory for only plot run')
    parser.add_argument('-its', '--model_name')
    args = parser.parse_args()

    with open(args.param_file) as f:
        params = yaml.load(f, Loader=yaml.FullLoader)
    use_cuda = torch.cuda.is_available() and args.use_cuda
    device = 'cuda:0' if use_cuda else 'cpu'
    
    if args.plot:
        doc = Documenter(params['run_name'], existing_run=args.model_dir)
    else:
        doc = Documenter(params['run_name'])
    shutil.copy(args.param_file, doc.get_file('params.yaml'))
    print('device: ', device)

    dtype = params.get('dtype', '')
    if dtype=='float64':
        torch.set_default_dtype(torch.float64)
    elif dtype=='float16':
        torch.set_default_dtype(torch.float16)
    elif dtype=='float32':
        torch.set_default_dtype(torch.float32)

    plot_params = {}
    plot_configs = [
        'plot_params/plot_layer_0.yaml',
        'plot_params/plot_layer_1.yaml',
        'plot_params/plot_layer_2.yaml',
        'plot_params/plots.yaml'
    ]
    calo_layer = params.get('calo_layer', None)
    if calo_layer is None:
        for file_name in plot_configs:
            with open(file_name) as f:
                plot_params.update(yaml.load(f, Loader=yaml.FullLoader))
    else:
        with open(plot_configs[calo_layer]) as f:
            plot_params.update(yaml.load(f, Loader=yaml.FullLoader))

    trainer = Trainer(params, device, doc)
    if args.plot:
        trainer.load(args.model_name)
    else:
        trainer.train()
    if 'bayesian' in params and params['bayesian']:
        trainer.plot_uncertaintys(plot_params)

if __name__=='__main__':
    main()
