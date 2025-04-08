#!/usr/bin/env python

"""A script to benchmark the performance/efficiency tradeoff of different
   ODE samplers for generating calorimeter showers."""

import Models
import os
import torch
from argparse import ArgumentParser
from challenge_files import evaluate
from documenter import Documenter
from torch.utils.data import DataLoader
from torchdiffeq import odeint
from Util.util import load_params

FIXED_SOLVERS = ['euler', 'midpoint', 'rk4']
ADAPTIVE_SOLVERS = ['dopri5']
BESPOKE_SOLVERS = ['BespokeEuler', 'BespokeMidpoint', 'BespokeNonStationary']
solver_choices = FIXED_SOLVERS + ADAPTIVE_SOLVERS + BESPOKE_SOLVERS

parser = ArgumentParser()
parser.add_argument('--shape_model', required=True)
parser.add_argument('--energy_model', required=True)
parser.add_argument('--solver', choices=solver_choices, required=True)
parser.add_argument('--no_cuda', action='store_true')
parser.add_argument('--bespoke_dir')
parser.add_argument('--steps', type=int)
parser.add_argument('--tols', type=float)
parser.add_argument('--n_samples', type=int, default=5000)
parser.add_argument('--n_runs', type=int, default=20)
parser.add_argument('--eval_mode', default='cls-high')
parser.add_argument('--batch_size', type=int, default=10_000)
args = parser.parse_args()

class SolveFunc:

    def __init__(self, net, cond, device):
        self.net = net
        self.cond = cond
        self.device = device
        self.nfe = 0

    def __call__(self, t, x):
        self.nfe += 1
        t = t.repeat((x.shape[0],1)).to(self.device)
        return self.net(x, t, self.cond)

def benchmark(args, doc, device):

    assert (args.steps is None) ^ (args.tols is None), \
        "Exactly one of `steps` or `tols` must be set!"

    # load shape and energy models
    models = {}
    for model_type in 'energy', 'shape':
        model_dir = getattr(args, model_type+'_model')
        params = load_params(os.path.join(model_dir, 'params.yaml'))
        params['eval_mode'] = args.eval_mode 
        model = getattr(Models, params.get('model', 'TBD'))(
            params, device=device, doc=Documenter(None, existing_run=model_dir, read_only=True)
        )
        model.load()
        # set to eval mode
        model.eval()
        models[model_type] = model

    
    if args.solver in BESPOKE_SOLVERS:
        # load bespoke solver
        solver = getattr(Models, args.solver)(
            params=load_params(os.path.join(args.bespoke_dir, 'params.yaml')),
            doc=Documenter(None, existing_run=args.bespoke_dir, read_only=True),
            device=device
        )
        solver.load()
    else:
        # or set odeint options
        solve_times = torch.tensor([0, 1], dtype=torch.float32, device=device)
        solver_kwargs = (
            {'options': {'step_size': 1/args.steps}}
            if args.solver in FIXED_SOLVERS else
            {'atol': args.tols, 'rtol': args.tols}
        )             
    
    for _ in range(args.n_runs):
        
        with torch.inference_mode():
            # initialize condition loader
            Eincs = DataLoader(
                dataset=torch.rand([args.n_samples, 1], device=device), batch_size=args.batch_size,
                shuffle=False
            )

            # loop over batches and generate sample
            samples, conds, nfes = [], [], []
            for Einc in Eincs:

                # first sample layer energies
                u_sample = models['energy'].sample_batch(Einc)
                cond = torch.cat([Einc, u_sample], dim=1)
                del u_sample

                # dispatch to chosen solver and sample shower
                if args.solver in BESPOKE_SOLVERS:
                    sample = solver.solve(cond)
                else:
                    solve_fn = SolveFunc(models['shape'].net, cond, device)
                    y0 = torch.randn((args.batch_size, *models['shape'].shape), device=device)
                    sample = odeint(
                        solve_fn, y0, solve_times, method=args.solver, **solver_kwargs
                    )[-1]
                
                # optionally keep track of nfe
                if args.solver in ADAPTIVE_SOLVERS:
                    nfes.append(solve_fn.nfe)
                
                # collect samples on cpu
                samples.append(sample.cpu())
                conds.append(cond.cpu())
                    
            # post-process
            sample = torch.vstack(samples)
            cond = torch.vstack(conds)
            for fn in models['shape'].transforms[::-1]:
                sample, cond = fn(sample, cond, rev=True)

        # classify
        evaluate.run_from_py(sample.numpy(), cond.numpy(), doc, models['shape'].params)

        # append mean NFE to the log
        if args.solver in ADAPTIVE_SOLVERS:
            with open(doc.get_file('eval/classifier_cls-high_2.txt'), 'a') as f:
                f.write(f"NFE: {sum(nfes)/len(nfes):.3f}\n")
        
if __name__ == '__main__':

    device = 'cuda' if torch.cuda.is_available() and  not args.no_cuda else 'cpu'
    precision = args.tols if args.solver in ADAPTIVE_SOLVERS else args.steps
    doc = Documenter(f'benchmark_{args.solver}_{precision}')
    torch.set_default_dtype(torch.float32) # was a PAIN to realise this is needed

    benchmark(args, doc, device)

        