import warnings

warnings.filterwarnings("ignore")

import argparse
import glob
import math
import os
import pickle

import numpy as np
import torch

parser = argparse.ArgumentParser()
parser.add_argument(
    "--overwrite", action="store_true", default=False, help="restart training?"
)
parser.add_argument(
    "--prefix", action="store", default="v1", help="Prefix for training?"
)
parser.add_argument(
    "--config", action="store", default="regressJet", help="Which config?"
)
parser.add_argument(
    "--learning_rate",
    "--lr",
    action="store",
    type=float,
    default=0.001,
    help="Learning rate",
)
parser.add_argument(
    "--epochs", action="store", default=100, type=int, help="Number of epochs."
)
parser.add_argument("--clip", action="store", type=float, default=None)
parser.add_argument("--dRN", action="store", type=float, default=0.4)
parser.add_argument(
    "--conv_params", action="store", default="( (0.0, [20, 20]), )", help="Conv params"
)
parser.add_argument(
    "--readout_params", action="store", default="(  0.0, [32, 32])", help="Conv params"
)
parser.add_argument("--small", action="store_true", help="Small?")

args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import sys

sys.path.insert(0, "..")
import tools.helpers as helpers
import tools.user as user

exec("import configs.%s as config" % args.config)

# reproducibility
torch.manual_seed(0)
import numpy as np

np.random.seed(0)

########################## directories ###########################
model_directory = os.path.join(
    user.model_directory, "SMEFTNet", args.config, args.prefix
)
os.makedirs(model_directory, exist_ok=True)
print("Using model directory", model_directory)
################### make model ###################################

model = config.get_model(
    dRN=args.dRN,
    conv_params=eval(args.conv_params),
    readout_params=eval(args.readout_params),
)

state_dict = torch.load(
    "/users/daohan.wang/SMEFTNet/NN/models/SMEFTNet/regress_wz_eft/eft_wz_regressed/best_state.pt"
)
model.load_state_dict(state_dict)
model.eval()


output_list = []
target_list = []
decay_phi_list = []
for i_data, data in enumerate(config.data_model.data_generator):
    pt, angles, features, scalar_features, weights, truth = config.data_model.getEvents(
        data
    )
    train_mask = torch.ones(pt.shape[0], dtype=torch.bool)
    train_mask[int(pt.shape[0] * 0.8) :] = False
    with torch.no_grad():
        out_test = model(
            pt=pt[~train_mask],
            angles=angles[~train_mask],
            features=features[~train_mask] if features is not None else None,
            scalar_features=(
                scalar_features[~train_mask] if scalar_features is not None else None
            ),
        )
        weights = weights[~train_mask]
        truth = truth[~train_mask]
        target = weights[:, 1]
        decay_phi = truth[:, 0]
        out_test_np = out_test.numpy()
        target_np = target.numpy()
        decay_phi_np = decay_phi.numpy()
    output_list.append(out_test_np)
    target_list.append(target_np)
    decay_phi_list.append(decay_phi_np)

output = np.concatenate(output_list)
target = np.concatenate(target_list)
decay_phi = np.concatenate(decay_phi_list)
np.savez(
    "/users/daohan.wang/SMEFTNet/test_data.npz",
    output=output,
    target=target,
    decay_phi=decay_phi,
)
