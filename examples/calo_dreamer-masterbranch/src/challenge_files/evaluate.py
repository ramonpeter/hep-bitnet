# pylint: disable=invalid-name
""" Main script to evaluate contributions to the Fast Calorimeter Challenge 2022

    modified by L. Favaro, A. Ore, S. Palacios for CaloDREAM

    input:
        - path to a folder containing .hdf5 samples. 
          The script loads only files with *samples.hdf5 in the name
    output:
        - metrics for evaluation (plots, classifier scores, etc.)

    usage:
        -i --input_file: path of the input files to be evaluated.
        -r --reference_file: Name and path of the reference .hdf5 file.
        -m --mode: Which metric to look at. Choices are
                   'all': does all of the below (with low-level classifier).
                   'avg': plots the average shower of the whole dataset.
                   'avg-E': plots the average showers at different energy (ranges).
                   'hist-p': plots histograms of high-level features.
                   'hist-chi': computes the chi2 difference of the histograms.
                   'hist': plots histograms and computes chi2.
                   'all-cls': only run classifiers in list_cls
                   'no-cls': does all of the above (no classifier).
                   'cls-low': trains a classifier on low-level features (voxels).
                   'cls-low-normed': trains a classifier on normalized voxels.
                   'cls-high': trains a classifier on high-level features (same as histograms).
        -d --dataset: Which dataset the evaluation is for. Choices are
                      '1-photons', '1-pions', '2', '3'
           --output_dir: Folder in which the evaluation results (plots, scores) are saved.
           --save_mem: If included, data is moved to the GPU batch by batch instead of once.
                       This reduced the memory footprint a lot, especially for datasets 2 and 3.

           --no_cuda: if added, code will not run on GPU, even if available.
           --which_cuda: Which GPU to use if multiple are available.

    additional options for the classifier start with --cls_ and can be found below.
"""

import argparse
import os
import pickle
from glob import glob

import numpy as np
import matplotlib.pyplot as plt
import h5py
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.calibration import calibration_curve
from sklearn.isotonic import IsotonicRegression

import challenge_files.HighLevelFeatures as HLF

from challenge_files.evaluate_plotting_helper import *

torch.set_default_dtype(torch.float64)

plt.rc("font", family="serif", size=16)
plt.rc("axes", titlesize="medium")
plt.rc("text.latex", preamble=r"\usepackage{amsmath}")
plt.rc("text", usetex=True)
#hardcoded labels for histograms
labels = ['ViT', 'latViT']

########## Parser Setup ##########

def define_parser():
    parser = argparse.ArgumentParser(description=('Evaluate calorimeter showers of the '+\
                                                  'Fast Calorimeter Challenge 2022.'))

    parser.add_argument('--input_file', '-i', help='Path of the inputs file to be evaluated.')
    parser.add_argument('--reference_file', '-r',
                        help='Name and path of the .hdf5 file to be used as reference. ')
    parser.add_argument('--mode', '-m', default='all',
                        choices=['all', 'all-cls', 'no-cls', 'avg', 'avg-E', 'hist-p', 
                                 'hist-chi', 'hist', 'cls-low', 'cls-low-normed', 'cls-high'],
                        help=("What metric to evaluate: " +\
                              "'avg' plots the shower average;" +\
                              "'avg-E' plots the shower average for energy ranges;" +\
                              "'hist-p' plots the histograms;" +\
                              "'hist-chi' evaluates a chi2 of the histograms;" +\
                              "'hist' evaluates a chi2 of the histograms and plots them;" +\
                              "'cls-low' trains a classifier on the low-level feautures;" +\
                              "'cls-low-normed' trains a classifier on the low-level feautures" +\
                              " with calorimeter layers normalized to 1;" +\
                              "'cls-high' trains a classifier on the high-level features;" +\
                              "'all' does the full evaluation, ie all of the above" +\
                              " with low-level classifier."))
    parser.add_argument('--dataset', '-d', choices=['1-photons', '1-pions', '2', '3'],
                        help='Which dataset is evaluated.')
    parser.add_argument('--output_dir', default='evaluation_results/',
                        help='Where to store evaluation output files (plots and scores).')

    parser.add_argument('--cut', type=float)
    parser.add_argument('--energy', type=float, default=None)

    parser.add_argument('--cls_n_layer', type=int, default=2,
                        help='Number of hidden layers in the classifier, default is 2.')
    parser.add_argument('--cls_n_hidden', type=int, default='512',
                        help='Hidden nodes per layer of the classifier, default is 512.')
    parser.add_argument('--cls_dropout_probability', type=float, default=0.,
                        help='Dropout probability of the classifier, default is 0.')

    parser.add_argument('--cls_batch_size', type=int, default=1000,
                        help='Classifier batch size, default is 1000.')
    parser.add_argument('--cls_n_epochs', type=int, default=50,
                        help='Number of epochs to train classifier, default is 50.')
    parser.add_argument('--cls_lr', type=float, default=2e-4,
                        help='Learning rate of the classifier, default is 2e-4.')

    # CUDA parameters
    parser.add_argument('--no_cuda', action='store_true', help='Do not use cuda.')
    parser.add_argument('--which_cuda', default=0, type=int,
                        help='Which cuda device to use')

    parser.add_argument('--save_mem', action='store_true',
                        help='Data is moved to GPU batch by batch instead of once in total.')
    return parser

########## Functions and Classes ##########

class DNN(torch.nn.Module):
    """ NN for vanilla classifier. Does not have sigmoid activation in last layer, should
        be used with torch.nn.BCEWithLogitsLoss()
    """
    def __init__(self, num_layer, num_hidden, input_dim, dropout_probability=0.):
        super(DNN, self).__init__()

        self.dpo = dropout_probability

        self.inputlayer = torch.nn.Linear(input_dim, num_hidden)
        self.outputlayer = torch.nn.Linear(num_hidden, 1)

        all_layers = [self.inputlayer, torch.nn.LeakyReLU(), torch.nn.Dropout(self.dpo)]
        for _ in range(num_layer):
            all_layers.append(torch.nn.Linear(num_hidden, num_hidden))
            all_layers.append(torch.nn.LeakyReLU())
            all_layers.append(torch.nn.Dropout(self.dpo))

        all_layers.append(self.outputlayer)
        self.layers = torch.nn.Sequential(*all_layers)

    def forward(self, x):
        """ Forward pass through the DNN """
        x = self.layers(x)
        return x

def prepare_low_data_for_classifier(voxel_orig, E_inc_orig, hlf_class, label, cut=0.0, normed=False, single_energy=None):
    """ takes hdf5_file, extracts Einc and voxel energies, appends label, returns array """
    voxel = voxel_orig.copy()
    E_inc = E_inc_orig.copy()
    if normed:
        E_norm_rep = []
        E_norm = []
        for idx, layer_id in enumerate(hlf_class.GetElayers()):
            E_norm_rep.append(np.repeat(hlf_class.GetElayers()[layer_id].reshape(-1, 1),
                                        hlf_class.num_voxel[idx], axis=1))
            E_norm.append(hlf_class.GetElayers()[layer_id].reshape(-1, 1))
        E_norm_rep = np.concatenate(E_norm_rep, axis=1)
        E_norm = np.concatenate(E_norm, axis=1)
    if normed:
        voxel = voxel / (E_norm_rep+1e-16)
        ret = np.concatenate([np.log10(E_inc), voxel, np.log10(E_norm+1e-8),
                              label*np.ones_like(E_inc)], axis=1)
    else:
        voxel = voxel / E_inc
        ret = np.concatenate([np.log10(E_inc), voxel, label*np.ones_like(E_inc)], axis=1)
    return ret

def prepare_high_data_for_classifier(voxel_orig, E_inc_orig, hlf_class, label, cut=0.0, single_energy=None):
    """ takes hdf5_file, extracts high-level features, appends label, returns array """
    E_inc = E_inc_orig.copy()
    E_tot = hlf_class.GetEtot()
    E_layer = []
    for layer_id in hlf_class.GetElayers():
        E_layer.append(hlf_class.GetElayers()[layer_id].reshape(-1, 1))
    EC_etas = []
    EC_phis = []
    Width_etas = []
    Width_phis = []
    for layer_id in hlf_class.layersBinnedInAlpha:
        EC_etas.append(hlf_class.GetECEtas()[layer_id].reshape(-1, 1))
        EC_phis.append(hlf_class.GetECPhis()[layer_id].reshape(-1, 1))
        Width_etas.append(hlf_class.GetWidthEtas()[layer_id].reshape(-1, 1))
        Width_phis.append(hlf_class.GetWidthPhis()[layer_id].reshape(-1, 1))
    E_layer = np.concatenate(E_layer, axis=1)
    EC_etas = np.concatenate(EC_etas, axis=1)
    EC_phis = np.concatenate(EC_phis, axis=1)
    Width_etas = np.concatenate(Width_etas, axis=1)
    Width_phis = np.concatenate(Width_phis, axis=1)
    ret = np.concatenate([np.log10(E_inc), np.log10(E_layer+1e-8), EC_etas/1e2, EC_phis/1e2,
                          Width_etas/1e2, Width_phis/1e2, label*np.ones_like(E_inc)], axis=1)
    return ret

def ttv_split(data1, data2, split=np.array([0.6, 0.2, 0.2])):
    """ splits data1 and data2 in train/test/val according to split,
        returns shuffled and merged arrays
    """
    #assert len(data1) == len(data2)
    if len(data1) < len(data2):
        data2 = data2[:len(data1)]
    elif len(data1) > len(data2):
        data1 = data1[:len(data2)]
    else:
        assert len(data1) == len(data2)
    num_events = (len(data1) * split).astype(int)
    np.random.shuffle(data1)
    np.random.shuffle(data2)
    train1, test1, val1 = np.split(data1, num_events.cumsum()[:-1])
    train2, test2, val2 = np.split(data2, num_events.cumsum()[:-1])
    train = np.concatenate([train1, train2], axis=0)
    test = np.concatenate([test1, test2], axis=0)
    val = np.concatenate([val1, val2], axis=0)
    np.random.shuffle(train)
    np.random.shuffle(test)
    np.random.shuffle(val)
    print(len(train), len(test), len(val))
    return train, test, val

def load_classifier(constructed_model, parser_args):
    """ loads a saved model """
    filename = parser_args.mode + '_' + parser_args.dataset + '.pt'
    checkpoint = torch.load(os.path.join(parser_args.output_dir, filename),
                            map_location=parser_args.device)
    constructed_model.load_state_dict(checkpoint['model_state_dict'])
    constructed_model.to(parser_args.device)
    constructed_model.eval()
    print('classifier loaded successfully')
    return constructed_model


def train_and_evaluate_cls(model, data_train, data_test, optim, arg):
    """ train the model and evaluate along the way"""
    best_eval_acc = float('-inf')
    arg.best_epoch = -1
    try:
        for i in range(arg.cls_n_epochs):
            train_cls(model, data_train, optim, i, arg)
            with torch.inference_mode():
                eval_acc, _, _ = evaluate_cls(model, data_test, arg)
            if eval_acc > best_eval_acc:
                best_eval_acc = eval_acc
                arg.best_epoch = i+1
                filename = arg.mode + '_' + arg.dataset + '.pt'
                torch.save({'model_state_dict':model.state_dict()},
                           os.path.join(arg.output_dir, filename))
            if eval_acc == 1.:
                break
    except KeyboardInterrupt:
        # training can be cut short with ctrl+c, for example if overfitting between train/test set
        # is clearly visible
        pass

def train_cls(model, data_train, optim, epoch, arg):
    """ train one step """
    model.train()
    for i, data_batch in enumerate(data_train):
        if arg.save_mem:
            data_batch = data_batch[0].to(arg.device)
        else:
            data_batch = data_batch[0]
        input_vector, target_vector = data_batch[:, :-1], data_batch[:, -1]
        output_vector = model(input_vector)
        criterion = torch.nn.BCEWithLogitsLoss()
        loss = criterion(output_vector, target_vector.unsqueeze(1))

        optim.zero_grad(set_to_none=True)
        loss.backward()
        optim.step()

        if i % (len(data_train)//2) == 0:
            print('Epoch {:3d} / {}, step {:4d} / {}; loss {:.4f}'.format(
                epoch+1, arg.cls_n_epochs, i, len(data_train), loss.item()))
        # PREDICTIONS
        pred = torch.round(torch.sigmoid(output_vector.detach()))
        target = torch.round(target_vector.detach())
        if i == 0:
            res_true = target
            res_pred = pred
        else:
            res_true = torch.cat((res_true, target), 0)
            res_pred = torch.cat((res_pred, pred), 0)

    print("Accuracy on training set is",
          accuracy_score(res_true.cpu(), res_pred.cpu()))

def evaluate_cls(model, data_test, arg, final_eval=False, calibration_data=None):
    """ evaluate on test set """
    model.eval()
    for j, data_batch in enumerate(data_test):
        if arg.save_mem:
            data_batch = data_batch[0].to(arg.device)
        else:
            data_batch = data_batch[0]
        input_vector, target_vector = data_batch[:, :-1], data_batch[:, -1]
        output_vector = model(input_vector)
        pred = output_vector.reshape(-1)
        target = target_vector.double()
        if j == 0:
            result_true = target
            result_pred = pred
        else:
            result_true = torch.cat((result_true, target), 0)
            result_pred = torch.cat((result_pred, pred), 0)
    BCE = torch.nn.BCEWithLogitsLoss()(result_pred, result_true)
    result_pred = torch.sigmoid(result_pred).cpu().numpy()
    result_true = result_true.cpu().numpy()
    eval_acc = accuracy_score(result_true, np.round(result_pred))
    print("Accuracy on test set is", eval_acc)
    eval_auc = roc_auc_score(result_true, result_pred)
    print("AUC on test set is", eval_auc)
    JSD = - BCE + np.log(2.)
    print("BCE loss of test set is {:.4f}, JSD of the two dists is {:.4f}".format(BCE,
                                                                                  JSD/np.log(2.)))
    if final_eval:
        prob_true, prob_pred = calibration_curve(result_true, result_pred, n_bins=10)
        print("unrescaled calibration curve:", prob_true, prob_pred)
        calibrator = calibrate_classifier(model, calibration_data, arg)
        rescaled_pred = calibrator.predict(result_pred)
        eval_acc = accuracy_score(result_true, np.round(rescaled_pred))
        print("Rescaled accuracy is", eval_acc)
        eval_auc = roc_auc_score(result_true, rescaled_pred)
        print("rescaled AUC of dataset is", eval_auc)
        prob_true, prob_pred = calibration_curve(result_true, rescaled_pred, n_bins=10)
        print("rescaled calibration curve:", prob_true, prob_pred)
        # calibration was done after sigmoid, therefore only BCELoss() needed here:
        BCE = torch.nn.BCELoss()(torch.tensor(rescaled_pred, dtype=torch.get_default_dtype()), torch.tensor(result_true, dtype=torch.get_default_dtype()))
        JSD = - BCE.cpu().numpy() + np.log(2.)
        otp_str = "rescaled BCE loss of test set is {:.4f}, "+\
            "rescaled JSD of the two dists is {:.4f}"
        print(otp_str.format(BCE, JSD/np.log(2.)))
    return eval_acc, eval_auc, JSD/np.log(2.)

def calibrate_classifier(model, calibration_data, arg):
    """ reads in calibration data and performs a calibration with isotonic regression"""
    model.eval()
    assert calibration_data is not None, ("Need calibration data for calibration!")
    for j, data_batch in enumerate(calibration_data):
        if arg.save_mem:
            data_batch = data_batch[0].to(arg.device)
        else:
            data_batch = data_batch[0]
        input_vector, target_vector = data_batch[:, :-1], data_batch[:, -1]
        output_vector = model(input_vector)
        pred = torch.sigmoid(output_vector).reshape(-1)
        target = target_vector.to(torch.float64)
        if j == 0:
            result_true = target
            result_pred = pred
        else:
            result_true = torch.cat((result_true, target), 0)
            result_pred = torch.cat((result_pred, pred), 0)
    result_true = result_true.cpu().numpy()
    result_pred = result_pred.cpu().numpy()
    iso_reg = IsotonicRegression(out_of_bounds='clip', y_min=1e-6, y_max=1.-1e-6).fit(result_pred,
                                                                                      result_true)
    return iso_reg

def check_file(given_file, arg, which=None):
    """ checks if the provided file has the expected structure based on the dataset """
    print("Checking if {} file has the correct form ...".format(
        which if which is not None else 'provided'))
    num_features = {'1-photons': 368, '1-pions': 533, '2': 6480, '3': 40500}[arg.dataset]
    num_events = given_file['incident_energies'].shape[0]
    assert given_file['showers'].shape[0] == num_events, \
        ("Number of energies provided does not match number of showers, {} != {}".format(
            num_events, given_file['showers'].shape[0]))
    assert given_file['showers'].shape[1] == num_features, \
        ("Showers have wrong shape, expected {}, got {}".format(
            num_features, given_file['showers'].shape[1]))

    print("Found {} events in the file.".format(num_events))
    print("Checking if {} file has the correct form: DONE \n".format(
        which if which is not None else 'provided'))

def extract_shower_and_energy(given_file, which, single_energy=None, max_len=-1):
    """ reads .hdf5 file and returns samples and their energy """
    print("Extracting showers from {} file ...".format(which))
    if single_energy is not None:
        energy_mask = given_file["incident_energies"][:] == single_energy
        energy = given_file["incident_energies"][:][energy_mask].reshape(-1, 1)
        shower = given_file["showers"][:][energy_mask.flatten()]
    else:
        shower = given_file['showers'][:max_len]
        energy = given_file['incident_energies'][:max_len]
    print("Extracting showers from {} file: DONE.\n".format(which))
    return shower.astype('float32', copy=False), energy.astype('float32', copy=False)

def load_reference(filename):
    """ Load existing pickle with high-level features for reference in plots """
    print("Loading file with high-level features.")
    with open(filename, 'rb') as file:
        hlf_ref = pickle.load(file)
    return hlf_ref

def save_reference(ref_hlf, fname):
    """ Saves high-level features class to file """
    print("Saving file with high-level features.")
    with open(fname, 'wb') as file:
        pickle.dump(ref_hlf, file)
    print("Saving file with high-level features DONE.")

def plot_histograms(hlf_classes, reference_class, arg, labels, input_names='', p_label=''):
    """ plots histograms based with reference file as comparison """
    plot_Etot_Einc(hlf_classes, reference_class, arg, labels, input_names, p_label)
    plot_E_layers(hlf_classes, reference_class, arg, labels, input_names, p_label)
    plot_ECEtas(hlf_classes, reference_class, arg, labels, input_names, p_label)
    plot_ECPhis(hlf_classes, reference_class, arg, labels, input_names, p_label)
    plot_ECWidthEtas(hlf_classes, reference_class, arg, labels, input_names, p_label)
    plot_ECWidthPhis(hlf_classes, reference_class, arg, labels, input_names, p_label)
    plot_sparsity(hlf_classes, reference_class, arg, labels, input_names, p_label)
    plot_weighted_depth_a(hlf_classes, reference_class, arg, labels, input_names, p_label)
    plot_weighted_depth_r(hlf_classes, reference_class, arg, labels, input_names, p_label)
    # grouped
    #plot_weighted_depth_a(hlf_classes, reference_class, arg, labels, input_names, p_label, l=9)
    #plot_weighted_depth_r(hlf_classes, reference_class, arg, labels, input_names, p_label, l=9)
    # no dataset 1 results
    #if arg.dataset[0] == '1':
    #    plot_Etot_Einc_discrete(hlf_class, reference_class, arg)

def eval_ui_dists(source_array, reference_array, documenter, params):
    args = args_class(params)
    args.output_dir = documenter.basedir + "/eval/"
    
    if not os.path.isdir(args.output_dir):
        os.makedirs(args.output_dir)

   
    # add label in source array
    source_array = np.concatenate((source_array, np.zeros(source_array.shape[0]).reshape(-1, 1)), axis=1)
    reference_array = np.concatenate((reference_array, np.ones(reference_array.shape[0]).reshape(-1, 1)), axis=1)
    train_data, test_data, val_data = ttv_split(source_array, reference_array)

    # set up device
    args.device = torch.device('cuda:'+str(args.which_cuda) \
                                   if torch.cuda.is_available() else 'cpu')
    print("Using {}".format(args.device))

    # set up DNN classifier
    input_dim = train_data.shape[1]-1
    DNN_kwargs = {'num_layer':args.cls_n_layer, # 2
                  'num_hidden':args.cls_n_hidden, # 512
                  'input_dim':input_dim,
                  'dropout_probability':args.cls_dropout_probability} # 0
    classifier = DNN(**DNN_kwargs)
    classifier.to(args.device)
    print(classifier)
    total_parameters = sum(p.numel() for p in classifier.parameters() if p.requires_grad)

    print("{} has {} parameters".format(args.mode, int(total_parameters)))

    optimizer = torch.optim.Adam(classifier.parameters(), lr=args.cls_lr)

    train_data = TensorDataset(torch.tensor(train_data, dtype=torch.get_default_dtype()).to(args.device))
    test_data = TensorDataset(torch.tensor(test_data, dtype=torch.get_default_dtype()).to(args.device))
    val_data = TensorDataset(torch.tensor(val_data, dtype=torch.get_default_dtype()).to(args.device))

    train_dataloader = DataLoader(train_data, batch_size=args.cls_batch_size, shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=args.cls_batch_size, shuffle=False)
    val_dataloader = DataLoader(val_data, batch_size=args.cls_batch_size, shuffle=False)

    train_and_evaluate_cls(classifier, train_dataloader, test_dataloader, optimizer, args)
    classifier = load_classifier(classifier, args)

    with torch.inference_mode():
        print("Now looking at independent dataset:")
        eval_acc, eval_auc, eval_JSD = evaluate_cls(classifier, val_dataloader, args,
                                                    final_eval=True,
                                                    calibration_data=test_dataloader)
    print("Final result of classifier test (AUC / JSD):")
    print("{:.4f} / {:.4f}".format(eval_auc, eval_JSD))
    with open(os.path.join(args.output_dir, 'classifier_{}_{}.txt'.format(args.mode,
                                                                        args.dataset)),
            'a') as f:
        f.write('Final result of classifier test (AUC / JSD):\n'+\
                '{:.4f} / {:.4f}\n\n'.format(eval_auc, eval_JSD))

   
########## Alternative Main ############

class args_class:
    def __init__(self, params):
        self.dataset = params.get("eval_dataset")
        self.mode = params.get("eval_mode", "all")
        self.cut = params.get("eval_cut", 0.015)
        self.energy = params.get("eval_energy", None)
        self.reference_file = params.get("eval_hdf5_file")
        self.which_cuda = 0

        self.cls_n_layer = params.get("eval_cls_n_layer", 2)
        self.cls_n_hidden = params.get("eval_cls_n_hidden", 512)
        self.cls_dropout_probability = params.get("eval_cls_dropout", 0.0)
        self.cls_lr = params.get("eval_cls_lr", 2.e-4)
        self.cls_batch_size = params.get("eval_cls_batch_size", 1000)
        self.cls_n_epochs = params.get("eval_cls_n_epochs", 100)
        self.save_mem = params.get("eval_cls_save_mem", True)

def run_from_py(sample, energy, doc, params):
    print("Running evaluation script run_from_py:")

    if not os.path.isdir(f"{doc.basedir}/eval/"):
        os.makedirs(f"{doc.basedir}/eval/")

    args = args_class(params)
    args.output_dir = doc.basedir + "/eval/"
    print("Input sample of shape: ")
    print(sample.shape)
    particle = {'1-photons': 'photon', '1-pions': 'pion',
                '2': 'electron', '3': 'electron'}[args.dataset]
    args.particle = particle

    args.min_energy = {'1-photons': 0.001, '1-pions': 0.001,
                       '2': 0.5e-3/0.033, '3': 0.5e-3/0.033}[args.dataset]

    hlf = HLF.HighLevelFeatures(particle,
                                filename='src/challenge_files/binning_dataset_{}.xml'.format(
                                    args.dataset.replace('-', '_')))
    
    #Checking for negative values, nans and infinities
    print("Checking for negative values, number of negative energies: ")
    print("input: ", (sample < 0.0).sum(), "\n")
    print("Checking for nans in the generated sample, number of nans: ")
    print("input: ", np.isnan(sample).sum(), "\n")
    print("Checking for infs in the generated sample, number of infs: ")
    print("input: ", np.isinf(sample).sum(),"\n")
    np.nan_to_num(sample, copy=False, nan=0.0, neginf=0.0, posinf=0.0)
 
    # Using a cut everywhere
    print("Using Everywhere a cut of {}".format(args.cut))
    sample[sample<args.cut] = 0.0

    # get reference folder and name of file
    args.source_dir, args.reference_file_name = os.path.split(args.reference_file)
    args.reference_file_name = os.path.splitext(args.reference_file_name)[0]

    reference_file = h5py.File(args.reference_file, 'r')
    check_file(reference_file, args, which='reference')

    reference_shower, reference_energy = extract_shower_and_energy(
        reference_file, which='reference', single_energy=args.energy,
        max_len=len(sample)
    )
    reference_shower[reference_shower<args.cut] = 0.0
    reference_hlf = HLF.HighLevelFeatures(particle,
                                              filename='src/challenge_files/binning_dataset_{}.xml'.format(
                                                  args.dataset.replace('-', '_')))
    reference_hlf.Einc = reference_energy

    args.x_scale = 'log'

    if args.mode in ['all', 'no-cls', 'avg']:
        print("Plotting average shower next to reference...")
        plot_layer_comparison(hlf, sample.mean(axis=0, keepdims=True),
                              reference_hlf, reference_shower.mean(axis=0, keepdims=True), args)
        print("Plotting average shower next to reference: DONE.\n")
        print("Plotting average shower...")
        hlf.DrawAverageShower(sample,
                              filename=os.path.join(args.output_dir,
                                                    'average_shower_dataset_{}.png'.format(
                                                        args.dataset)),
                              title="Shower average")
        if hasattr(reference_hlf, 'avg_shower'):
            pass
        else:
            reference_hlf.avg_shower = reference_shower.mean(axis=0, keepdims=True)
        hlf.DrawAverageShower(reference_hlf.avg_shower,
                              filename=os.path.join(
                                  args.output_dir,
                                  'reference_average_shower_dataset_{}.png'.format(
                                      args.dataset)),
                              title="Shower average reference dataset")
        print("Plotting average shower: DONE.\n")

        print("Plotting randomly selected reference and generated shower: ")
        hlf.DrawSingleShower(sample[:5], 
                             filename=os.path.join(args.output_dir,
                                                    'single_shower_dataset_{}.png'.format(
                                                            args.dataset)),
                             title="Single shower")
        hlf.DrawSingleShower(reference_shower[:5], 
                             filename=os.path.join(args.output_dir,
                                                    'reference_single_shower_dataset_{}.png'.format(
                                                            args.dataset)),
                             title="Reference single shower")



    if args.mode in ['all', 'no-cls', 'avg-E']:
        print("Plotting average showers for different energies ...")
        if '1' in args.dataset:
            target_energies = 2**np.linspace(8, 23, 16)
            plot_title = ['shower average at E = {} MeV'.format(int(en)) for en in target_energies]
        else:
            target_energies = 10**np.linspace(3, 6, 4)
            plot_title = []
            for i in range(3, 7):
                plot_title.append('shower average for E in [{}, {}] MeV'.format(10**i, 10**(i+1)))
        for i in range(len(target_energies)-1):
            filename = 'average_shower_dataset_{}_E_{}.png'.format(args.dataset,
                                                                   target_energies[i])
            which_showers = ((energy >= target_energies[i]) & \
                             (energy < target_energies[i+1])).squeeze()
            hlf.DrawAverageShower(sample[which_showers],
                                  filename=os.path.join(args.output_dir, filename),
                                  title=plot_title[i])
            if hasattr(reference_hlf, 'avg_shower_E'):
                pass
            else:
                reference_hlf.avg_shower_E = {}
            if target_energies[i] in reference_hlf.avg_shower_E:
                pass
            else:
                which_showers = ((reference_hlf.Einc >= target_energies[i]) & \
                             (reference_hlf.Einc < target_energies[i+1])).squeeze()
                reference_hlf.avg_shower_E[target_energies[i]] = \
                    reference_shower[which_showers].mean(axis=0, keepdims=True)

            hlf.DrawAverageShower(reference_hlf.avg_shower_E[target_energies[i]],
                                  filename=os.path.join(args.output_dir,
                                                        'reference_'+filename),
                                  title='reference '+plot_title[i])

        print("Plotting average shower for different energies: DONE.\n")

    if args.mode in ['all', 'no-cls', 'hist-p', 'hist-chi', 'hist']:
        print("Calculating high-level features for histograms ...")
        hlf.CalculateFeatures(sample)
        hlf.Einc = energy

        if reference_hlf.E_tot is None:
            reference_hlf.CalculateFeatures(reference_shower)

        print("Calculating high-level features for histograms: DONE.\n")

        if args.mode in ['all', 'no-cls', 'hist-chi', 'hist']:
            with open(os.path.join(args.output_dir, 'histogram_chi2_{}.txt'.format(args.dataset)),
                      'w') as f:
                f.write('List of chi2 of the plotted histograms,'+\
                        ' see eq. 15 of 2009.03796 for its definition.\n')
        print("Plotting histograms ...")
        if args.dataset == '1-photons':
            p_label = r'$\gamma$ DS-1'
        elif args.dataset == '1-pions':
            p_label = r'$\pi^{+}$ DS-1'
        elif args.dataset == '2':
            p_label = r'$e^{+}$ DS-2'
        else:
            p_label = r'$e^{+}$ DS-3'

        plot_histograms([hlf,], reference_hlf, args, labels, ['',], )
        plot_cell_dist([sample,], reference_shower, args, labels, ['',], p_label)
        print("Plotting histograms: DONE. \n")
    
    if args.mode in ['all', 'all-cls', 'cls-low', 'cls-high', 'cls-low-normed']:
        if args.mode in ['all', 'all-cls']:
            list_cls = ['cls-low', 'cls-high']
        else:
            list_cls = [args.mode]

        print("Calculating high-level features for classifier ...")
        
        print("Using {} as cut for the showers ...".format(args.cut))
        # set a cut on low energy voxels !only low level!
        cut = args.cut

        hlf.CalculateFeatures(sample)
        hlf.Einc = energy

        if reference_hlf.E_tot is None:
            reference_hlf.CalculateFeatures(reference_shower)
            #save_reference(reference_hlf,
            #               os.path.join(args.source_dir, args.reference_file_name + '.pkl'))

        print("Calculating high-level features for classifer: DONE.\n")
        for key in list_cls:
            if (args.mode in ['cls-low']) or (key in ['cls-low']):
                source_array = prepare_low_data_for_classifier(sample, energy, hlf, 0., cut=cut,
                                                               normed=False, single_energy=args.energy)
                reference_array = prepare_low_data_for_classifier(reference_shower, reference_energy, reference_hlf, 1., cut=cut,
                                                                  normed=False, single_energy=args.energy)
            elif (args.mode in ['cls-low-normed']) or (key in ['cls_low_normed']):
                source_array = prepare_low_data_for_classifier(sample, energy, hlf, 0., cut=cut,
                                                               normed=True, single_energy=args.energy)
                reference_array = prepare_low_data_for_classifier(reference_shower, reference_energy, reference_hlf, 1., cut=cut,
                                                                  normed=True, single_energy=args.energy)
            elif (args.mode in ['cls-high']) or (key in ['cls-high']):
                source_array = prepare_high_data_for_classifier(sample, energy, hlf, 0., cut=cut, single_energy=args.energy)
                reference_array = prepare_high_data_for_classifier(reference_shower, reference_energy, reference_hlf, 1., cut=cut,
                                                                    single_energy=args.energy)

            train_data, test_data, val_data = ttv_split(source_array, reference_array)

            # set up device
            args.device = torch.device('cuda:'+str(args.which_cuda) \
                                       if torch.cuda.is_available() else 'cpu')
            print("Using {}".format(args.device))

            # set up DNN classifier
            input_dim = train_data.shape[1]-1
            DNN_kwargs = {'num_layer':args.cls_n_layer, # 2
                          'num_hidden':args.cls_n_hidden, # 512
                          'input_dim':input_dim,
                          'dropout_probability':args.cls_dropout_probability} # 0
            classifier = DNN(**DNN_kwargs)
            classifier.to(args.device)
            print(classifier)
            total_parameters = sum(p.numel() for p in classifier.parameters() if p.requires_grad)

            print("{} has {} parameters".format(args.mode, int(total_parameters)))

            optimizer = torch.optim.Adam(classifier.parameters(), lr=args.cls_lr)

            if args.save_mem:
                train_data = TensorDataset(torch.tensor(train_data, dtype=torch.get_default_dtype()))
                test_data = TensorDataset(torch.tensor(test_data, dtype=torch.get_default_dtype()))
                val_data = TensorDataset(torch.tensor(val_data, dtype=torch.get_default_dtype()))
            else:
                train_data = TensorDataset(torch.tensor(train_data, dtype=torch.get_default_dtype()).to(args.device))
                test_data = TensorDataset(torch.tensor(test_data, dtype=torch.get_default_dtype()).to(args.device))
                val_data = TensorDataset(torch.tensor(val_data, dtype=torch.get_default_dtype()).to(args.device))

            train_dataloader = DataLoader(train_data, batch_size=args.cls_batch_size, shuffle=True)
            test_dataloader = DataLoader(test_data, batch_size=args.cls_batch_size, shuffle=False)
            val_dataloader = DataLoader(val_data, batch_size=args.cls_batch_size, shuffle=False)

            train_and_evaluate_cls(classifier, train_dataloader, test_dataloader, optimizer, args)
            classifier = load_classifier(classifier, args)

            with torch.inference_mode():
                print("Now looking at independent dataset:")
                eval_acc, eval_auc, eval_JSD = evaluate_cls(classifier, val_dataloader, args,
                                                            final_eval=True,
                                                            calibration_data=test_dataloader)
            print("Final result of classifier test (AUC / JSD):")
            print("{:.4f} / {:.4f}".format(eval_auc, eval_JSD))
            with open(os.path.join(args.output_dir, 'classifier_{}_{}_{}.txt'.format(args.mode,
                                                                            key, args.dataset)),
                      'a') as f:
                f.write('Final result of classifier test (AUC / JSD):\n'+\
                        '{:.4f} / {:.4f}\n\n'.format(eval_auc, eval_JSD))



########## Main ##########

def main(raw_args=None):
    parser = define_parser()
    args = parser.parse_args(raw_args)

    if not os.path.isdir(args.output_dir):
        os.makedirs(args.output_dir)
        
    # check_file(source_file, args, which='input')
    particle = {'1-photons': 'photon', '1-pions': 'pion',
                '2': 'electron', '3': 'electron'}[args.dataset]
    args.particle = particle
    # minimal readout per voxel, ds1: from Michele, ds2/3: 0.5 keV / 0.033 scaling factor
    args.min_energy = {'1-photons': 0.001, '1-pions': 0.001,
                       '2': 0.5e-3/0.033, '3': 0.5e-3/0.033}[args.dataset]

    # get the path of all  the input files to be evaluated
    list_inputs = glob(args.input_file+'/*samples*.hdf5')
    showers = []
    energies = []
    hlfs = []
    
    # Using a cut everywhere
    print("Using Everywhere a cut of {}".format(args.cut))

    for n, file in enumerate(list_inputs):
        with h5py.File(file, 'r') as source_file:
            hlfs.append( HLF.HighLevelFeatures(
                particle, filename='src/challenge_files/binning_dataset_{}.xml'.format(
                                        args.dataset.replace('-', '_'))
                )
            )
            
            shower, energy = extract_shower_and_energy(source_file, which='input', single_energy=args.energy)
        
        #Checking for negative values, nans and infinities
        print(f"Cheking input file: {file}")
        print("Checking for negative values, number of negative energies: ")
        print("input: ", (shower < 0.0).sum(), "\n")
        print("Checking for nans in the generated sample, number of nans: ")
        print("input: ", np.isnan(shower).sum(), "\n")
        print("Checking for infs in the generated sample, number of infs: ")
        print("input: ", np.isinf(shower).sum(),"\n")
        np.nan_to_num(shower, copy=False, nan=0.0, neginf=0.0, posinf=0.0)
 
        shower[shower<args.cut] = 0.0
        showers.append(shower), energies.append(energy)

    with h5py.File(args.reference_file, 'r') as reference_file:
        # check_file(reference_file, args, which='reference')
        reference_shower, reference_energy = extract_shower_and_energy(
            reference_file, which='reference', single_energy=args.energy
        )
    reference_shower[reference_shower<args.cut] = 0.0

    #if os.path.exists(os.path.join(args.source_dir, args.reference_file_name + '.pkl')):
    #   print("Loading .pkl reference")
    #    reference_hlf = load_reference(os.path.join(args.source_dir,
    #                                                args.reference_file_name + '.pkl'))
    #else:
    print(f'{args.mode=}')
    print("Computing .pkl reference")
    reference_hlf = HLF.HighLevelFeatures(particle,
                                              filename='src/challenge_files/binning_dataset_{}.xml'.format(
                                                  args.dataset.replace('-', '_')))

    print('Loaded HLF object')
    reference_hlf.Einc = reference_energy
    #save_reference(reference_hlf,
    #                 os.path.join(args.source_dir, args.reference_file_name + '.pkl'))

    args.x_scale = 'log'

    # evaluations:
    if args.mode in ['all', 'no-cls', 'avg']:
        for n, file in enumerate(list_inputs):
            input_name = os.path.basename(file)
            print(f"Plotting averages for input: {input_name}")
            print("Plotting average shower next to reference...")
            plot_layer_comparison(hlfs[n], showers[n].mean(axis=0, keepdims=True),
                                  reference_hlf, reference_shower.mean(axis=0, keepdims=True),
                                  args, input_name)
            print("Plotting average shower next to reference: DONE.\n")
            print("Plotting average shower...")
            hlfs[n].DrawAverageShower(showers[n],
                                  filename=os.path.join(args.output_dir,
                                                        'average_shower_dataset_{}_{}.png'.format(
                                                            args.dataset, input_name)),
                                  title="Shower average")
            if hasattr(reference_hlf, 'avg_shower'):
                pass
            else:
                reference_hlf.avg_shower = reference_shower.mean(axis=0, keepdims=True)
                #save_reference(reference_hlf,
                #               os.path.join(args.source_dir, args.reference_file_name + '.pkl'))
            reference_hlf.DrawAverageShower(reference_hlf.avg_shower,
                                  filename=os.path.join(
                                      args.output_dir,
                                      'reference_average_shower_dataset_{}.png'.format(
                                          args.dataset)),
                                  title="Shower average reference dataset")
            print("Plotting average shower: DONE.\n")

            print("Plotting randomly selected reference and generated shower: ")
            hlfs[n].DrawSingleShower(showers[n][:5], 
                                 filename=os.path.join(args.output_dir,
                                                        'single_shower_dataset_{}_{}.png'.format(
                                                                args.dataset, input_name)),
                                 title="Single shower")
        reference_hlf.DrawSingleShower(reference_shower[:5], 
                            filename=os.path.join(args.output_dir,
                                                'reference_single_shower_dataset_{}.png'.format(
                                                            args.dataset)),
                            title="Reference single shower")

    if args.mode in ['all', 'no-cls', 'avg-E']:
        for n, file in enumerate(list_inputs):
            input_name = os.path.basename(file)
            print(f"Plotting average showers for different energies for input {input_name}...")
            if '1' in args.dataset:
                target_energies = 2**np.linspace(8, 23, 16)
                plot_title = ['shower average at E = {} MeV'.format(int(en)) for en in target_energies]
            else:
                target_energies = 10**np.linspace(3, 6, 4)
                plot_title = []
                for i in range(3, 7):
                    plot_title.append('shower average for E in [{}, {}] MeV'.format(10**i, 10**(i+1)))
            for i in range(len(target_energies)-1):
                filename = 'average_shower_dataset_{}_E_{}_{}.png'.format(args.dataset,
                                                                    target_energies[i], input_name)
                which_showers = ((energy >= target_energies[i]) & \
                                 (energy < target_energies[i+1])).squeeze()
                hlfs[n].DrawAverageShower(showers[n][which_showers],
                                      filename=os.path.join(args.output_dir, filename),
                                      title=plot_title[i])
                if hasattr(reference_hlf, 'avg_shower_E'):
                    pass
                else:
                    reference_hlf.avg_shower_E = {}
                if target_energies[i] in reference_hlf.avg_shower_E:
                    pass
                else:
                    which_showers = ((reference_hlf.Einc >= target_energies[i]) & \
                                 (reference_hlf.Einc < target_energies[i+1])).squeeze()
                    reference_hlf.avg_shower_E[target_energies[i]] = \
                        reference_shower[which_showers].mean(axis=0, keepdims=True)
                    #save_reference(reference_hlf,
                    #               os.path.join(args.source_dir, args.reference_file_name + '.pkl'))

                reference_hlf.DrawAverageShower(reference_hlf.avg_shower_E[target_energies[i]],
                                filename=os.path.join(args.output_dir,
                                                        'reference_'+filename),
                                title='reference '+plot_title[i])

            print("Plotting average shower for different energies: DONE.\n")

    if args.mode in ['all', 'no-cls', 'hist-p', 'hist-chi', 'hist']:
        print("Calculating high-level features for histograms ...")
        if reference_hlf.E_tot is None:
            reference_hlf.CalculateFeatures(reference_shower)
            #save_reference(reference_hlf,
            #               os.path.join(args.source_dir, args.reference_file_name + '.pkl'))
        input_names = []
        for n, file in enumerate(list_inputs):
            hlfs[n].CalculateFeatures(showers[n])
            hlfs[n].Einc = energies[n]

            input_name = os.path.basename(file)
            input_names.append(input_name)
            with open(os.path.join(args.output_dir, 'histogram_chi2_{}_{}.txt'
                                    .format(args.dataset, input_name)),'w') as f:
                f.write(f'List of chi2 of the plotted histograms of {input_name},'+\
                        ' see eq. 15 of 2009.03796 for its definition.\n')
        
        print("Plotting histograms ...")
        if args.dataset == '1-photons':
            p_label = r'$\gamma$ DS-1'
        elif args.dataset == '1-pions':
            p_label = r'$\pi^{+}$ DS-1'
        elif args.dataset == '2':
            p_label = r'$e^{+}$ DS-2'
        else:
            p_label = r'$e^{+}$ DS-3'

        plot_histograms(hlfs, reference_hlf, args, labels, input_names, p_label)
        plot_cell_dist(showers, reference_shower, args, labels, input_names, p_label)
        print("Plotting histograms: DONE. \n")

    print('at classification branch')
    if args.mode in ['all', 'all-cls', 'cls-low', 'cls-high', 'cls-low-normed']:
        print('in classification branch')
        if args.mode in ['all', 'all-cls']:
            list_cls = ['cls-low', 'cls-high']
        else:
            list_cls = [args.mode]

        for n, file in enumerate(list_inputs):
            input_name = os.path.basename(file)
            print("Calculating high-level features for classifier for input {input_name}...")
        
            print("Using {} as cut for the showers ...".format(args.cut))
            # set a cut on low energy voxels !only low level!
            cut = args.cut

            hlfs[n].CalculateFeatures(showers[n])
            hlfs[n].Einc = energies[n]

            if reference_hlf.E_tot is None:
                reference_hlf.CalculateFeatures(reference_shower)
            #save_reference(reference_hlf,
            #               os.path.join(args.source_dir, args.reference_file_name + '.pkl'))

            print("Calculating high-level features for classifer: DONE.\n")
            for key in list_cls:
                if (args.mode in ['cls-low']) or (key in ['cls-low']):
                    source_array = prepare_low_data_for_classifier(showers[n], energies[n], hlfs[n], 0., cut=cut,
                                                               normed=False, single_energy=args.energy)
                    reference_array = prepare_low_data_for_classifier(reference_shower, reference_energy, reference_hlf, 1., cut=cut,
                                                                  normed=False, single_energy=args.energy)
                elif (args.mode in ['cls-low-normed']) or (key in ['cls-low-normed']):
                    source_array = prepare_low_data_for_classifier(showers[n], energies[n], hlfs[n], 0., cut=cut,
                                                               normed=True, single_energy=args.energy)
                    reference_array = prepare_low_data_for_classifier(reference_shower, reference_energy, reference_hlf, 1., cut=cut,
                                                                  normed=True, single_energy=args.energy)
                elif (args.mode in ['cls-high']) or (key in ['cls-high']):
                    source_array = prepare_high_data_for_classifier(showers[n], energies[n], hlfs[n], 0., cut=cut, single_energy=args.energy)
                    reference_array = prepare_high_data_for_classifier(reference_shower, reference_energy, reference_hlf, 1., cut=cut,
                                                                    single_energy=args.energy)

                train_data, test_data, val_data = ttv_split(source_array, reference_array)

                # set up device
                args.device = torch.device('cuda:'+str(args.which_cuda) \
                                       if torch.cuda.is_available() and not args.no_cuda else 'cpu')
                print("Using {}".format(args.device))

                # set up DNN classifier
                input_dim = train_data.shape[1]-1
                DNN_kwargs = {'num_layer':args.cls_n_layer,
                              'num_hidden':args.cls_n_hidden,
                              'input_dim':input_dim,
                              'dropout_probability':args.cls_dropout_probability}
                classifier = DNN(**DNN_kwargs)
                classifier.to(args.device)
                print(classifier)
                total_parameters = sum(p.numel() for p in classifier.parameters() if p.requires_grad)

                print("{} has {} parameters".format(args.mode, int(total_parameters)))

                optimizer = torch.optim.Adam(classifier.parameters(), lr=args.cls_lr)

                if args.save_mem:
                    train_data = TensorDataset(torch.tensor(train_data, dtype=torch.get_default_dtype()))
                    test_data = TensorDataset(torch.tensor(test_data, dtype=torch.get_default_dtype()))
                    val_data = TensorDataset(torch.tensor(val_data, dtype=torch.get_default_dtype()))
                else:
                    train_data = TensorDataset(torch.tensor(train_data, dtype=torch.get_default_dtype()).to(args.device))
                    test_data = TensorDataset(torch.tensor(test_data, dtype=torch.get_default_dtype()).to(args.device))
                    val_data = TensorDataset(torch.tensor(val_data, dtype=torch.get_default_dtype()).to(args.device))

                train_dataloader = DataLoader(train_data, batch_size=args.cls_batch_size, shuffle=True)
                test_dataloader = DataLoader(test_data, batch_size=args.cls_batch_size, shuffle=False)
                val_dataloader = DataLoader(val_data, batch_size=args.cls_batch_size, shuffle=False)

                train_and_evaluate_cls(classifier, train_dataloader, test_dataloader, optimizer, args)
                classifier = load_classifier(classifier, args)

                with torch.inference_mode():
                    print("Now looking at independent dataset:")
                    eval_acc, eval_auc, eval_JSD = evaluate_cls(classifier, val_dataloader, args,
                                                            final_eval=True,
                                                            calibration_data=test_dataloader)
                print(f"Final result of classifier test {key} for {input_name} (AUC / JSD):")
                print("{:.4f} / {:.4f}".format(eval_auc, eval_JSD))
                with open(os.path.join(args.output_dir, 'classifier_{}_{}_{}_{}.txt'
                                       .format(args.mode,key, args.dataset, input_name)),'a') as f:
                    f.write('Final result of classifier test (AUC / JSD):\n'+\
                            '{:.4f} / {:.4f}\n\n'.format(eval_auc, eval_JSD))


if __name__ == '__main__':
    main()
