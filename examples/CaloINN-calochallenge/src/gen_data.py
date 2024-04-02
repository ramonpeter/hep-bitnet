import h5py
import argparse
import os
import yaml
import time

import numpy as np
import torch
import pandas as pd
from trainer import Trainer
from documenter import Documenter

def split_samples(file_s, fraction=0.5, name1=None, name2=None, savedir=None):
    input_file = h5py.File(file_s, 'r')

    key_len = []
    for key in input_file.keys():
        key_len.append(len(input_file[key]))
    key_len = np.array(key_len)

    assert np.all(key_len==key_len[0])

    cut_index = int(fraction * key_len[0])

    train_file = h5py.File(savedir+name1+'.hdf5', 'w')
    test_file = h5py.File(savedir+name2+'.hdf5', 'w')

    for key in input_file.keys():
        train_file.create_dataset(key, data=input_file[key][:cut_index])
        test_file.create_dataset(key, data=input_file[key][cut_index:])

    train_file.close()
    test_file.close()

def merge_samples(gen_samp, true_samp, filename=None, savedir=None):
    gen_samp = h5py.File(gen_samp, 'r')
    true_samp = h5py.File(true_samp, 'r')

    key_len_1 = []
    for key in gen_samp.keys():
        key_len_1.append(len(gen_samp[key]))
    key_len_1 = np.array(key_len_1)
    assert np.all(key_len_1 == key_len_1[0])
    key_len_2 = []
    for key in true_samp.keys():
        key_len_2.append(len(true_samp[key]))
    key_len_2 = np.array(key_len_2)
    assert np.all(key_len_2 == key_len_2[0])

    assert np.all(key_len_1 == key_len_2)

    file1_name = 'gen_samp'
    file2_name = 'true_samp'

    if filename is None:
        new_file_name = 'merged_'+ file1_name + '_' + file2_name + '.hdf5'
    else:
        new_file_name = filename+'.hdf5'
    new_file = h5py.File(savedir+new_file_name, 'w')

    shuffle_order = np.arange(key_len_1[0]+key_len_2[0])
    np.random.shuffle(shuffle_order)

    for key in gen_samp.keys():
        data1 = gen_samp[key][:]
        data2 = true_samp[key][:]
        data = np.concatenate([data1, data2])
        new_file.create_dataset(key, data=data[shuffle_order])

    truth1 = np.zeros(key_len_1[0])
    truth2 = np.ones(key_len_2[0])
    truth = np.concatenate([truth1, truth2])
    new_file.create_dataset('label', data=truth[shuffle_order])

    new_file.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--model_dir', help='dir to saved model')
    parser.add_argument('--model_name')
    parser.add_argument('--p_type', help='piplus, eplus or gamma')
    parser.add_argument('--no_cuda', action='store_true')
    parser.add_argument('--data_dir')
    parser.add_argument('--mult', default=False, action='store_true')
    parser.add_argument('--new_samp', default=False, action='store_true')

    args = parser.parse_args()

    #set dir
    model_dir = args.model_dir
    cls_train_dir = '/remote/gpu06/favaro/calo_inn/datasets/cls_data/train_cls_'+args.p_type+'.hdf5'
    cls_test_dir = '/remote/gpu06/favaro/calo_inn/datasets/cls_data/test_cls_'+args.p_type+'.hdf5'
    cls_val_dir = '/remote/gpu06/favaro/calo_inn/datasets/cls_data/val_cls_'+args.p_type+'.hdf5'

    #load model
    with open(model_dir+'params.yaml') as f:
        params = yaml.load(f, Loader=yaml.FullLoader)
    device = 'cuda:0' if not args.no_cuda else 'cpu'
    doc = Documenter(params['run_name'], existing_run=args.model_dir)
    params["train_data_path"] = args.data_dir + 'train_' + args.p_type + '.hdf5'
    params["test_data_path"] = args.data_dir + 'test_' + args.p_type + '.hdf5'
    trainer = Trainer(params, device, doc)
    trainer.load(epoch=args.model_name)

    if args.mult:
        trainer.model.enable_map()
        #get all showers
        gen_samp = trainer.generate(60000)
        merge_samples(args.model_dir+'samples.hdf5', cls_train_dir, filename='train_cls_full_'+args.p_type, savedir=args.model_dir)
        print("AVG gen. time 1: ", trainer.avg_gen_time)

        gen_samp = trainer.generate(20000)
        merge_samples(args.model_dir+'samples.hdf5', cls_test_dir, filename='test_cls_full_'+args.p_type, savedir=args.model_dir)
        print("AVG gen. time 2: ", trainer.avg_gen_time)

        gen_samp = trainer.generate(20000)
        merge_samples(args.model_dir+'samples.hdf5', cls_val_dir, filename='val_cls_full_'+args.p_type, savedir=args.model_dir)
        print("AVG gen. time 3: ", trainer.avg_gen_time)
    else:
        if args.new_samp:
            if trainer.model.bayesian:
                trainer.model.enable_map()
            gen_samp = trainer.generate(100000)
            print("AVG gen. time: ", trainer.avg_gen_time)
        split_samples(args.model_dir+'samples.hdf5', 0.6, name1='samples_train', name2='samples_testval', savedir=args.model_dir)
        split_samples(args.model_dir+'samples_testval.hdf5', 0.5, name1='samples_test', name2='samples_val', savedir=args.model_dir)

        merge_samples(args.model_dir+'samples_train.hdf5', cls_train_dir, filename='train_cls_full_'+args.p_type, savedir=args.model_dir)
        merge_samples(args.model_dir+'samples_test.hdf5', cls_test_dir, filename='test_cls_full_'+args.p_type, savedir=args.model_dir)
        merge_samples(args.model_dir+'samples_val.hdf5', cls_val_dir, filename='val_cls_full_'+args.p_type, savedir=args.model_dir)

