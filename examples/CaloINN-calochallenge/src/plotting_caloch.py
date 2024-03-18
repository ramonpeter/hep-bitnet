import numpy as np
import torch
from glob import glob
import yaml
import os

import evaluate

if __name__ == '__main__':
    
    dirs = glob('../../temp/detector_flow_final_samples/ds1-photons/')

    mode = 'hist'
    for i in dirs:
        #with open(i+'params.yaml') as f:
        #    params = yaml.load(f, Loader=yaml.FullLoader)
        #reference = params['val_data_path']
        #ref_path, _  = os.path.split(params['val_data_path'])
        reference = "/remote/gpu06/favaro/datasets/calo_challenge/gamma_data_2.hdf5"
        dataset = '1-photons'
        #reference = ref_path+'/dataset_2_2.hdf5'
        #dataset = params['eval_dataset']
        single_energy = None
        #single_energy = params.get('single_energy', None)
        #single_energy = 262144
        print("Evaluating directory:")
        print(i)
        if single_energy is not None:
            if np.log2(single_energy) > 18:
                batch = 50
            else:
                batch = 1000
            evaluate.main(f"-i {i}/inn_samples.hdf5 -r {reference} -m {mode} -d {dataset} --output_dir {i}/eval/final/ --cut 1.0e0 --energy {single_energy} --cls_batch_size {batch}".split())
        else:
            batch = 1000
            evaluate.main(f"-i {i}/inn_samples.hdf5 -i2 {i}/vaeinn_samples.hdf5 -r {reference} -m {mode} -d {dataset} --output_dir {i}/eval/final/ --cut 1e0 --cls_batch_size {batch}".split())

