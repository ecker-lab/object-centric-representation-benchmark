import os
import h5py
import argparse
import numpy as np

        
# Initiate the parser
parser = argparse.ArgumentParser()
parser.add_argument('--path', type=str, default='ocrb/data/datasets/')
parser.add_argument('--dataset', type=str)
parser.add_argument('--out_path', type=str, default='ocrb/data/datasets/')
parser.add_argument('--testsets',  help='Extract VMDS test sets.', action='store_true')


def save(args, data, key):
    out_dir = os.path.join(args.out_path, args.dataset)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    out_path = os.path.join(out_dir, '{}_{}'.format(args.dataset, key))
    print('Save {}'.format(out_path))
    np.save(out_path, data)


# Read arguments from the command line
args = parser.parse_args()

if args.testsets:
    assert args.dataset == 'vmds', 'Additional testsets only for VMDS available.'

# load h5py file
hf = h5py.File(os.path.join(args.path, args.dataset + '.hdf5'), 'r')
keys = hf.keys()

for key in keys:
    if isinstance(hf[key], h5py.Group):
        if args.testsets:
            g = hf.get(key)
            keys_ = hf[key].keys()
            for k in keys_:
                data = np.array(g.get(k))
                save(args, data, k)
            
    elif isinstance(hf[key], h5py.Dataset):
        if not args.testsets:
            data = np.array(hf.get(key))
            save(args, data, key)
