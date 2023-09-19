import os
import h5py
import json
from tqdm import tqdm
import numpy as np


def create_data(g, train_samples, train_targets, split, length):
    # We create a group for the train set and the val set
    group = g.create_group(split)
    dt_sample = h5py.vlen_dtype(np.dtype(np.uint8))
    dt_target = np.dtype('int16')
    
    # Create two datasets: one for storing the images and the other for storing the targets
    dset = group.create_dataset(
                    'images',
                    (length,),
                    dtype=dt_sample,
                )
    dtargets = group.create_dataset(
            'targets',
            (length,),
            dtype=dt_target,
        )
    
    # We loop over the images paths list and targets list and store the images as binaries and targets as ints 
    for idx, (sample, target) in tqdm(enumerate(zip(train_samples, train_targets))):        
        with open(sample, 'rb') as f:
            img_string = f.read()
            
            dset[idx] = np.array(list(img_string), dtype=np.uint8)
            dtargets[idx] = target
        

  
if __name__ == "__main__":
    # First, we create 4 lists: 
    # A list that contains the images paths of the training set
    # A list that contains the images paths of the validation set 
    # A list for the train target data
    # A list for the val target data 

    root = "/p/scratch/training2324/data/"
    syn_to_class = {}

    print("read train labels")
    with open(os.path.join(root, "imagenet_class_index.json"), "rb") as f:
        json_file = json.load(f)
        for class_id, v in json_file.items():
            syn_to_class[v[0]] = int(class_id)

    print("read val labels")
    with open(os.path.join(root, "ILSVRC2012_val_labels.json"), "rb") as f:
        val_to_syn = json.load(f)

    print("read samples")
    splits = ["train", "val"]
    train_samples = []
    train_targets = []

    val_samples = []
    val_targets = []
    for split in splits:
        samples_dir = os.path.join(root, "ILSVRC/Data/CLS-LOC", split)
        for entry in os.listdir(samples_dir):
            if split == "train":
                syn_id = entry
                target = syn_to_class[syn_id]
                syn_folder = os.path.join(samples_dir, syn_id)
                for sample in os.listdir(syn_folder):
                    sample_path = os.path.join(syn_folder, sample)
                    train_samples.append(sample_path)
                    train_targets.append(target)
            elif split == "val":
                syn_id = val_to_syn[entry]
                target = syn_to_class[syn_id]
                sample_path = os.path.join(samples_dir, entry)
                val_samples.append(sample_path)
                val_targets.append(target)


    # We create an h5 file.
    # The extension can be either .hdf5 or h5
    with h5py.File('/p/scratch/training2303/data/ImageNet.h5', "w") as f:
        for split in splits:
            if split == "train":
                create_data(f, train_samples[:length], train_targets[:length], split, length)
            else :
                length = len(val_samples)
                create_data(f, val_samples, val_targets, split, length)




