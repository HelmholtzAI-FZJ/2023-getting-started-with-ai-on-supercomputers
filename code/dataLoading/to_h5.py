import os
import h5py

from tqdm import tqdm
import numpy as np
import pickle

def create_data(g, root, train_samples, train_targets, split, length):
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
        with open(os.path.join(root, sample), 'rb') as f:
            img_string = f.read()
            
            dset[idx] = np.array(list(img_string), dtype=np.uint8)
            dtargets[idx] = target
        

  
def main():
    # First, we create 4 lists: 
    # A list that contains the images paths of the training set
    # A list that contains the images paths of the validation set 
    # A list for the train target data
    # A list for the val target data 

    root = "/p/scratch/training2324/data/"
    splits = ["train", "val"]

    with open(os.path.join(root, "train_data.pkl"), "rb") as f:
        train_data = pickle.load(f)

    train_samples = list(train_data.keys())[:1000]
    train_targets = list(train_data.values())[:1000]


    with open(os.path.join(root, "val_data.pkl"), "rb") as f:
        val_data = pickle.load(f)

    val_samples = list(val_data.keys())[:1000]
    val_targets = list(val_data.values())[:1000]



    # We create an h5 file.
    # The extension can be either .hdf5 or h5
    with h5py.File('/p/scratch/training2324/data/ImageNet1k.h5', "w") as f:
        for split in splits:
            if split == "train":
                create_data(f, root, train_samples, train_targets, split, len(train_samples))
            else :
                create_data(f, root, val_samples, val_targets, split, len(val_samples))


if __name__ == "__main__":
    main()

