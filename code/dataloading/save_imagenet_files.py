import argparse
import json
import os
from tqdm import tqdm
import pyarrow as pa
import h5py
import numpy as np

def save_files(args):
    syn_to_class = {}

    with open(os.path.join(args.data_root, "imagenet_class_index.json"), "rb") as f:
        json_file = json.load(f)
        for class_id, v in json_file.items():
            syn_to_class[v[0]] = int(class_id)

    with open(os.path.join(args.data_root, "ILSVRC2012_val_labels.json"), "rb") as f:
        val_to_syn = json.load(f)


    splits = ["train", "val"]
    train_samples = []
    train_targets = []

    val_samples = []
    val_targets = []
    for split in splits:
        samples_dir = os.path.join(args.data_root, "ILSVRC/Data/CLS-LOC", split)
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

    if args.dset_type == "arrow":
        save_arrow(args, splits, train_samples, train_targets, val_samples, val_targets)
    elif args.dset_type == "h5":
        save_h5(splits, train_samples, train_targets, val_samples, val_targets)
    else:
        raise ValueError("Invalid dset_type")


def save_arrow(args, splits, train_samples, train_targets, val_samples, val_targets):

    binary_t = pa.binary()
    uint16_t = pa.uint16()

    schema = pa.schema([
        pa.field('image_data', binary_t),
        pa.field('label', uint16_t),
    ])

    for split in splits:
        if split == "train":
            samples = train_samples
            targets = train_targets
        else:
            samples = val_samples
            targets = val_targets
        with pa.OSFile(
                os.path.join(args.target_folder, f'ImageNet-{split}.arrow'),
                'wb',
        ) as f:
            with pa.ipc.new_file(f, schema) as writer:
                for (sample, label) in zip(samples, targets):
                    with open(sample, 'rb') as f:
                        img_string = f.read()

                    image_data = pa.array([img_string], type=binary_t)
                    label = pa.array([label], type=uint16_t)

                    batch = pa.record_batch([image_data, label], schema=schema)

                    writer.write(batch)

def save_h5(splits, train_samples, train_targets, val_samples, val_targets):
    with h5py.File(os.path.join(args.target_folder, 'ImageNet.h5'), "w") as f:
        
        for split in splits:
            if split == "train":
                samples = train_samples
                targets = train_targets
            else:
                samples = val_samples
                targets = val_targets
                
            group = f.create_group(split)
            dt_sample = h5py.vlen_dtype(np.dtype(np.uint8))
            dt_target = np.dtype('int16')
            dset = group.create_dataset(
                            'images',
                            (len(samples),),
                            dtype=dt_sample,
                        )
            dtargets = group.create_dataset(
                    'targets',
                    (len(samples),),
                    dtype=dt_target,
                )
            
            for idx, (sample, target) in tqdm(enumerate(zip(samples, targets))):        
                with open(sample, 'rb') as f:

                    img_string = f.read()
                    
                    dset[idx] = np.array(list(img_string), dtype=np.uint8)
                    dtargets[idx] = target
                



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default="/p/scratch/training2402/data/")
    parser.add_argument('--dset_type', choices=['h5', 'arrow'])
    parser.add_argument('--target_folder', type=str, required=True)
    args = parser.parse_args()
    save_files(args)