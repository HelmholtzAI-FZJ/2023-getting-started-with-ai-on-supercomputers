import os
import io
import time
import argparse

import torch 
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from torch.nn.utils.rnn import pad_sequence
import torchvision.transforms as T

import pandas as pd
import pyarrow as pa
from PIL import Image
from matplotlib import pyplot as plt
import spacy

spacy_eng = spacy.load("en_core_web_sm")

class Vocabulary:
    def __init__(self, freq_threshold):
        self.freq_threshold = freq_threshold
        self.itos = {0: "<PAD>", 1: "<SOS>", 2: "<EOS>", 3: "<UNK>"}
        self.stoi = {v: k for k,v in self.itos.items()}
    
    def __len__(self):
        return len(self.itos)
  
    @staticmethod
    def tokenize(text):
        return [token.text.lower() for token in spacy_eng.tokenizer(text)]
  
    def build_vocab(self, sent_list):
        freqs = {}
        idx = 4
        for sent in sent_list:
            sent = str(sent)
            for word in self.tokenize(sent):
                if word not in freqs:
                    freqs[word] = 1
                else:
                    freqs[word] += 1

                if freqs[word] == self.freq_threshold:
                    self.itos[idx] = word
                    self.stoi[word] = idx
                    idx += 1

    def numericalize(self, sents):
        tokens = self.tokenize(sents)
        return [self.stoi[token] if token in self.stoi else self.stoi["<UNK>"] 
                for token in tokens]
    

# Padding the captions according to the largest caption in the batch
class CapCollat:
    def __init__(self, pad_seq, batch_first=False):
        self.pad_seq = pad_seq
        self.batch_first = batch_first
  
    def __call__(self, batch):
        imgs = [itm[0].unsqueeze(0) for itm in batch]
        imgs = torch.cat(imgs, dim=0)

        target_caps = [itm[1] for itm in batch]
        target_caps = pad_sequence(target_caps, batch_first=self.batch_first,
                                   padding_value=self.pad_seq)
        return imgs, target_caps
    

class FlickrDataset(Dataset):
    def __init__(self, root_dir, annot, transforms=None, freq_threshold=5):
        self.root_dir = root_dir
        self.df = pd.read_csv(annot, delimiter=',')
        self.transforms = transforms
        self.img_pts = self.df['image_name']
        self.caps = self.df[' comment']
        self.vocab = Vocabulary(freq_threshold)
        self.vocab.build_vocab(self.caps.tolist())

    def __len__(self):
        return len(self.df)
  
    def __getitem__(self, idx):
        captions = self.caps[idx]
        img_pt = self.img_pts[idx]
        img = Image.open(os.path.join(self.root_dir, img_pt)).convert('RGB')

        if self.transforms is not None:
            img = self.transforms(img)

        numberized_caps = []
        numberized_caps += [self.vocab.stoi["<SOS>"]] # stoi string to index
        numberized_caps += self.vocab.numericalize(captions)
        numberized_caps += [self.vocab.stoi["<EOS>"]]

        return img, torch.tensor(numberized_caps)
    

class FlickrArrow(Dataset):
    def __init__(self, file_name, transforms=None, freq_threshold=5):
        self.data_root = file_name
        self.transforms = transforms

        
        self.arrowfile = None
        self.reader = None
        with pa.OSFile(self.data_root, 'rb') as f:
            with pa.ipc.open_file(f) as reader:
                self._len = reader.num_record_batches
                self.vocab = Vocabulary(freq_threshold)
                self.vocab.build_vocab(reader.read_pandas()['caption'].tolist())
        

    def __len__(self):
        return self._len

    def __getitem__(self, idx):
        if self.arrowfile is None:
            self.arrowfile = pa.OSFile(self.data_root, 'rb')
            self.reader = pa.ipc.open_file(self.arrowfile)

        row = self.reader.get_batch(idx)
        img_string = row['image_data'][0].as_py()
        caption = row['caption'][0].as_py()

        with io.BytesIO(img_string) as byte_stream:
            with Image.open(byte_stream) as img:
                img = img.convert("RGB")

        with io.BytesIO(caption) as byte_stream:
            string_data = caption.decode('utf-8')
  
        if self.transforms:
            img = self.transforms(img)
            
        numberized_caps = []
        numberized_caps += [self.vocab.stoi["<SOS>"]] # stoi string to index
        numberized_caps += self.vocab.numericalize(string_data)
        numberized_caps += [self.vocab.stoi["<EOS>"]]

        return img, torch.tensor(numberized_caps)



def main(args):

    transform = T.Compose([
            T.ToTensor(),
            T.Resize((256, 256))
        ])

    if args.dset_type == 'fs':
        flickr_datasets = FlickrDataset(args.file_name, "/p/scratch/training2402/data/Flickr30K/flickr30k_images/flickr30k_images/results.csv", transform)
    elif args.dset_type == 'arrow':
        flickr_datasets = FlickrArrow(args.file_name, transform)
    else:
        assert False

    pad_idx = flickr_datasets.vocab.stoi["<PAD>"]
    sampler = DistributedSampler(
        flickr_datasets,
        num_replicas=int(os.getenv('SLURM_NTASKS')),
        rank=int(os.getenv('SLURM_PROCID')),
        shuffle=args.shuffle,
    )

    dataloader = DataLoader(
        flickr_datasets,
        batch_size=128,
        num_workers=int(os.getenv('SRUN_CPUS_PER_TASK')),
        pin_memory=True,
        sampler=sampler,
        collate_fn=CapCollat(pad_seq=pad_idx, batch_first=True)
    )
    
    start_time = time.time()
    
    for _ in dataloader:
        pass
    
    end_time = time.time()

    td = (end_time - start_time)
    print(time.strftime("%H:%M:%S", time.gmtime(td)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--file_name', '-d', required=True)
    parser.add_argument('--dset_type', choices=['fs', 'h5', 'arrow'])
    parser.add_argument('--shuffle', action='store_true')
    args = parser.parse_args()
    main(args)