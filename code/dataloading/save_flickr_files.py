import os
import pandas as pd
import pyarrow as pa
from tqdm import tqdm   

root_folder = "/p/scratch/training2402/data/Flickr30K/flickr30k_images/flickr30k_images"
csv_file = "/p/scratch/training2402/data/Flickr30K/flickr30k_images/flickr30k_images/results.csv"
target_folder = "/p/scratch/training2402/data"

root_dir = root_folder
df = pd.read_csv(csv_file, delimiter='|')

# df.loc[(df["image_name"] == "2199200615.jpg") & (df[" comment_number"] == df.loc[(df["image_name"] == "2199200615.jpg")][" comment_number"][19999]), " comment"] = "A dog runs across the grass ."
img_pts = df['image_name'].to_list()
caps = df[' comment'].to_list()

binary_t = pa.binary()

schema = pa.schema([
    pa.field('image_data', binary_t),
    pa.field('caption', binary_t),
])

with pa.OSFile(
        os.path.join(target_folder, f'flicker.arrow'),
        'wb',
) as f:
    with pa.ipc.new_file(f, schema) as writer:
        for (sample, label) in tqdm(zip(img_pts, caps)):
            with open(os.path.join(root_folder,sample), 'rb') as f:
                img_string = f.read()

            image_data = pa.array([img_string], type=binary_t)
            cap = pa.array([label], type=binary_t) 
 
            batch = pa.record_batch([image_data, cap], schema=schema)
            writer.write(batch)
