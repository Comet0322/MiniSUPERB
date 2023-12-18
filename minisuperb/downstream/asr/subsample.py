import os
from tqdm import tqdm
from glob import glob
import torchaudio
import pandas as pd
import argparse
import numpy as np

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, help='Root directory of voxceleb1')

    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--metadata', type=str, default='downstream/asr/len_for_bucket/train-clean-100.csv')
    parser.add_argument('--output', type=str, default='downstream/asr/len_for_bucket/train-clean-10-random.csv')
    args = parser.parse_args()

    return args

    
if __name__ == '__main__':
    args = get_args()
    df = pd.read_csv(args.metadata)

    count = 0
    split_list = []
    path_list = []
    label_list = []
    noise_list = []
    length_list = []
    for i, row in tqdm(df.iterrows(), total=len(df)):
        try:
            path_list.append(row.file_path)
            length_list.append(row.length)
            # split_list.append(split)
            label_list.append(np.nan)
            if "other" in row.file_path:
                noise = "other"
            else:
                noise = "clean"

            noise_list.append(noise)
        except:
            pass
    
    new_df = pd.DataFrame({"file_path": path_list, "length": length_list, "label": label_list, "noise": noise_list})
    # new_df = new_df.groupby('noise', group_keys=False).apply(lambda x: x.sample(24, random_state=args.seed))
    new_df = new_df.apply(lambda x: x.sample(new_df.shape[0]//10, random_state=args.seed))
    print(new_df.length.sum()/16000/3600, "hours")
    new_df[["file_path","length","label"]].to_csv(args.output)

    print(f"write sampled metadata to {args.output}")
