import os
from tqdm import tqdm
from glob import glob
import torchaudio
import pandas as pd
import argparse

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, help='Root directory of voxceleb1')

    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--metadata', type=str, default='downstream/voxceleb1/veri_test_class.txt')
    parser.add_argument('-o', '--output_path', default='downstream/voxceleb1/mini_veri_test_class.txt', type=str, help='Path to store output', required=False)
    parser.add_argument('-k', '--k_shot', default=1, type=int, required=False)

    args = parser.parse_args()

    return args

    
if __name__ == '__main__':
    args = get_args()
    df = pd.read_csv("downstream/voxceleb1/veri_test_class.txt", sep=" ", header=None)

    count = 0
    length_list = []
    split_list = []
    path_list = []
    spk_list = []
    for i, row in tqdm(df.iterrows(), total=len(df)):
        split, wav_path = row
        path = os.path.join(args.root, '*/wav', wav_path.strip())
        try:
            length_list.append(torchaudio.info(glob(path)[0]).num_frames/16000)
            split_list.append(split)
            path_list.append(wav_path.strip())
            spk_list.append(wav_path.split("/")[0])
        except:
            pass
    
    new_df = pd.DataFrame({"split": split_list, "path": path_list, "length": length_list, "spk": spk_list})
    train_df = new_df[new_df["split"] == 1]
    dev_df = new_df[new_df["split"] == 2]
    test_df = new_df[new_df["split"] == 3]
    part_train_df = train_df.groupby('spk', group_keys=False).apply(lambda x: x.sample(args.k_shot, random_state=args.seed))
    # part_train_df = train_df.groupby('spk', group_keys=False).apply(lambda x: x.sort_values(by="length", ascending=False).iloc[:args.k_shot])
    
    new_total_df = pd.concat([part_train_df, dev_df, test_df]).reset_index(drop=True)
    new_total_df[["split", "path"]].to_csv(args.output_path, sep=" ", header=None, index=False)
    print(part_train_df.shape[0])
    print(f"write sampled metadata to {args.output_path}")
    print("original hours: ", train_df.length.sum()/3600)
    print("sampled hours: ", part_train_df.length.sum()/3600)
