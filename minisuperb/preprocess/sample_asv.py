import argparse
import pathlib
import random
import random

import tqdm
from pathlib import Path
from librosa.util import find_files
from joblib.parallel import Parallel, delayed
from torchaudio.sox_effects import apply_effects_file
import os

EFFECTS = [
    ["channels", "1"],
    ["rate", "16000"],
    ["gain", "-3.0"],
    ["silence", "1", "0.1", "0.1%", "-1", "0.1", "0.1%"],
]

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--total_minutes', type=int, default=600)

    parser.add_argument('--target_dir', type=str)

    parser.add_argument('--seed', type=int, default=179)

    parser.add_argument('--root', type=str, help='Root directory of voxceleb1')

    parser.add_argument('-o', '--output_path', default='downstream/sv_voxceleb1/dev_meta_data/mini_train.txt', type=str, help='Path to store output', required=False)

    args = parser.parse_args()

    return args

if __name__ == '__main__':
    args = get_args()
    random.seed(args.seed)
    all_speakers = []
    def trimmed_length(path):
        wav_sample, _ = apply_effects_file(path, EFFECTS)
        wav_sample = wav_sample.squeeze(0)
        length = wav_sample.shape[0]
        return length/16000
    
    wav_paths = find_files(pathlib.Path(args.root))
    wav_lengths = Parallel(n_jobs=12)(delayed(trimmed_length)(path) for path in tqdm.tqdm(wav_paths, desc="Preprocessing"))
    wav_tags = [Path(path).parts[-3:] for path in wav_paths]
    wav_paths = [os.path.join(*tags) for tags in wav_tags]

    path_info = {}
    total_length=0
    for path,length, tag in zip(wav_paths,wav_lengths, wav_tags):
        speaker_id,wav_id, wav_sample_id = tag
        if speaker_id not in path_info.keys():
            path_info[speaker_id]={}
        if wav_id not in path_info[speaker_id].keys():
            path_info[speaker_id][wav_id]={'path':[],'length':[]}
        path_info[speaker_id][wav_id]['path'].append(path)
        path_info[speaker_id][wav_id]['length'].append(length)
        total_length += length


    sample_paths=[]
    sample_lengths=[]
    sample_time=0
    for speaker_id, videos in path_info.items():
        video_list = list(videos.keys())
        spkeaker_time = 0
        random.shuffle(video_list)
        for video_id in video_list:
            audio_num = len(videos[video_id]['path'])
            i = random.choice(range(audio_num))
            fname, length = videos[video_id]['path'][i], videos[video_id]['length'][i]
            
            sample_paths.append(fname)
            sample_lengths.append(length)
            spkeaker_time+=length
            sample_time+=length
            if spkeaker_time > 25:
                break
        if sample_time > args.total_minutes*60:
            break

    print(f'[Data down sampling] {len(wav_paths)} -> {len(sample_paths)}')
    print(f'[Total time down sampling] {total_length} -> {sample_time}')
    with open(args.output_path,"w") as f:
        for path in sample_paths:
            f.write(f"{path}\n")
    
