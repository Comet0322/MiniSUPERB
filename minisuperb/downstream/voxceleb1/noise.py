import os
from glob import glob
from tqdm import tqdm
import pandas as pd
import torch
import torchaudio
import concurrent.futures

from speechbrain.pretrained import EncoderClassifier
import torch
import torchaudio
from torchaudio.prototype.pipelines import SQUIM_OBJECTIVE as bundle
from tqdm import tqdm

model = bundle.get_model()
model.to("cuda")

metadata = "/home/jerrymark/data/Few-Shot-SUPERB/s3prl/downstream/voxceleb1/veri_test_class.txt"
root = "/home/jerrymark/data/Few-Shot-SUPERB/data/voxceleb1"
output_path = "/home/jerrymark/data/Few-Shot-SUPERB/data/extracted_feats/voxceleb1_x-vector"


df = pd.read_csv(metadata, sep=" ", header=None)


def calculate_sisdr(file):
    path = glob(f"{root}/*/wav/{file}")[0]
    wav, sr = torchaudio.load(path)
    assert sr == bundle.sample_rate
    with torch.no_grad():
        out = model(wav)
    sisdr = out[2].item()

    return file, sisdr


file_list = df.iloc[:, 1].tolist()
num_threads = 4

# Create a ThreadPoolExecutor with the specified number of threads
with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
    # Use the executor.map() function to parallelize the calculation across multiple threads
    results = list(tqdm(executor.map(calculate_sisdr, file_list), total=len(file_list)))
