#!/bin/bash
model=$1
data_dir=$2
seed=8830

experiment=asr-$model
extracted_path=$data_dir/extracted_feats/asr-$model
libri_root=$data_dir/LibriSpeech
bucket_file=downstream/asr

python3 preprocess/generate_len_for_bucket.py -i $libri_root -o $bucket_file -a .flac

extract_overide="config.downstream_expert.datarc.train_batch_size=1,,config.downstream_expert.datarc.eval_batch_size=1"
echo "Stage 1: Extracting features"
python3 run_downstream.py -m extract \
    -d asr \
    -u $model \
    -n $experiment \
    -t "all" \
    -o $extract_overide",,config.downstream_expert.datarc.libri_root=$libri_root" \
    --extracted_path $extracted_path

echo "Stage 2: Training"
python3 run_downstream.py -m train \
    -d asr \
    -u $model \
    -n $experiment \
    -o "config.downstream_expert.datarc.libri_root=$libri_root" \
    --extracted_path $extracted_path \
    --use_extracted_feature \
    --upstream_feature_normalize
