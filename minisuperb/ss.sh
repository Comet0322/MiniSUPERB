#!/bin/bash
task=separation_stft2
model=$1
data_dir=$2
seed=8830

python downstream/separation_stft2/scripts/LibriMix/data_prepare.py \
--part train-100 $data_dir/mini_libri2mix downstream/separation_stft2/datasets/Libri2Mix

python downstream/separation_stft2/scripts/LibriMix/data_prepare.py \
--part dev $data_dir/mini_libri2mix downstream/separation_stft2/datasets/Libri2Mix

python downstream/separation_stft2/scripts/LibriMix/data_prepare.py \
--part test $data_dir/mini_libri2mix downstream/separation_stft2/datasets/Libri2Mix

echo "Training with extracted features"
experiment=$task-$model
extracted_path=$data_dir/extracted_feats/ss-$model

extract_overide="config.downstream_expert.datarc.train_batch_size=1,,config.downstream_expert.datarc.eval_batch_size=1"
echo "Stage 1: Extracting features" \ &&
python3 run_downstream.py -m extract \
    -c downstream/separation_stft2/configs/cfg.yaml \
    -d $task \
    -u $model \
    -n $experiment \
    -t "all" \
    -o $extract_overide \
    --extracted_path $extracted_path

echo "Stage 2: Training"
python3 run_downstream.py -m train \
    -c downstream/separation_stft2/configs/cfg.yaml \
    -d $task \
    -u $model \
    -n $experiment \
    --extracted_path $extracted_path \
    --use_extracted_feature\
    --upstream_feature_normalize
