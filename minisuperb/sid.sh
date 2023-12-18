#!/bin/bash
model=$1
data_dir=$2
seed=8830
experiment=voxceleb1-$model
extracted_path=$data_dir/extracted_feats/sid-$model


extract_overide="config.downstream_expert.datarc.train_batch_size=1,,config.downstream_expert.datarc.eval_batch_size=1"

echo "Stage 1: Extracting features"
python3 run_downstream.py -m extract \
    -d voxceleb1 \
    -u $model \
    -n $experiment \
    -t "all" \
    -o $extract_overide",,config.downstream_expert.datarc.file_path=$data_dir/voxceleb1" \
    --extracted_path $extracted_path \
    --extract_scene_feature

echo "Stage 2: Training"
python3 run_downstream.py -m train \
    -d voxceleb1 \
    -u $model \
    -n $experiment \
    -o "config.downstream_expert.datarc.file_path=$data_dir/voxceleb1" \
    --extracted_path $extracted_path \
    --use_extracted_feature \
    --extract_scene_feature \
    --upstream_feature_normalize