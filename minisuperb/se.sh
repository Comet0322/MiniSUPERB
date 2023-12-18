#!/bin/bash
task=enhancement_stft2
model=$1
data_dir=$2
seed=8830

python downstream/enhancement_stft2/scripts/Voicebank/data_prepare.py \
    $data_dir/noisy-vctk-16k downstream/enhancement_stft2/datasets/voicebank --part train
python downstream/enhancement_stft2/scripts/Voicebank/data_prepare.py \
    $data_dir/noisy-vctk-16k downstream/enhancement_stft2/datasets/voicebank --part dev
python downstream/enhancement_stft2/scripts/Voicebank/data_prepare.py \
    $data_dir/noisy-vctk-16k downstream/enhancement_stft2/datasets/voicebank --part test



echo "Training with extracted features"
experiment=$task-$model
extracted_path=$data_dir/extracted_feats/se-$model

extract_overide="config.downstream_expert.datarc.train_batch_size=1,,config.downstream_expert.datarc.eval_batch_size=1"
echo "Stage 1: Extracting features"
python3 run_downstream.py -m extract \
    -d $task \
    -u $model \
    -c downstream/enhancement_stft2/configs/cfg_voicebank.yaml \
    -n $experiment \
    -t "all" \
    -o $extract_overide \
    --extracted_path $extracted_path

echo "Stage 2: Training"
python3 run_downstream.py -m train \
    -d $task \
    -u $model \
    -c downstream/enhancement_stft2/configs/cfg_voicebank.yaml \
    -n $experiment \
    --extracted_path $extracted_path \
    --use_extracted_feature \
    --upstream_feature_normalize
