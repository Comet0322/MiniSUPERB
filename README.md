# MiniSUPERB
 **NOTE: This repository is still under development. Some features may not be fully functional.**

MiniSUPERB is a proxy dataset for [SUPERB and the SUPERB Challenge](https://superbbenchmark.org/). It provides a simplified and accessible way to evaluate SSL speech models.

The following diagram provides an intuitive illustration of how MiniSUPERB accelerates the evaluation process for SSL speech models:
![Evaluation framework compairson](static/diagram.png)

The figure shows how our results approximate the model rankings of the SUPERB Challenge:
<img src="static/score.png" width="600">

For more details, please refer to the [original paper](https://arxiv.org/abs/2305.19011).

## Environment compatibilities

The project was developed using the following environments.

| Env | versions |
| --- | --- |
| os  | `ubuntu-20.04` |
| python | `3.10` |
| pytorch | `1.12.1` |

## Introduction and Usages

MiniSUPERB supports four downstream tasks:
- Automatic Speech Recognition (ASR)
- Speaker Idendification (SID)
- Speech Enhancement (SE)
- Source Separation (SS)

The following upstream models are supported:
| Models       | Upstream Model Name                      | Paper                                     |
|--------------|------------------------------------------|-------------------------------------------|
| WavLM        | wavlm_base, wavlm_base_plus, wavlm_large | [arxiv](https://arxiv.org/abs/2110.13900) |
| HuBERT       | hubert_base, hubert_large_ll60k          | [arxiv](https://arxiv.org/abs/2106.07447) |
| Wav2Vec 2.0  | wav2vec2, wav2vec2_large_ll60k           | [arxiv](https://arxiv.org/abs/2006.11477) |
| Modified-CPC | modified_cpc                             | [arxiv](https://arxiv.org/abs/2002.02848) |
| TERA         | tera                                     | [arxiv](https://arxiv.org/abs/2007.06028) |
| DeCoAR 2.0   | decoar2                                  | [arxiv](https://arxiv.org/abs/2012.06659) |
| Filter Bank  | fbank, fbank_no_cmvn (used for SID)      |                                           |

## Usage
### Prepare data
#### ASR

1. Download [librispeech_finetuning.tgz] (https://github.com/facebookresearch/libri-light/blob/main/data_preparation/README.md) and dev-clean, and test-clean from [LibriSpeech](https://www.openslr.org/12).

2. Unzip and check the prepared file structure
    ```bash
    DataStorage
    └── LibriSpeech/
        ├── librispeech_finetuning/
        ├── dev-clean/
        └── test-clean/
    ```

#### SID
1. Download dataset from [Voxceleb1](https://www.robots.ox.ac.uk/~vgg/data/voxceleb/vox1.html) and unzip them.
    ```bash
    voxceleb1_root="DataStorage/VoxCeleb1/"
    mkdir -p $voxceleb1_root/dev
    mkdir -p $voxceleb1_root/test

    # prepare dev
    cd $voxceleb1_root/dev/
    wget https://thor.robots.ox.ac.uk/~vgg/data/voxceleb/vox1a/vox1_dev_wav_partaa
    wget https://thor.robots.ox.ac.uk/~vgg/data/voxceleb/vox1a/vox1_dev_wav_partab
    wget https://thor.robots.ox.ac.uk/~vgg/data/voxceleb/vox1a/vox1_dev_wav_partac
    wget https://thor.robots.ox.ac.uk/~vgg/data/voxceleb/vox1a/vox1_dev_wav_partad
    cat vox1_dev* > vox1_dev_wav.zip
    unzip vox1_dev_wav.zip

    # prepare test
    cd $voxceleb1_root/test/
    wget https://thor.robots.ox.ac.uk/~vgg/data/voxceleb/vox1a/vox1_test_wav.zip
    unzip vox1_test_wav.zip
    ```
2. Check prepared file structure
    ```bash
    DataStorage
    └── Voxceleb1/
        ├── dev/
        │   └── wav/
        │       └──Speaker id folders
        └── test/
            └── wav/
                └──Speaker id folders
    ```
#### SE
1. Download Voicebank-DEMAND dataset prepared by [s3prl](https://github.com/s3prl/s3prl) 
    ```bash
    wget http://140.112.21.28:9000/noisy-vctk-16k.zip
    unzip noisy-vctk-16k.zip
    ```

2. Check the unzipped voicebank directory structure

    ```bash
    DataStorage
        └── noisy-vctk-16k/
            ├── clean_testset_wav_16k/
            ├── clean_trainset_28spk_wav_16k/
            ├── noisy_testset_wav_16k/
            ├── noisy_trainset_28spk_wav_16k/
            ├── testset_txt/
            └── trainset_28spk_txt/
    ```
#### SS
1. Simulate Libri2Mix data for source separation. For source separation, we only need 16kHz and min condition. 
**Make sure that SoX is installed on your machine** 

    ```bash
    # Download the script and simulate Libri2Mix dataset
    git clone https://github.com/s3prl/LibriMix.git
    cd LibriMix 
    ./generate_librimix_ss.sh DataStorage
    ```
2. Check the unzipped voicebank directory structure
    ```bash
    DataStorage
        └── Libri2Mix/
            └── wav16k/
                └── min/
                    ├── train-100/
                    ├── dev/
                    ├── test/
                    └── metadata/
    ```

### SSL Model Evaluation
Start a new downstream training experiment with the following command:

```bash
cd minisuperb

# To evaluate a model on ASR:
bash asr.sh UpstreamModelName DataStorage

# To evaluate a model on SID:
bash sid.sh UpstreamModelName DataStorage

# SE, SS are still under development
# To evaluate a model on SE:
bash se.sh UpstreamModelName DataStorage

# To evaluate a model on SS):
bash ss.sh UpstreamModelName DataStorage
```

## Installation

1. Install **sox** on your OS

    For Linux :
    ```
    conda install -c conda-forge sox
    ```
2. Install dependencies `pip install -e ".[all]"`

## Features Under Development

    1. Support for custom upstream models 
    2. Provide download links for sampled datasets
    3. Evaluation Scripts for Speech Enhancement (SE) and Source Separation (SS)
    4. Pipeline to calculate MiniSUPERB score for custom SSL models.

## License

The majority of this project is licensed under the Apache License version 2.0, however all the files authored by Facebook, Inc. (which have explicit copyright statement on the top) are licensed under CC-BY-NC.
