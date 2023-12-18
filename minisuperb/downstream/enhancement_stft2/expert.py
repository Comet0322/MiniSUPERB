# -*- coding: utf-8 -*- #
"""*********************************************************************************************"""
#   FileName     [ expert.py ]
#   Synopsis     [ the speech enhancement downstream wrapper ]
#   Source       [ Reference some code from https://github.com/funcwj/uPIT-for-speech-separation, https://github.com/asteroid-team/asteroid and https://github.com/khhungg/BSSE-SE]
#   Author       [ Zili Huang ]
#   Copyright    [ Copyright(c), Johns Hopkins University ]
"""*********************************************************************************************"""

###############
# IMPORTATION #
###############
import os
import math
import random
import h5py
import numpy as np
from pathlib import Path
from collections import defaultdict
import librosa

# -------------#
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pack_sequence, pad_sequence
import torch.nn.functional as F

# -------------#
from .model import SepRNN
from .dataset import SeparationDataset
from asteroid.metrics import get_metrics
from .loss import EnhLoss, SISDRLoss
from minisuperb.utility.data import get_extracted_ss_dataset

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

COMPUTE_METRICS = ["si_sdr", "stoi", "pesq"]
EPS = 1e-10


def match_length(feat_list, length_list):
    assert len(feat_list) == len(length_list)
    bs = len(length_list)
    new_feat_list = []
    for i in range(bs):
        # print(feat_list[i].size(0), length_list[i])
        assert abs(feat_list[i].size(0) - length_list[i]) < 5
        if feat_list[i].size(0) == length_list[i]:
            new_feat_list.append(feat_list[i])
        elif feat_list[i].size(0) > length_list[i]:
            new_feat_list.append(feat_list[i][:length_list[i], :])
        else:
            new_feat = torch.zeros(length_list[i], feat_list[i].size(1)).to(
                feat_list[i].device)
            new_feat[:feat_list[i].size(0), :] = feat_list[i]
            new_feat_list.append(new_feat)
    return new_feat_list


class DownstreamExpert(nn.Module):
    """
    Used to handle downstream-specific operations
    eg. downstream forward, metric computation, contents to log
    """

    def __init__(self, upstream_dim, upstream_rate, downstream_expert, expdir,
                 **kwargs):
        super(DownstreamExpert, self).__init__()
        self.upstream_dim = upstream_dim
        self.upstream_rate = upstream_rate
        assert self.upstream_rate in [160, 320]
        self.datarc = downstream_expert["datarc"]
        self.loaderrc = downstream_expert["loaderrc"]
        self.modelrc = downstream_expert["modelrc"]
        self.expdir = expdir

        if kwargs["use_extracted_feature"] and kwargs["mode"] != "extract":
            assert (os.path.exists(
                os.path.join(kwargs["extracted_path"], "train"))
                    and os.path.exists(
                        os.path.join(kwargs["extracted_path"], "dev"))
                    and os.path.exists(
                        os.path.join(kwargs["extracted_path"], "test"))
                    ), "You should extract your features first!"
            dataset_cls = get_extracted_ss_dataset(
                SeparationDataset,
                kwargs['extract_to_single_file'],
            )
        else:
            dataset_cls = SeparationDataset
        dataset_cls_ger = lambda data_dir, split: dataset_cls(
            data_dir,
            rate=self.datarc['rate'],
            src=self.datarc['src'],
            tgt=self.datarc['tgt'],
            n_fft=self.datarc['n_fft'],
            hop_length=self.datarc['hop_length'],
            win_length=self.datarc['win_length'],
            window=self.datarc['window'],
            center=self.datarc['center'],
            extracted_path=kwargs["extracted_path"],
            split_name=split)
        self.train_dataset = dataset_cls_ger(
            self.loaderrc["train_dir"],
            split="train",
        )
        self.dev_dataset = dataset_cls_ger(
            self.loaderrc["dev_dir"],
            split="dev",
        )
        self.test_dataset = dataset_cls_ger(
            self.loaderrc["test_dir"],
            split="test",
        )

        if self.modelrc["model"] == "SepRNN":
            self.model = SepRNN(input_dim=self.upstream_dim,
                                num_bins=int(self.datarc['n_fft'] / 2 + 1),
                                rnn=self.modelrc["rnn"],
                                num_spks=self.datarc['num_speakers'],
                                num_layers=self.modelrc["rnn_layers"],
                                hidden_size=self.modelrc["hidden_size"],
                                dropout=self.modelrc["dropout"],
                                non_linear=self.modelrc["non_linear"],
                                bidirectional=self.modelrc["bidirectional"])
        else:
            raise ValueError("Model type not defined.")

        self.loss_type = self.modelrc["loss_type"]
        self.log = self.modelrc["log"]
        self.objective = EnhLoss(self.datarc['num_speakers'], self.loss_type,
                                 self.modelrc["mask_type"], self.log)

        self.register_buffer("best_score", torch.ones(1) * -10000)

    def _get_train_dataloader(self, dataset):
        return DataLoader(
            dataset,
            batch_size=self.loaderrc["train_batch_size"],
            shuffle=True,
            num_workers=self.loaderrc["num_workers"],
            drop_last=False,
            pin_memory=True,
            collate_fn=dataset.collate_fn,
        )

    def _get_eval_dataloader(self, dataset):
        return DataLoader(
            dataset,
            batch_size=self.loaderrc["eval_batch_size"],
            shuffle=False,
            num_workers=self.loaderrc["num_workers"],
            drop_last=False,
            pin_memory=True,
            collate_fn=dataset.collate_fn,
        )

    def get_dataloader(self, mode, batch_size=None):
        """
        Args:
            mode: string
                'train', 'dev' or 'test'
        Return:
            a torch.utils.data.DataLoader returning each batch in the format of:
            [wav1, wav2, ...], your_other_contents1, your_other_contents2, ...
            where wav1, wav2 ... are in variable length
            each wav is torch.FloatTensor in cpu with:
                1. dim() == 1
                2. sample_rate == 16000
                3. directly loaded by torchaudio
        """
        if batch_size:
            self.loaderrc['train_batch_size'] = batch_size
            self.loaderrc['eval_batch_size'] = batch_size

        if mode == "train":
            return self._get_train_dataloader(self.train_dataset)
        elif mode == "dev":
            return self._get_eval_dataloader(self.dev_dataset)
        elif mode == "test":
            return self._get_eval_dataloader(self.test_dataset)

    def forward(self, mode, features, source_attr, uttname_list, source_wav,
                target_attr, target_wav_list, feat_length, wav_length, records,
                **kwargs):
        """
        Args:
            mode: string
                'train', 'dev' or 'test' for this forward step

            features:
                list of unpadded features [feat1, feat2, ...]
                each feat is in torch.FloatTensor and already
                put in the device assigned by command-line args

            source_attr:
                source_attr is a dict containing the STFT information 
                for the mixture. source_attr['magnitude'] stores the STFT
                magnitude, source_attr['phase'] stores the STFT phase and
                source_attr['stft'] stores the raw STFT feature. The shape
                is [bs, max_length, feat_dim]
            
            uttname_list:
                list of utterance names

            source_wav:
                source_wav contains the raw waveform for the mixture,
                and it has the shape of [bs, max_wav_length]

            target_attr:
                similar to source_attr, it contains the STFT information
                for individual sources. It only has two keys ('magnitude' and 'phase')
                target_attr['magnitude'] is a list of length n_srcs, and
                target_attr['magnitude'][i] has the shape [bs, max_length, feat_dim]

            target_wav_list:
                target_wav_list contains the raw waveform for the individual
                sources, and it is a list of length n_srcs. target_wav_list[0]
                has the shape [bs, max_wav_length]

            feat_length:
                length of STFT features

            wav_length:
                length of raw waveform

            records:
                defaultdict(list), by appending contents into records,
                these contents can be averaged and logged on Tensorboard
                later by self.log_records every log_step

        Return:
            loss:
                the loss to be optimized, should not be detached
        """
        if self.upstream_rate == 320:
            features_upsample = [
                torch.repeat_interleave(feat, 2, dim=0) for feat in features
            ]
            features = features_upsample

        # match the feature length to STFT feature length
        features = match_length(features, feat_length)
        features = pack_sequence(features)
        mask_list = self.model(features)
        assert len(mask_list) == 1

        # evaluate the enhancement quality of predict sources
        if mode == 'dev' or mode == 'test':
            if self.log == 'none':
                predict_stfts = torch.squeeze(mask_list[0].cpu() *
                                              source_attr['stft'])
                predict_stfts_np = np.transpose(predict_stfts.data.numpy())
            elif self.log == 'log1p':
                phase = source_attr['stft'] / (source_attr['stft'].abs() + EPS)
                predict_stfts = torch.expm1(mask_list[0].cpu() * torch.log1p(
                    source_attr['stft'].abs())) * phase
                predict_stfts = torch.squeeze(predict_stfts)
                predict_stfts_np = np.transpose(predict_stfts.data.numpy())
            else:
                raise ValueError("log type not defined.")

            assert len(wav_length) == 1
            # reconstruct the signal using iSTFT
            predict_srcs_np = librosa.util.fix_length(librosa.istft(
                predict_stfts_np,
                hop_length=self.datarc['hop_length'],
                win_length=self.datarc['win_length'],
                window=self.datarc['window'],
                center=self.datarc['center']),
                                                      size=wav_length[0])
            predict_srcs_np = np.expand_dims(predict_srcs_np, axis=0)
            gt_srcs_np = torch.cat(target_wav_list, 0).data.cpu().numpy()
            mix_np = source_wav.data.cpu().numpy()

            utt_metrics = get_metrics(
                mix_np,
                gt_srcs_np,
                predict_srcs_np,
                sample_rate=self.datarc['rate'],
                metrics_list=COMPUTE_METRICS,
                compute_permutation=False,
            )

            for metric in COMPUTE_METRICS:
                input_metric = "input_" + metric
                assert metric in utt_metrics and input_metric in utt_metrics
                imp = utt_metrics[metric] - utt_metrics[input_metric]
                if metric not in records:
                    records[metric] = []
                if metric == "si_sdr":
                    records[metric].append(imp)
                elif metric == "stoi" or metric == "pesq":
                    records[metric].append(utt_metrics[metric])
                else:
                    raise ValueError("Metric type not defined.")

            assert 'batch_id' in kwargs
            if kwargs[
                    'batch_id'] % 1000 == 0:  # Save the prediction every 1000 examples
                records['mix'].append(mix_np)
                records['hypo'].append(predict_srcs_np)
                records['ref'].append(gt_srcs_np)
                records['uttname'].append(uttname_list[0])

        if self.loss_type == "MSE" or self.loss_type == "L1":
            loss = self.objective.compute_loss(mask_list, feat_length,
                                               source_attr, target_attr)
        else:
            raise ValueError("Loss type not defined.")

        records["loss"].append(loss.item())
        return loss

    # interface
    def log_records(self, mode, records, logger, global_step, batch_ids,
                    total_batch_num, **kwargs):
        """
        Args:
            mode: string
                'train':
                    records and batchids contain contents for `log_step` batches
                    `log_step` is defined in your downstream config
                    eg. downstream/example/config.yaml
                'dev' or 'test' :
                    records and batchids contain contents for the entire evaluation dataset

            records:
                defaultdict(list), contents already appended

            logger:
                Tensorboard SummaryWriter
                please use f'{prefix}your_content_name' as key name
                to log your customized contents

            global_step:
                The global_step when training, which is helpful for Tensorboard logging

            batch_ids:
                The batches contained in records when enumerating over the dataloader

            total_batch_num:
                The total amount of batches in the dataloader

        Return:
            a list of string
                Each string is a filename we wish to use to save the current model
                according to the evaluation result, like the best.ckpt on the dev set
                You can return nothing or an empty list when no need to save the checkpoint
        """
        if mode == 'train':
            avg_loss = np.mean(records["loss"])
            logger.add_scalar(f"enhancement_stft2/{mode}-loss",
                              avg_loss,
                              global_step=global_step)
            return []
        else:
            eval_result = open(Path(self.expdir) / f"{mode}_metrics.txt", "w")
            avg_loss = np.mean(records["loss"])
            logger.add_scalar(f"enhancement_stft2/{mode}-loss",
                              avg_loss,
                              global_step=global_step)
            for metric in COMPUTE_METRICS:
                avg_metric = np.mean(records[metric])
                if mode == "test" or mode == "dev":
                    print("Average {} of {} utts is {:.4f}".format(
                        metric, len(records[metric]), avg_metric))
                    print(metric, avg_metric, file=eval_result)

                logger.add_scalar(f'enhancement_stft2/{mode}-' + metric,
                                  avg_metric,
                                  global_step=global_step)

            save_ckpt = []
            assert 'pesq' in records
            if mode == "dev" and np.mean(records['pesq']) > self.best_score:
                self.best_score = torch.ones(1) * np.mean(records['pesq'])
                save_ckpt.append(f'{mode}-best.ckpt')

            for s in ['mix', 'ref', 'hypo', 'uttname']:
                assert s in records
            for i in range(len(records['uttname'])):
                utt = records['uttname'][i]
                mix_wav, ref_wav, hypo_wav = records['mix'][i][
                    0, :], records['ref'][i][0, :], records['hypo'][i][0, :]
                mix_wav = librosa.util.normalize(mix_wav,
                                                 norm=np.inf,
                                                 axis=None)
                ref_wav = librosa.util.normalize(ref_wav,
                                                 norm=np.inf,
                                                 axis=None)
                hypo_wav = librosa.util.normalize(hypo_wav,
                                                  norm=np.inf,
                                                  axis=None)
                logger.add_audio('step{:06d}_{}_mix.wav'.format(
                    global_step, utt),
                                 mix_wav,
                                 global_step=global_step,
                                 sample_rate=self.datarc['rate'])
                logger.add_audio('step{:06d}_{}_ref.wav'.format(
                    global_step, utt),
                                 ref_wav,
                                 global_step=global_step,
                                 sample_rate=self.datarc['rate'])
                logger.add_audio('step{:06d}_{}_hypo.wav'.format(
                    global_step, utt),
                                 hypo_wav,
                                 global_step=global_step,
                                 sample_rate=self.datarc['rate'])

            return save_ckpt
