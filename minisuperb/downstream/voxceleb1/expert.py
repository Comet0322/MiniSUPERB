# -*- coding: utf-8 -*- #
"""*********************************************************************************************"""
#   FileName     [ expert.py ]
#   Synopsis     [ the phone linear downstream wrapper ]
#   Author       [ S3PRL ]
#   Copyright    [ Copyleft(c), Speech Lab, NTU, Taiwan ]
"""*********************************************************************************************"""

###############
# IMPORTATION #
###############
import os
from pathlib import Path
#-------------#
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, DistributedSampler
from torch.distributed import is_initialized
from torch.nn.utils.rnn import pad_sequence
#-------------#
from ..model import *
from .dataset import SpeakerClassifiDataset
from minisuperb.utility.data import get_extracted_dataset


class DownstreamExpert(nn.Module):
    """
    Used to handle downstream-specific operations
    eg. downstream forward, metric computation, contents to log
    """

    def __init__(self, upstream_dim, downstream_expert, expdir, **kwargs):
        super(DownstreamExpert, self).__init__()
        self.upstream_dim = upstream_dim
        self.downstream = downstream_expert
        self.datarc = downstream_expert['datarc']
        self.modelrc = downstream_expert['modelrc']
        self.expdir = expdir

        root_dir = Path(self.datarc['file_path'])

        if kwargs["use_extracted_feature"] and kwargs["mode"] != "extract":
            assert (os.path.exists(
                os.path.join(kwargs["extracted_path"], "train"))
                    and os.path.exists(
                        os.path.join(kwargs["extracted_path"], "dev"))
                    and os.path.exists(
                        os.path.join(kwargs["extracted_path"], "test"))
                    ), "You should extract your features first!"
            dataset_cls = get_extracted_dataset(
                SpeakerClassifiDataset, kwargs['extract_to_single_file'])
        else:
            dataset_cls = SpeakerClassifiDataset
        dataset_cls_ger = lambda split: dataset_cls(
            split,
            root_dir,
            self.datarc['meta_data'],
            self.datarc['max_timestep'],
            expdir=self.expdir,
            extracted_path=kwargs["extracted_path"],
            split_name=split)

        self.train_dataset = dataset_cls_ger('train')
        self.dev_dataset = dataset_cls_ger('dev')
        self.test_dataset = dataset_cls_ger('test')

        model_cls = eval(self.modelrc['select'])
        model_conf = self.modelrc.get(self.modelrc['select'], {})
        self.projector = nn.Linear(upstream_dim, self.modelrc['projector_dim'])
        self.model = model_cls(
            input_dim=self.modelrc['projector_dim'],
            output_dim=self.train_dataset.speaker_num,
            **model_conf,
        )
        self.objective = nn.CrossEntropyLoss()
        self.register_buffer('best_score', torch.zeros(1))

    def _get_train_dataloader(self, dataset):
        sampler = DistributedSampler(dataset) if is_initialized() else None
        return DataLoader(dataset,
                          batch_size=self.datarc['train_batch_size'],
                          shuffle=(sampler is None),
                          sampler=sampler,
                          prefetch_factor=self.datarc.get(
                              'prefetch_factor', 2),
                          num_workers=self.datarc['num_workers'],
                          collate_fn=dataset.collate_fn)

    def _get_eval_dataloader(self, dataset):
        return DataLoader(dataset,
                          batch_size=self.datarc['eval_batch_size'],
                          shuffle=False,
                          num_workers=self.datarc['num_workers'],
                          collate_fn=dataset.collate_fn)

    def get_train_dataloader(self):
        return self._get_train_dataloader(self.train_dataset)

    def get_dev_dataloader(self):
        return self._get_eval_dataloader(self.dev_dataset)

    def get_test_dataloader(self):
        return self._get_eval_dataloader(self.test_dataset)

    # Interface
    def get_dataloader(self, mode, batch_size=None):
        if batch_size:
            self.datarc['train_batch_size'] = batch_size
            self.datarc['eval_batch_size'] = batch_size
        return eval(f'self.get_{mode}_dataloader')()

    # Interface
    def forward(self, mode, features, labels, filenames, data_id, records,
                **kwargs):
        device = features[0].device
        features_len = torch.tensor([len(feat) for feat in features],
                                    dtype=torch.int,
                                    device=device)
        features = pad_sequence(features, batch_first=True)
        features = self.projector(features)
        predicted, _ = self.model(features, features_len)

        labels = torch.tensor(labels, dtype=torch.long, device=features.device)
        loss = self.objective(predicted, labels)

        predicted_classid = predicted.max(dim=-1).indices
        records['acc'] += (
            predicted_classid == labels).view(-1).cpu().float().tolist()
        records['loss'].append(loss.item())

        records['filename'] += filenames
        records['predict_speaker'] += SpeakerClassifiDataset.label2speaker(
            predicted_classid.cpu().tolist())
        records['truth_speaker'] += SpeakerClassifiDataset.label2speaker(
            labels.cpu().tolist())

        return loss

    # interface
    def log_records(self, mode, records, logger, global_step, **kwargs):
        save_names = []
        for key in ["acc", "loss"]:
            average = torch.FloatTensor(records[key]).mean().item()
            logger.add_scalar(f'voxceleb1/{mode}-{key}',
                              average,
                              global_step=global_step)
            with open(Path(self.expdir) / "log.log", 'a') as f:
                if key == 'acc':
                    print(f"{mode} {key}: {average}")
                    f.write(f'{mode} at step {global_step}: {average}\n')
                    if mode == 'dev' and average > self.best_score:
                        self.best_score = torch.ones(1) * average
                        f.write(
                            f'New best on {mode} at step {global_step}: {average}\n'
                        )
                        save_names.append(f'{mode}-best.ckpt')

        if mode in ["dev", "test"]:
            with open(Path(self.expdir) / f"{mode}_predict.txt", "w") as file:
                lines = [
                    f"{f} {p}\n" for f, p in zip(records["filename"],
                                                 records["predict_speaker"])
                ]
                file.writelines(lines)

            with open(Path(self.expdir) / f"{mode}_truth.txt", "w") as file:
                lines = [
                    f"{f} {l}\n" for f, l in zip(records["filename"],
                                                 records["truth_speaker"])
                ]
                file.writelines(lines)

        return save_names
