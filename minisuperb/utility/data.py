from pathlib import Path

import torch
from torch.distributed.distributed_c10d import is_initialized
from torch.utils.data import Dataset, DistributedSampler

SAMPLE_RATE = 16000

def get_extracted_dataset(dataset_cls, extract_to_single_file=False, feature_only=True):
        
    class ExtractedDataset(dataset_cls):
        def __init__(self, *args, **kwargs):
            self._split = kwargs.pop("split_name")
            self._extracted_path = Path(kwargs["extracted_path"]) / self._split
            self._use_single_file = extract_to_single_file
            self._feature_only = feature_only

            if self._use_single_file:
                self._all_data = torch.load(self._extracted_path / "all_data.ckpt", "cpu")
            super().__init__(*args, **kwargs)
        def __getitem__(self, index):
            if self._feature_only:
                others = super().__getitem__(index)[1:]
            if self._use_single_file:
                if self._feature_only:
                    feature = self._all_data[index]
                else:
                    feature, *others = self._all_data[index]
                # copy to avoid memory leakage
                if isinstance(feature, (list, tuple)):
                    feature = tuple(f.clone() for f in feature)
                elif isinstance(feature, torch.Tensor):
                    feature = feature.clone()
                return feature, *others
            else:
                def path2name(path):
                    return Path("-".join((Path(path).parts)[-3:])).stem

                x_file = path2name(self.dataset[index])
                path = self._extracted_path / f"{x_file}.ckpt"
                if self._feature_only:
                    return torch.load(path, "cpu")
                else:
                    return torch.load(path, "cpu"), others
    
    return ExtractedDataset

def get_extracted_asr_dataset(dataset_cls, extract_to_single_file=False, feature_only=False):
        
    class ExtractedDataset(dataset_cls):
        def __init__(self, *args, **kwargs):
            self._split = kwargs.pop("split_name")
            self._extracted_path = Path(kwargs["extracted_path"]) / self._split
            self._use_single_file = extract_to_single_file
            self._feature_only = feature_only

            if self._use_single_file:
                self._all_data = torch.load(self._extracted_path / "all_data.ckpt", "cpu")
            super().__init__(*args, **kwargs)
        def __getitem__(self, index):
            if self._feature_only:
                others = super().__getitem__(index)[1:]
            if self._use_single_file:
                if self._feature_only:
                    feature = self._all_data[index]
                else:
                    feature, *others = self._all_data[index]
                # copy to avoid memory leakage
                if isinstance(feature, (list, tuple)):
                    feature = tuple(f.clone() for f in feature)
                elif isinstance(feature, torch.Tensor):
                    feature = feature.clone()
                return feature, *others
            else:
                feature_batch, label_batch, filename_batch = [], [], []
                for x_file in self.X[index]:
                    x_file = Path(x_file).stem
                    path = self._extracted_path / f"{x_file}.ckpt"
                    loader = torch.load(path, "cpu")
                    feature_batch.append(loader[0])
                    label_batch.append(loader[1])
                    filename_batch.append(loader[2])
                return feature_batch, label_batch, filename_batch
    
    return ExtractedDataset

def get_extracted_ss_dataset(dataset_cls, extract_to_single_file=False, feature_only=False):
        
    class ExtractedDataset(dataset_cls):
        def __init__(self, *args, **kwargs):
            self._split = kwargs.pop("split_name")
            self._extracted_path = Path(kwargs["extracted_path"]) / self._split
            self._use_single_file = extract_to_single_file
            self._feature_only = feature_only
            self._ori_collate_fn = super().collate_fn
            if self._use_single_file:
                self._all_data = torch.load(self._extracted_path / "all_data.ckpt", "cpu")
            super().__init__(*args, **kwargs)

        def collate_fn(self, *args):
            result = self._ori_collate_fn(*args)
            uttname_list = result[2]
            feature_list = []
            for reco in uttname_list:
                src_path = self.reco2path[reco][self.src[0]]
                x_file = Path(src_path).stem
                ckpt_path = self._extracted_path / f"{x_file}.ckpt"
                feature_list.append(torch.load(ckpt_path, "cpu")[0])

            return feature_list, *result[1:]

    return ExtractedDataset
    
def get_ddp_sampler(dataset: Dataset, epoch: int):
    """
    This function will create a DistributedSampler if DDP is initialized,
    and will just return None if DDP is not initialized.
    """
    if is_initialized():
        sampler = DistributedSampler(dataset)
        sampler.set_epoch(epoch)
    else:
        sampler = None
    return sampler
