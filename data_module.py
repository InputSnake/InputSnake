from typing import Any
from typing import Optional
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
import h5py


class EHRDataset(Dataset):
    def __init__(self, data, merge, task, teacher_forcing):
        self.merge = merge
        self.task = task
        self.teacher_forcing = teacher_forcing
        self.time_series = data["time_series"][()]
        self.time_invariant = data["time_invariant"][()]
        if self.task == "iteration":
            self.diag_label = data["diagnoses_LABEL"][()]
            self.los_label = data["los_LABEL"][()]
            self.mort_label = data["mortality_LABEL"][()]
        else:
            self.label = data[f"{task}_LABEL"][()]

    def __getitem__(self, index):
        ts = self.time_series[index]
        ti = self.time_invariant[index]
        hours, dimension = ts.shape
        if self.merge:
            tc = np.hstack((ts, np.tile(ti, (hours, 1))))
            x = torch.from_numpy(tc).float()
        else:
            ts = torch.from_numpy(ts).float()
            ti = torch.from_numpy(ti).float()
            x = (ti, ts)
        if self.task == "iteration":
            diag_label = torch.tensor(self.diag_label[index]).float()
            los_label = torch.tensor(self.los_label[index]).float()
            mort_label = torch.tensor(self.mort_label[index]).float()
            y = (diag_label, los_label, mort_label)
        else:
            y = torch.tensor(self.label[index]).float()

        if self.teacher_forcing:
            return (x, y), y
        else:
            return x, y

    def __len__(self):
        return len(self.time_series)


class MimicDataModule(pl.LightningDataModule):
    def __init__(self,
                 data_dir: str = "/path/to/the/processed/data",
                 task: str = "diagnoses",  
                 teacher_forcing: bool = True,  
                 batch_size: int = 64,
                 merge: bool = True,
                 num_workers: int = 12,
                 **kwargs: Any) -> None:
        super().__init__()
        self.task = task
        self.teacher_forcing = teacher_forcing
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.merge = merge
        self.num_workers = num_workers

    def prepare_data(self):
        pass

    def setup(self, stage: Optional[str] = None):
        hf = h5py.File(self.data_dir, "r")
        if stage == "fit" or stage is None:
            self.train = EHRDataset(hf["train"], self.merge, self.task, self.teacher_forcing)
            self.val = EHRDataset(hf["val"], self.merge, self.task, self.teacher_forcing)
        if stage == "test" or stage is None:
            self.test = EHRDataset(hf["test"], self.merge, self.task, self.teacher_forcing)
        hf.close()

    def train_dataloader(self):
        
        return DataLoader(self.train, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val, batch_size=self.batch_size*5, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test, batch_size=self.batch_size*5, num_workers=self.num_workers)

