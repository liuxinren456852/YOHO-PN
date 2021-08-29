import time
from dataset import ThreeDMatchDataset
import torch


def get_dataloader(root, split, batch_size=1, num_workers=4, shuffle=True, drop_last=True):
    dataset = ThreeDMatchDataset(
        root=root,
        split=split,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last
    )
    dataset.initial()
    dataloader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        drop_last=drop_last
    )

    return dataloader


