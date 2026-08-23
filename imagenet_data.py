import os
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset
from torch.utils.data.distributed import DistributedSampler
from torchvision import datasets, transforms


class NumericImageFolder(datasets.ImageFolder):
    def find_classes(self, directory):
        classes = [entry.name for entry in os.scandir(directory) if entry.is_dir()]
        if not classes:
            raise FileNotFoundError(f"Couldn't find any class folder in {directory}.")
        if all(name.isdigit() for name in classes):
            classes = sorted(classes, key=lambda name: int(name))
            class_to_idx = {name: int(name) for name in classes}
        else:
            classes = sorted(classes)
            class_to_idx = {name: idx for idx, name in enumerate(classes)}
        return classes, class_to_idx


def _maybe_limit(ds, limit):
    if limit and limit > 0:
        return Subset(ds, range(min(limit, len(ds))))
    return ds


def make_imagenet_loaders(
    data_root,
    img,
    resize,
    batch_size,
    workers,
    limit_train=0,
    limit_val=0,
    distributed=False,
    rank=0,
    world_size=1,
    persistent_workers=True,
):
    data_root = Path(data_root)
    train_dir = data_root / "train"
    val_dir = data_root / "val"
    if not train_dir.is_dir():
        raise FileNotFoundError(f"Missing ImageNet train directory: {train_dir}")
    if not val_dir.is_dir():
        raise FileNotFoundError(f"Missing ImageNet val directory: {val_dir}")

    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(img),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    val_tf = transforms.Compose([
        transforms.Resize(resize),
        transforms.CenterCrop(img),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    train_ds = NumericImageFolder(train_dir, transform=train_tf)
    val_ds = NumericImageFolder(val_dir, transform=val_tf)
    n_classes = max(train_ds.class_to_idx.values()) + 1
    train_ds = _maybe_limit(train_ds, limit_train)
    val_ds = _maybe_limit(val_ds, limit_val)

    pin = torch.cuda.is_available()
    loader_kwargs = dict(
        batch_size=batch_size,
        num_workers=workers,
        pin_memory=pin,
        persistent_workers=persistent_workers and workers > 0,
    )
    train_sampler = DistributedSampler(
        train_ds, num_replicas=world_size, rank=rank, shuffle=True
    ) if distributed else None
    val_sampler = DistributedSampler(
        val_ds, num_replicas=world_size, rank=rank, shuffle=False
    ) if distributed else None

    train_loader = DataLoader(
        train_ds,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        drop_last=False,
        **loader_kwargs,
    )
    val_loader = DataLoader(
        val_ds,
        shuffle=False,
        sampler=val_sampler,
        drop_last=False,
        **loader_kwargs,
    )
    return train_loader, val_loader, n_classes, train_sampler
