import functools
import os

import lightning as L
import torch
import torchvision
import torchvision.transforms as transforms


class CifarClassifierLightningDataModule(L.LightningDataModule):
    def __init__(
        self,
        batch_size: int,
        num_workers: int,
        pin_memory: bool,
    ) -> None:
        """Assign parameters.

        Args:
            batch_size (int): Batch size
            num_workers (int): Number of process to use when loading images during training.
            pin_memory (bool): Should data been pin to memory flag.
        """
        super().__init__()

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

    def setup(self, stage: str) -> None:
        """Set up module before training.

        Args:
            stage (str): Training stage.
        """
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        self.train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
        self.validation_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
        self.test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

    def train_dataloader(self) -> torch.utils.data.DataLoader:
        """Build training dataloader.

        Returns:
            torch.utils.data.DataLoader: DataLoader object.
        """
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            drop_last=True,
            pin_memory=self.pin_memory,
        )

    def val_dataloader(self) -> torch.utils.data.DataLoader:
        """Build validation dataloader.

        Returns:
            torch.utils.data.DataLoader: DataLoader object.
        """
        return torch.utils.data.DataLoader(
            self.validation_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=self.pin_memory,
        )

    def test_dataloader(self) -> torch.utils.data.DataLoader:
        """Build test dataloader.

        Returns:
            torch.utils.data.DataLoader: DataLoader object.
        """
        return torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            drop_last=False,
            pin_memory=self.pin_memory,
        )
