import pytorch_lightning as pl
import torchvision

from torch.utils.data import random_split, DataLoader
from torchvision import transforms


class TorchCifar10DataModule(pl.LightningDataModule):
    """
    Class for specifying my pytorch-lightning data module.
    The way data are prepared and data loaders are defined here.
    """

    def __init__(self, root_dir, batch_size=64) -> None:
        """Data module initialization.
        Args:
            root_dir (str): Path to the file containing data.
            batch_size (int, optional): Number of samples in a batch. Defaults to 64.
        """
        super().__init__()
        self.root_dir = root_dir
        self.batch_size = batch_size

    def setup(self, stage):
        # make assignments here (val/train/test split)
        # called on every process in DDP

        # transforms for images
        transform = transforms.Compose(
            [
                transforms.ToTensor(),  # converts a PIL image into a torch tensor
                transforms.Normalize(
                    (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
                ),  # normalize inputs into [-1;1]
            ]
        )

        # Assign train/val datasets for use in dataloaders
        if stage == "fit" or stage is None:
            training_set = torchvision.datasets.CIFAR10(
                root=self.root_dir, train=True, download=True, transform=transform
            )
            self.training_set, self.validation_set = random_split(training_set, [45000, 5000])

        # Assign test dataset for use in dataloader(s)
        if stage == "test" or stage is None:
            self.test_set = torchvision.datasets.CIFAR10(
                root=self.root_dir, train=False, download=True, transform=transform
            )

    def train_dataloader(self):
        """
        Return the data loader associated to the training set.
        """
        return DataLoader(self.training_set, batch_size=self.batch_size, num_workers=4)

    def val_dataloader(self):
        """
        Return the data loader associated to the validation set.
        """
        return DataLoader(self.validation_set, batch_size=self.batch_size, num_workers=4)

    def test_dataloader(self):
        """Return the data loader associated to the test set.
        """
        return DataLoader(self.test_set, batch_size=self.batch_size, num_workers=4)
