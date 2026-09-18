import torch
from torch.utils.data import Dataset, random_split, DataLoader
import numpy as np
from common.data import rotate

class MetricLearningDataset(Dataset):

    def __init__(self, configurations, order_params, rotate=False, rotate_p = None):
        self.configurations = configurations
        self.order_params = order_params
        self._rotate = rotate
        self._rotate_p = rotate_p
        if rotate:
            self._rng = np.random.default_rng()

    def __len__(self):
        return len(self.configurations)
    
    def __getitem__(self, idx):
        item_config = self.configurations[idx]
        item_order_params = self.order_params[idx]
        if self._rotate:
            angle = self._rng.choice([0, 90, 180, 270], p=self._rotate_p)
            if angle != 0:
                item_config = rotate(item_config, angle)
        return (
            torch.FloatTensor(item_config).permute(2, 0, 1),     # (4, 8, 8)
            torch.FloatTensor(item_order_params)
        )


def make_balanced_train_test_dataset(configurations, order_params, batch_size=32, train_ratio=0.8, rotate=False, rotate_p=None):
    """
    Split configurations and their descriptor vectors into train and test loaders.

    Parameters
    ----------
    rotate : bool
        Apply a random C4 rotation to each configuration on access.
    rotate_p : array_like, optional
        Probabilities of the four rotations.
    """
    dataset = MetricLearningDataset(configurations, order_params, rotate, rotate_p)
    train_size = int(train_ratio * len(dataset))
    test_size = len(dataset) - train_size

    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=True)

    return train_loader, test_loader