import torch
from torch.utils.data import Dataset, random_split, DataLoader
import numpy as np

class BalancedContrastiveDataset(Dataset):
    """
    Dataset that ensures balanced positive and negative pairs for contrastive learning.
    """
    def __init__(self, configurations, labels, positive_ratio=0.5, rotate=True, rotate_p=None):
        self.configurations = configurations
        self.labels = labels
        self.positive_ratio = positive_ratio
        self._rotate = rotate
        self._rotate_p = rotate_p
        if rotate:
            self._rng = np.random.default_rng()
        
        # Pre-compute indices for each class to speed up sampling
        self.class_indices = {}
        unique_labels = np.unique(labels)
        for label in unique_labels:
            if label != -1:  # Exclude noise points from HDBSCAN
                self.class_indices[label] = np.where(labels == label)[0]
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        anchor = self.configurations[idx]
        anchor_label = self.labels[idx]
        
        # Decide if this should be a positive or negative pair
        is_positive = np.random.random() < self.positive_ratio
        
        if is_positive and anchor_label in self.class_indices:
            # Sample positive: same class, exclude anchor itself
            positive_candidates = self.class_indices[anchor_label]
            positive_candidates = positive_candidates[positive_candidates != idx]
            
            if len(positive_candidates) > 0:
                contrast_idx = np.random.choice(positive_candidates)
                label = 1.0
            else:
                # Fallback to negative if no positives available
                contrast_idx = self._sample_negative(idx, anchor_label)
                label = 0.0
        else:
            # Sample negative: different class
            contrast_idx = self._sample_negative(idx, anchor_label)
            label = 0.0
        
        contrast = self.configurations[contrast_idx]
        # Randomly rotate contrast whatever it might be
        # if self._rotate:
        #     angle = self._rng.choice([0, 90, 180, 270], p=self._rotate_p)
        #     if angle != 0:
        #         contrast = rotate(contrast, angle)
        
        return (
            torch.FloatTensor(anchor).permute(2, 0, 1),     # (4, 8, 8)
            torch.FloatTensor(contrast).permute(2, 0, 1),   # (4, 8, 8)
            torch.tensor(label, dtype=torch.float32)
        )
    
    def _sample_negative(self, anchor_idx, anchor_label):
        """Sample a negative example (different class)"""
        # Get all indices with different labels
        negative_mask = (self.labels != anchor_label) & (self.labels != -1)
        negative_indices = np.where(negative_mask)[0]
        
        if len(negative_indices) > 0:
            return np.random.choice(negative_indices)
        else:
            # Fallback: return any index except anchor
            candidates = np.arange(len(self.labels))
            candidates = candidates[candidates != anchor_idx]
            return np.random.choice(candidates)

def make_balanced_train_test_dataset(configurations, labels, batch_size=32, train_ratio=0.8, positive_ratio=0.5):
    """
    Create balanced train/test datasets with controlled positive/negative ratio.
    
    Parameters:
    -----------
    positive_ratio : float
        Fraction of pairs that should be positive (same class)
    """
    dataset = BalancedContrastiveDataset(configurations, labels, positive_ratio=positive_ratio)
    train_size = int(train_ratio * len(dataset))
    test_size = len(dataset) - train_size

    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=True)

    return train_loader, test_loader
