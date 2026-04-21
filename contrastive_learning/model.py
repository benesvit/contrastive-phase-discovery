import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
import numpy as np

    
class SiameseEncoder(nn.Module):
    def __init__(self, embedding_size=2):
        super(SiameseEncoder, self).__init__()
        self.conv1 = nn.Conv2d(4, 32, 3, padding=1, padding_mode='circular', bias=False)
        self.bn1   = nn.BatchNorm2d(32)

        self.conv2 = nn.Conv2d(32, 64, 3, padding=1, padding_mode='zeros', bias=False)
        self.bn2   = nn.BatchNorm2d(64)

        self.pool = nn.AdaptiveAvgPool2d(1)

        self.fc1 = nn.Linear(64, 32, bias=False)
        self.bn_fc1 = nn.BatchNorm1d(32)

        self.fc2 = nn.Linear(32, 16)
        self.bn_fc2 = nn.BatchNorm1d(16)

        self.fc3 = nn.Linear(16, embedding_size)

        self.dropout = nn.Dropout(0.20)
        
    def forward(self, x):
        x = F.silu(self.bn1(self.conv1(x)))
        x = F.silu(self.bn2(self.conv2(x)))
        x = self.pool(x)                 # (N,64,1,1)
        x = torch.flatten(x, 1)          # (N,64)
        x = self.bn_fc1(self.fc1(x))      # (N,128)
        x = F.silu(x)
        x = self.dropout(x)

        x = self.bn_fc2(self.fc2(x))
        x = F.silu(x)
        x = self.dropout(x)

        x = self.fc3(x)
        x = F.normalize(x, p=2, dim=1)   # unit-norm embeddings
        return x

class SiameseNetwork(nn.Module):
    def __init__(self, embedding_size=2):
        super(SiameseNetwork, self).__init__()
        self.encoder = SiameseEncoder(embedding_size)
    
    def forward(self, anchor, contrast):
        # Same encoder processes both inputs
        embedding_a = self.encoder(anchor)
        embedding_b = self.encoder(contrast)
        return embedding_a, embedding_b
    

@dataclass
class TrainConfig:
    model: SiameseNetwork
    optimizer: object
    device: torch.device
    scheduler: object
    train_loader: object
    test_loader: object

def contrastive_loss(embedding_a, embedding_b, label, margin=1.0):
    """
    Contrastive loss function.
    label: 1 if same class, 0 if different class
    """
    
    label = label.float().view(-1)  # important to ensure label is float tensor
    distance = F.pairwise_distance(embedding_a, embedding_b, p=2)
    # Contrastive loss
    positive_loss = label * distance.pow(2)
    negative_loss = (1 - label) * torch.clamp(margin - distance, min=0.0).pow(2)
    
    return 0.5 * (positive_loss + negative_loss).mean()



def train_step(config: TrainConfig, anchor_batch, contrast_batch, labels, margin=1.0):
    config.model.train()
    config.optimizer.zero_grad()

    # Ensure dtypes/shapes
    labels = labels.float().view(-1).to(config.device)
    anchor_batch = anchor_batch.to(config.device)
    contrast_batch = contrast_batch.to(config.device)

    # (voliteľné) skip batch size 1 kvôli BN
    if anchor_batch.size(0) < 2:
        return 0.0

    embedding_a, embedding_b = config.model(anchor_batch, contrast_batch)
    loss = contrastive_loss(embedding_a, embedding_b, labels, margin=margin)
    loss.backward()
    config.optimizer.step()
    return loss.item()

# Training function
def test_step(config : TrainConfig, anchor_batch, contrast_batch, labels,margin=1.0):
    config.model.eval()
    with torch.no_grad():
        anchor_batch = anchor_batch.to(config.device)
        contrast_batch = contrast_batch.to(config.device)
        labels = labels.to(config.device)

        embedding_a, embedding_b = config.model(anchor_batch, contrast_batch)
        loss = contrastive_loss(embedding_a, embedding_b, labels,margin=margin)
    
    return loss.item()

# Training loop
def train_model(config: TrainConfig, num_epochs=100, steps_per_epoch=100, margin=1.0):
    avg_loss, avg_val_loss = [], []
    for epoch in range(num_epochs):
        config.model.train()
        total_loss = 0
        for batch in config.train_loader:
            # Get batch from your iterator
            anchor_configs, contrast_configs, y_labels = batch
            
            # Prepare data
            # anchor_configs = Configurations[list(anchor_indices)]
            # contrast_configs = Configurations[list(contrast_indices)]
            # y_labels = np.array(labels, dtype=np.float32)
            
            # Train step
            loss = train_step(config, anchor_configs, contrast_configs, y_labels, margin=margin)
            total_loss += loss
        # scheduler step
        config.scheduler.step()

        avg_loss.append(total_loss / len(config.train_loader))
        config.model.eval()
        val_total_loss = 0
        for test_batch in config.test_loader:
            anchor_configs, contrast_configs, y_labels = test_batch
            val_loss = test_step(config, anchor_configs, contrast_configs, y_labels, margin=margin)
            val_total_loss += val_loss
        avg_val_loss.append(val_total_loss / len(config.test_loader))

        print(f"Epoch {epoch+1}/{num_epochs}, Average Loss: {avg_loss[-1]:.6f}, Average Val Loss: {avg_val_loss[-1]:.6f}")
    return avg_loss, avg_val_loss


def get_embeddings(config: TrainConfig, configurations):
    config.model.eval()
    embeddings = []
    
    with torch.no_grad():
        for i in range(0, len(configurations), 32):  # Process in batches
            batch = configurations[i:i+32]
            batch_tensor = torch.FloatTensor(batch).permute(0, 3, 1, 2).to(config.device)
            embedding = config.model.encoder(batch_tensor)
            embeddings.append(embedding.cpu().numpy())
    
    return np.vstack(embeddings)
