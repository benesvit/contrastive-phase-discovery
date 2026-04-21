import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
import numpy as np


# class Embedder(nn.Module):
#     def __init__(self, embedding_size=4) -> None:
#         super(Embedder, self).__init__()
#         self.conv1 = nn.Conv2d(4, 32, 3, padding=1, padding_mode='circular', bias=False)
#         self.bn1   = nn.BatchNorm2d(32)

#         self.conv2 = nn.Conv2d(32, 64, 3, padding=1, padding_mode='zeros', bias=False)
#         self.bn2   = nn.BatchNorm2d(64)

#         self.pool = nn.AdaptiveAvgPool2d(1)

#         self.fc1 = nn.Linear(64, 128, bias=False)
#         self.bn_fc = nn.BatchNorm1d(128)

#         self.fc2 = nn.Linear(128, embedding_size)
#         self.dropout = nn.Dropout(0.20)
        
#     def forward(self, x):
#         x = F.silu(self.bn1(self.conv1(x)))
#         x = F.silu(self.bn2(self.conv2(x)))
#         x = self.pool(x)                 # (N,64,1,1)
#         x = torch.flatten(x, 1)          # (N,64)
#         x = self.bn_fc(self.fc1(x))      # (N,128)
#         x = F.silu(x)
#         x = self.dropout(x)
#         x = self.fc2(x)                  # (N,embedding_size)
#         x = F.normalize(x, p=2, dim=1)   # unit-norm embeddings
#         return x
    
    
# class Embedder(nn.Module):
#     def __init__(self, embedding_size=4) -> None:
#         super(Embedder, self).__init__()
#         self.conv1 = nn.Conv2d(4, 32, 3, padding=1, padding_mode='circular', bias=False)
#         self.bn1   = nn.BatchNorm2d(32)

#         self.conv2 = nn.Conv2d(32, 64, 3, padding=1, padding_mode='zeros', bias=False)
#         self.bn2   = nn.BatchNorm2d(64)

#         self.pool = nn.AdaptiveAvgPool2d(1)

#         self.fc1 = nn.Linear(64, 128, bias=False)
#         self.bn_fc = nn.BatchNorm1d(128)

#         self.fc2 = nn.Linear(128, 256)
#         self.dropout = nn.Dropout(0.20)

#         self.fc3 = nn.Linear(256, embedding_size)
        
#     def forward(self, x):
#         x = F.silu(self.bn1(self.conv1(x)))
#         x = F.silu(self.bn2(self.conv2(x)))
#         x = self.pool(x)                 # (N,64,1,1)
#         x = torch.flatten(x, 1)          # (N,64)
#         x = self.bn_fc(self.fc1(x))      # (N,128)
#         x = F.silu(x)
#         x = self.dropout(x)
#         x = self.fc2(x)
#         x = F.silu(x)
#         x = self.dropout(x)
#         x = self.fc3(x)                  # (N,embedding_size)
#         x = F.normalize(x, p=2, dim=1)   # unit-norm embeddings
#         return x

# class Embedder(nn.Module):
#     def __init__(self, embedding_size=2):
#         super(Embedder, self).__init__()
#         self.conv1 = nn.Conv2d(4, 32, 3, padding=1, padding_mode='circular', bias=False)
#         self.bn1   = nn.BatchNorm2d(32)

#         self.conv2 = nn.Conv2d(32, 64, 3, padding=1, padding_mode='zeros', bias=False)
#         self.bn2   = nn.BatchNorm2d(64)

#         self.pool = nn.AdaptiveAvgPool2d(1)

#         self.fc1 = nn.Linear(64, 32, bias=False)
#         self.bn_fc1 = nn.BatchNorm1d(32)

#         self.fc2 = nn.Linear(32, 16)
#         self.bn_fc2 = nn.BatchNorm1d(16)

#         self.fc25 = nn.Linear(16, 8, bias=False)
#         self.bn_fc25 = nn.BatchNorm1d(8)
#         self.fc3 = nn.Linear(8, embedding_size)

#         self.dropout = nn.Dropout(0.20)
        
#     def forward(self, x):
#         x = F.silu(self.bn1(self.conv1(x)))
#         x = F.silu(self.bn2(self.conv2(x)))
#         x = self.pool(x)                 # (N,64,1,1)
#         x = torch.flatten(x, 1)          # (N,64)
#         x = self.bn_fc1(self.fc1(x))      # (N,128)
#         x = F.silu(x)
#         x = self.dropout(x)

#         x = self.bn_fc2(self.fc2(x))
#         x = F.silu(x)
#         x = self.dropout(x)

#         x = self.bn_fc25(self.fc25(x))
#         x = F.silu(x)
#         x = self.dropout(x)

#         x = self.fc3(x)
#         x = F.normalize(x, p=2, dim=1)   # unit-norm embeddings
#         return x


class Embedder(nn.Module):
    def __init__(self, embedding_size=2):
        super(Embedder, self).__init__()
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


# class Embedder(nn.Module):
#     def __init__(self, embedding_size=2):
#         super(Embedder, self).__init__()
#         self.conv1 = nn.Conv2d(4, 32, 3, padding=1, padding_mode='circular', bias=False)
#         self.bn1   = nn.BatchNorm2d(32)

#         self.conv2 = nn.Conv2d(32, 64, 3, padding=1, padding_mode='zeros', bias=False)
#         self.bn2   = nn.BatchNorm2d(64)

#         self.conv3 = nn.Conv2d(64, 128, 3, padding=1, padding_mode='zeros', bias=False)
#         self.bn3 = nn.BatchNorm2d(128)

#         self.pool = nn.AdaptiveAvgPool2d(1)


#         self.fc0 = nn.Linear(128, 64, bias=False)
#         self.bn_fc0 = nn.BatchNorm1d(64)


#         self.fc1 = nn.Linear(64, 32, bias=False)
#         self.bn_fc1 = nn.BatchNorm1d(32)

#         self.fc2 = nn.Linear(32, 16)
#         self.bn_fc2 = nn.BatchNorm1d(16)

#         self.fc3 = nn.Linear(16, embedding_size)

#         self.dropout = nn.Dropout(0.20)
        
#     def forward(self, x):
#         x = F.silu(self.bn1(self.conv1(x)))
#         x = F.silu(self.bn2(self.conv2(x)))
#         x = F.silu(self.bn3(self.conv3(x)))

#         x = self.pool(x)                 # (N,64,1,1)
#         x = torch.flatten(x, 1)          # (N,64)

#         x = self.bn_fc0(self.fc0(x))      # (N,128)
#         x = F.silu(x)
#         x = self.dropout(x)

#         x = self.bn_fc1(self.fc1(x))      # (N,128)
#         x = F.silu(x)
#         x = self.dropout(x)

#         x = self.bn_fc2(self.fc2(x))
#         x = F.silu(x)
#         x = self.dropout(x)

#         x = self.fc3(x)
#         x = F.normalize(x, p=2, dim=1)   # unit-norm embeddings
#         return x
    

@dataclass
class TrainConfig:
    model: Embedder
    optimizer: object
    device: torch.device
    loss_function: callable
    Q_distance: callable
    scheduler: object
    train_loader: object
    test_loader: object


def train_step(config: TrainConfig, configurations_batch, order_params_batch):
    config.model.train()
    config.optimizer.zero_grad()

    # Ensure dtypes/shapes
    order_params_batch = order_params_batch.to(config.device)
    configurations_batch = configurations_batch.to(config.device)

    embeddings = config.model(configurations_batch)
    distances = config.Q_distance(order_params_batch)
    loss = config.loss_function(distances,embeddings)
    loss.backward()
    config.optimizer.step()
    return loss.item()

# Training function
def test_step(config: TrainConfig, configurations_batch, order_params_batch):
    config.model.eval()
    with torch.no_grad():
        order_params_batch = order_params_batch.to(config.device)
        configurations_batch = configurations_batch.to(config.device)

        embeddings = config.model(configurations_batch)
        distances = config.Q_distance(order_params_batch)
        loss = config.loss_function(distances,embeddings)

    return loss.item()

# Training loop
def train_model(config: TrainConfig, num_epochs=100):
    avg_loss, avg_val_loss = [], []
    for epoch in range(num_epochs):
        config.model.train()
        total_loss = 0
        for batch in config.train_loader:
            # Get batch from your iterator
            
            configurations_batch, order_params_batch = batch
            
            # Train step
            loss = train_step(config=config,
                            configurations_batch=configurations_batch,
                            order_params_batch=order_params_batch)
            total_loss += loss
        # scheduler step
        config.scheduler.step()

        avg_loss.append(total_loss / len(config.train_loader))
        config.model.eval()
        val_total_loss = 0
        for test_batch in config.test_loader:
            configurations_batch, order_params_batch = test_batch
            val_loss = test_step(config=config,
                                configurations_batch=configurations_batch,
                                order_params_batch=order_params_batch)
            val_total_loss += val_loss
        avg_val_loss.append(val_total_loss / len(config.test_loader))

        print(f"Epoch {epoch+1}/{num_epochs}, Average Loss: {avg_loss[-1]:.6f}, Average Val Loss: {avg_val_loss[-1]:.6f}")
    return avg_loss, avg_val_loss


def get_embeddings(config, configurations):
    config.model.eval()
    with torch.no_grad():
        batch_tensor = torch.FloatTensor(configurations).permute(0, 3, 1, 2).to(config.device)
        embedding = config.model(batch_tensor)
    return embedding.cpu().numpy()