# Load the datasets for time-series graph embedding prediction
import pandas as pd

# Load the provided files
# nodes = pd.read_csv('/mnt/data/Nodes.csv')
# edges_plant = pd.read_csv('/mnt/data/Edges (Plant).csv')
# edges_storage = pd.read_csv('/mnt/data/Edges (Storage Location).csv')
# sales_order = pd.read_csv('/mnt/data/Sales Order.csv')
# production = pd.read_csv('/mnt/data/Production .csv')

# # Inspect the structure of each dataset
# nodes_info = nodes.info()
# edges_plant_info = edges_plant.info()
# edges_storage_info = edges_storage.info()
# sales_order_info = sales_order.info()
# production_info = production.info()

# nodes_head = nodes.head()
# edges_plant_head = edges_plant.head()
# edges_storage_head = edges_storage.head()
# sales_order_head = sales_order.head()
# production_head = production.head()

# (nodes_info, edges_plant_info, edges_storage_info, sales_order_info, production_info, 
#  nodes_head, edges_plant_head, edges_storage_head, sales_order_head, production_head)

import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader

class GraphEmbeddingDataset(Dataset):
    def __init__(self, temporal_data, edge_data, window_size):
        """
        Args:
            temporal_data: Combined DataFrame of temporal node features with 'Date' as index.
            edge_data: DataFrame containing edges to represent graph structure.
            window_size: Number of time steps in the sliding window.
        """
        self.temporal_data = temporal_data.set_index('Date')
        self.edge_data = edge_data
        self.window_size = window_size

        # Prepare features and targets
        self.features, self.targets = self._prepare_data()

    def _prepare_data(self):
        features, targets = [], []
        temporal_values = self.temporal_data.values

        for i in range(len(temporal_values) - self.window_size):
            feature_window = temporal_values[i:i + self.window_size]
            target = temporal_values[i + self.window_size]
            features.append(feature_window)
            targets.append(target)

        return np.array(features), np.array(targets)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]

# Load the temporal data files
temporal_files = [
    '/mnt/data/Sales Order.csv',
    '/mnt/data/Production .csv'
]
temporal_data = [pd.read_csv(file) for file in temporal_files]

# Combine temporal data on 'Date'
combined_temporal_data = temporal_data[0]
for df in temporal_data[1:]:
    combined_temporal_data = pd.merge(combined_temporal_data, df, on='Date', how='inner')

# Load edge data
edges_plant = pd.read_csv('/mnt/data/Edges (Plant).csv')
edges_storage = pd.read_csv('/mnt/data/Edges (Storage Location).csv')

# Combine edge data
edges_combined = pd.concat([edges_plant, edges_storage], ignore_index=True)

# Parameters
window_size = 5

# Split into train, validation, and test sets
train_ratio = 0.7
val_ratio = 0.2
n = len(combined_temporal_data)
train_end = int(n * train_ratio)
val_end = int(n * (train_ratio + val_ratio))

train_data = combined_temporal_data.iloc[:train_end]
val_data = combined_temporal_data.iloc[train_end:val_end]
test_data = combined_temporal_data.iloc[val_end:]

# Create datasets
train_dataset = GraphEmbeddingDataset(train_data, edges_combined, window_size)
val_dataset = GraphEmbeddingDataset(val_data, edges_combined, window_size)
test_dataset = GraphEmbeddingDataset(test_data, edges_combined, window_size)

# Create dataloaders
batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

print("Data loaders created for time-series graph embedding prediction.")
