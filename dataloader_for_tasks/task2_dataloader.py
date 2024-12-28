# multinode forcasting
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader

class MultiNodeForecastingDataset(Dataset):
    def __init__(self, temporal_data, edge_data, window_size, target_column):
        """
        Args:
            temporal_data: Combined DataFrame of temporal features with 'Date' as index.
            edge_data: DataFrame containing edges to represent graph structure.
            window_size: Number of time steps in the sliding window.
            target_column: The column to be predicted.
        """
        self.temporal_data = temporal_data
        self.edge_data = edge_data
        self.window_size = window_size
        self.target_column = target_column

        # Prepare features and targets
        self.features, self.targets = self._prepare_data()

    def _prepare_data(self):
        features, targets = [], []
        temporal_values = self.temporal_data.values
        target_idx = self.temporal_data.columns.get_loc(self.target_column)

        for i in range(len(temporal_values) - self.window_size):
            feature_window = temporal_values[i:i + self.window_size]
            target = temporal_values[i + self.window_size, target_idx]
            features.append(feature_window)
            targets.append(target)

        return np.array(features), np.array(targets)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]

# Load the temporal data files
temporal_files = [
    '/mnt/data/Sales Order .csv',
    '/mnt/data/Delivery to Distributor.csv',
    '/mnt/data/Factory Issue.csv',
    '/mnt/data/Production .csv'
]
temporal_data = [pd.read_csv(file).set_index('Date') for file in temporal_files]

# Combine temporal data
combined_temporal_data = pd.concat(temporal_data, axis=1)

# Load edge data
edges_plant = pd.read_csv('/mnt/data/Edges (Plant).csv')
edges_storage = pd.read_csv('/mnt/data/Edges (Storage Location).csv')

# Parameters
window_size = 5
target_column = 'SOS008L02P'  # Example target column

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
train_dataset = MultiNodeForecastingDataset(train_data, edges_plant, window_size, target_column)
val_dataset = MultiNodeForecastingDataset(val_data, edges_plant, window_size, target_column)
test_dataset = MultiNodeForecastingDataset(test_data, edges_plant, window_size, target_column)

# Create dataloaders
batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

print("Data loaders created for train, validation, and test sets.")
