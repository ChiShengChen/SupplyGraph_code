# Demand forecast based on single node
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader

class TemporalDataset(Dataset):
    def __init__(self, data, target_column, window_size):
        self.data = data
        self.target_column = target_column
        self.window_size = window_size

        # Prepare features and targets
        self.features, self.targets = self._prepare_data()

    def _prepare_data(self):
        features, targets = [], []
        for i in range(len(self.data) - self.window_size):
            feature_window = self.data.iloc[i:i + self.window_size].values
            target = self.data.iloc[i + self.window_size][self.target_column]
            features.append(feature_window)
            targets.append(target)
        return np.array(features), np.array(targets)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]

# Load the datasets
nodes = pd.read_csv('/mnt/data/Nodes.csv')
sales_order = pd.read_csv('/mnt/data/Sales Order .csv')
delivery_to_distributor = pd.read_csv('/mnt/data/Delivery to Distributor.csv')
factory_issue = pd.read_csv('/mnt/data/Factory Issue.csv')
production = pd.read_csv('/mnt/data/Production .csv')

# Merge all temporal data based on the Date column
sales_order.set_index('Date', inplace=True)
delivery_to_distributor.set_index('Date', inplace=True)
factory_issue.set_index('Date', inplace=True)
production.set_index('Date', inplace=True)

# Combine into a single DataFrame
data = sales_order.join([delivery_to_distributor, factory_issue, production], how='inner', lsuffix='_sales', rsuffix='_other')
data.reset_index(inplace=True)

# Select a target column (e.g., 'SOS008L02P_sales') and window size
target_column = 'SOS008L02P_sales'
window_size = 5

# Split the dataset into train, validation, and test sets
train_ratio = 0.7
val_ratio = 0.2

train_size = int(len(data) * train_ratio)
val_size = int(len(data) * val_ratio)
test_size = len(data) - train_size - val_size

train_data = data.iloc[:train_size]
val_data = data.iloc[train_size:train_size + val_size]
test_data = data.iloc[train_size + val_size:]

# Create PyTorch datasets
train_dataset = TemporalDataset(train_data, target_column, window_size)
val_dataset = TemporalDataset(val_data, target_column, window_size)
test_dataset = TemporalDataset(test_data, target_column, window_size)

# Create DataLoaders
batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

print("Data preparation complete.")
