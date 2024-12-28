# Load datasets for long-term and short-term demand forecasting
# import pandas as pd

# # Load the provided files
# production = pd.read_csv('/mnt/data/Production .csv')
# sales_order = pd.read_csv('/mnt/data/Sales Order.csv')
# nodes = pd.read_csv('/mnt/data/Nodes.csv')

# # Inspect the structure and initial data of each file
# production_info = production.info()
# sales_order_info = sales_order.info()
# nodes_info = nodes.info()

# production_head = production.head()
# sales_order_head = sales_order.head()
# nodes_head = nodes.head()

# (production_info, sales_order_info, nodes_info, production_head, sales_order_head, nodes_head)

import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader

class LongShortTermDemandDataset(Dataset):
    def __init__(self, temporal_data, window_size_short, window_size_long, target_column, prediction_horizon_short, prediction_horizon_long):
        """
        Args:
            temporal_data: DataFrame of temporal features with 'Date' as index.
            window_size_short: Number of time steps for short-term input.
            window_size_long: Number of time steps for long-term input.
            target_column: The column to be predicted.
            prediction_horizon_short: Number of steps to predict for short-term.
            prediction_horizon_long: Number of steps to predict for long-term.
        """
        self.temporal_data = temporal_data.set_index('Date')
        self.window_size_short = window_size_short
        self.window_size_long = window_size_long
        self.target_column = target_column
        self.prediction_horizon_short = prediction_horizon_short
        self.prediction_horizon_long = prediction_horizon_long

        # Prepare features and targets
        self.features_short, self.features_long, self.targets_short, self.targets_long = self._prepare_data()

    def _prepare_data(self):
        features_short, features_long, targets_short, targets_long = [], [], [], []
        temporal_values = self.temporal_data[self.target_column].values

        for i in range(len(temporal_values) - max(self.window_size_long, self.prediction_horizon_long)):
            if i + self.window_size_short + self.prediction_horizon_short <= len(temporal_values):
                short_window = temporal_values[i:i + self.window_size_short]
                short_target = temporal_values[i + self.window_size_short:i + self.window_size_short + self.prediction_horizon_short]
                features_short.append(short_window)
                targets_short.append(short_target)

            if i + self.window_size_long + self.prediction_horizon_long <= len(temporal_values):
                long_window = temporal_values[i:i + self.window_size_long]
                long_target = temporal_values[i + self.window_size_long:i + self.window_size_long + self.prediction_horizon_long]
                features_long.append(long_window)
                targets_long.append(long_target)

        return (np.array(features_short), np.array(features_long),
                np.array(targets_short), np.array(targets_long))

    def __len__(self):
        return len(self.features_short)

    def __getitem__(self, idx):
        return (self.features_short[idx], self.features_long[idx],
                self.targets_short[idx], self.targets_long[idx])

# Load temporal data
production = pd.read_csv('/mnt/data/Production .csv')
sales_order = pd.read_csv('/mnt/data/Sales Order.csv')

# Combine temporal data on 'Date'
temporal_data = pd.merge(production, sales_order, on='Date', how='inner')

# Parameters
window_size_short = 7  # Last 7 days for short-term input
window_size_long = 180  # Last 6 months (approx.) for long-term input
target_column = 'SOS008L02P'  # Example target column
prediction_horizon_short = 7  # Predict next 7 days
target_horizon_long = 180  # Predict next 6 months (approx.)

# Split into train, validation, and test sets
train_ratio = 0.7
val_ratio = 0.2
n = len(temporal_data)
train_end = int(n * train_ratio)
val_end = int(n * (train_ratio + val_ratio))

train_data = temporal_data.iloc[:train_end]
val_data = temporal_data.iloc[train_end:val_end]
test_data = temporal_data.iloc[val_end:]

# Create datasets
train_dataset = LongShortTermDemandDataset(train_data, window_size_short, window_size_long, target_column, prediction_horizon_short, target_horizon_long)
val_dataset = LongShortTermDemandDataset(val_data, window_size_short, window_size_long, target_column, prediction_horizon_short, target_horizon_long)
test_dataset = LongShortTermDemandDataset(test_data, window_size_short, window_size_long, target_column, prediction_horizon_short, target_horizon_long)

# Create dataloaders
batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

print("Long-term and short-term demand forecasting dataloaders created.")
