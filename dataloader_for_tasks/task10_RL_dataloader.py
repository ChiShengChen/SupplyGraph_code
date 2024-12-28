# Load datasets for mixed-mode prediction dataloader
import pandas as pd

# Load the provided files
# delivery_to_distributor = pd.read_csv('/mnt/data/Delivery To distributor.csv')
# factory_issue = pd.read_csv('/mnt/data/Factory Issue.csv')
# production = pd.read_csv('/mnt/data/Production .csv')
# sales_order = pd.read_csv('/mnt/data/Sales Order.csv')
# edges_plant = pd.read_csv('/mnt/data/Edges (Plant).csv')
# edges_product_group = pd.read_csv('/mnt/data/Edges (Product Group).csv')
# edges_product_subgroup = pd.read_csv('/mnt/data/Edges (Product Sub-Group).csv')
# edges_storage_location = pd.read_csv('/mnt/data/Edges (Storage Location).csv')
# nodes = pd.read_csv('/mnt/data/Nodes.csv')

# # Inspect the structure and initial data of each file
# delivery_info = delivery_to_distributor.info()
# factory_info = factory_issue.info()
# production_info = production.info()
# sales_order_info = sales_order.info()
# edges_plant_info = edges_plant.info()
# edges_product_group_info = edges_product_group.info()
# edges_product_subgroup_info = edges_product_subgroup.info()
# edges_storage_location_info = edges_storage_location.info()
# nodes_info = nodes.info()

# delivery_head = delivery_to_distributor.head()
# factory_head = factory_issue.head()
# production_head = production.head()
# sales_order_head = sales_order.head()
# edges_plant_head = edges_plant.head()
# edges_product_group_head = edges_product_group.head()
# edges_product_subgroup_head = edges_product_subgroup.head()
# edges_storage_location_head = edges_storage_location.head()
# nodes_head = nodes.head()

# (
#     delivery_info, factory_info, production_info, sales_order_info,
#     edges_plant_info, edges_product_group_info, edges_product_subgroup_info, edges_storage_location_info,
#     nodes_info, delivery_head, factory_head, production_head, sales_order_head, edges_plant_head,
#     edges_product_group_head, edges_product_subgroup_head, edges_storage_location_head, nodes_head
# )


import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader

class MixedModeRLDataset(Dataset):
    def __init__(self, node_data, edge_data, temporal_data, window_size, target_columns):
        """
        Args:
            node_data: DataFrame containing static node attributes.
            edge_data: DataFrame containing static edge attributes.
            temporal_data: DataFrame of temporal node/edge features with 'Date' as index.
            window_size: Number of time steps for input features.
            target_columns: List of columns for multi-task prediction.
        """
        self.node_data = node_data.set_index('Node')
        self.edge_data = edge_data
        self.temporal_data = temporal_data.set_index('Date')
        self.window_size = window_size
        self.target_columns = target_columns

        # Prepare features and targets
        self.features, self.targets = self._prepare_data()

    def _prepare_data(self):
        features, targets = [], []
        temporal_values = self.temporal_data.values

        for i in range(len(temporal_values) - self.window_size):
            feature_window = temporal_values[i:i + self.window_size]
            target = temporal_values[i + self.window_size, :]

            features.append(feature_window)
            targets.append(target)

        return np.array(features), np.array(targets)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]

# Load temporal data
delivery_to_distributor = pd.read_csv('/mnt/data/Delivery To distributor.csv')
factory_issue = pd.read_csv('/mnt/data/Factory Issue.csv')
production = pd.read_csv('/mnt/data/Production .csv')
sales_order = pd.read_csv('/mnt/data/Sales Order.csv')

# Merge temporal data on 'Date'
temporal_data = sales_order.copy()
temporal_data = temporal_data.merge(production, on='Date', how='inner', suffixes=('', '_prod'))
temporal_data = temporal_data.merge(delivery_to_distributor, on='Date', how='inner', suffixes=('', '_delivery'))
temporal_data = temporal_data.merge(factory_issue, on='Date', how='inner', suffixes=('', '_factory'))

# Load node and edge data
nodes = pd.read_csv('/mnt/data/Nodes.csv')
edges_plant = pd.read_csv('/mnt/data/Edges (Plant).csv')
edges_storage_location = pd.read_csv('/mnt/data/Edges (Storage Location).csv')

# Combine edge data
edge_data = pd.concat([edges_plant, edges_storage_location], ignore_index=True)

# Parameters
window_size = 7  # Use the past 7 days for predictions
target_columns = temporal_data.columns[1:]  # All columns except 'Date'

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
train_dataset = MixedModeRLDataset(nodes, edge_data, train_data, window_size, target_columns)
val_dataset = MixedModeRLDataset(nodes, edge_data, val_data, window_size, target_columns)
test_dataset = MixedModeRLDataset(nodes, edge_data, test_data, window_size, target_columns)

# Create dataloaders
batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

print("Mixed-mode RL dataloaders created for train, validation, and test sets.")
