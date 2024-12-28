# Load datasets for subnetwork local prediction
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader


# import pandas as pd

# # Load the provided files
# nodes = pd.read_csv('/mnt/data/Nodes.csv')
# nodes_index = pd.read_csv('/mnt/data/NodesIndex.csv')
# production = pd.read_csv('/mnt/data/Production .csv')
# sales_order = pd.read_csv('/mnt/data/Sales Order.csv')
# edges_plant = pd.read_csv('/mnt/data/Edges (Plant).csv')
# edges_storage = pd.read_csv('/mnt/data/Edges (Storage Location).csv')

# # Inspect the structure and initial data of each file
# nodes_info = nodes.info()
# nodes_index_info = nodes_index.info()
# production_info = production.info()
# sales_order_info = sales_order.info()
# edges_plant_info = edges_plant.info()
# edges_storage_info = edges_storage.info()

# nodes_head = nodes.head()
# nodes_index_head = nodes_index.head()
# production_head = production.head()
# sales_order_head = sales_order.head()
# edges_plant_head = edges_plant.head()
# edges_storage_head = edges_storage.head()

# (nodes_info, nodes_index_info, production_info, sales_order_info, edges_plant_info, edges_storage_info,
#  nodes_head, nodes_index_head, production_head, sales_order_head, edges_plant_head, edges_storage_head)


class SubnetworkDataset(Dataset):
    def __init__(self, temporal_data, edge_data, node_subset, window_size, target_column):
        """
        Args:
            temporal_data: Combined DataFrame of temporal node features with 'Date' as index.
            edge_data: DataFrame containing edges to represent graph structure.
            node_subset: List of nodes defining the subnetwork.
            window_size: Number of time steps in the sliding window.
            target_column: The column to be predicted.
        """
        self.temporal_data = temporal_data.set_index('Date')
        self.edge_data = edge_data
        self.node_subset = node_subset
        self.window_size = window_size
        self.target_column = target_column

        # Filter temporal data for the subnetwork
        self.temporal_data = self.temporal_data[self.node_subset]

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

# Load temporal data
production = pd.read_csv('/mnt/data/Production .csv')
sales_order = pd.read_csv('/mnt/data/Sales Order.csv')

# Combine temporal data on 'Date'
temporal_data = pd.merge(production, sales_order, on='Date', how='inner')

# Load edge data
edges_plant = pd.read_csv('/mnt/data/Edges (Plant).csv')
edges_storage = pd.read_csv('/mnt/data/Edges (Storage Location).csv')

# Define subnetwork (example: all nodes connected to a specific plant)
plant_id = 1903
subnetwork_edges = edges_plant[edges_plant['Plant'] == plant_id]
node_subset = list(set(subnetwork_edges['node1']).union(set(subnetwork_edges['node2'])))

# Parameters
window_size = 5
target_column = 'SOS008L02P'  # Example target column

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
train_dataset = SubnetworkDataset(train_data, subnetwork_edges, node_subset, window_size, target_column)
val_dataset = SubnetworkDataset(val_data, subnetwork_edges, node_subset, window_size, target_column)
test_dataset = SubnetworkDataset(test_data, subnetwork_edges, node_subset, window_size, target_column)

# Create dataloaders
batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

print("Subnetwork dataloaders created for train, validation, and test sets.")
