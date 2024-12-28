# Load the datasets for multi-edge heterogeneity prediction
import pandas as pd

# # Load the provided files
# edges_plant = pd.read_csv('/mnt/data/Edges (Plant).csv')
# edges_storage = pd.read_csv('/mnt/data/Edges (Storage Location).csv')
# delivery_to_distributor = pd.read_csv('/mnt/data/Delivery To distributor.csv')
# factory_issue = pd.read_csv('/mnt/data/Factory Issue.csv')
# production = pd.read_csv('/mnt/data/Production .csv')
# sales_order = pd.read_csv('/mnt/data/Sales Order.csv')
# nodes = pd.read_csv('/mnt/data/Nodes.csv')

# # Inspect the structure of each dataset
# edges_plant_info = edges_plant.info()
# edges_storage_info = edges_storage.info()
# delivery_info = delivery_to_distributor.info()
# factory_info = factory_issue.info()
# production_info = production.info()
# sales_order_info = sales_order.info()
# nodes_info = nodes.info()

# edges_plant_head = edges_plant.head()
# edges_storage_head = edges_storage.head()
# delivery_head = delivery_to_distributor.head()
# factory_head = factory_issue.head()
# production_head = production.head()
# sales_order_head = sales_order.head()
# nodes_head = nodes.head()

# (edges_plant_info, edges_storage_info, delivery_info, factory_info, production_info, sales_order_info, nodes_info,
#  edges_plant_head, edges_storage_head, delivery_head, factory_head, production_head, sales_order_head, nodes_head)
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader

class MultiEdgeHeterogeneityDataset(Dataset):
    def __init__(self, temporal_data, edge_data, heterogeneity_columns, window_size, target_column):
        """
        Args:
            temporal_data: Combined DataFrame of temporal node features with 'Date' as index.
            edge_data: DataFrame containing edges to represent graph structure.
            heterogeneity_columns: Columns in edge_data representing heterogeneity.
            window_size: Number of time steps in the sliding window.
            target_column: The column to be predicted.
        """
        self.temporal_data = temporal_data.set_index('Date')
        self.edge_data = edge_data
        self.heterogeneity_columns = heterogeneity_columns
        self.window_size = window_size
        self.target_column = target_column

        # Prepare features and targets
        self.features, self.targets = self._prepare_data()

    def _prepare_data(self):
        features, targets = [], []
        for _, edge in self.edge_data.iterrows():
            node1, node2 = edge['node1'], edge['node2']
            heterogeneity = edge[self.heterogeneity_columns].values

            if node1 in self.temporal_data.columns and node2 in self.temporal_data.columns:
                edge_data = self.temporal_data[[node1, node2, self.target_column]].values
                for i in range(len(edge_data) - self.window_size):
                    feature_window = edge_data[i:i + self.window_size, :-1]
                    feature_with_heterogeneity = np.hstack([feature_window.flatten(), heterogeneity])
                    target = edge_data[i + self.window_size, -1]
                    features.append(feature_with_heterogeneity)
                    targets.append(target)
        return np.array(features), np.array(targets)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]

# Load the temporal data files
temporal_files = [
    '/mnt/data/Delivery To distributor.csv',
    '/mnt/data/Factory Issue.csv',
    '/mnt/data/Production .csv',
    '/mnt/data/Sales Order.csv'
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
heterogeneity_columns = ['Plant', 'Storage Location']  # Example heterogeneity columns
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
train_dataset = MultiEdgeHeterogeneityDataset(train_data, edges_combined, heterogeneity_columns, window_size, target_column)
val_dataset = MultiEdgeHeterogeneityDataset(val_data, edges_combined, heterogeneity_columns, window_size, target_column)
test_dataset = MultiEdgeHeterogeneityDataset(test_data, edges_combined, heterogeneity_columns, window_size, target_column)

# Create dataloaders
batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

print("Data loaders created for multi-edge heterogeneity prediction.")
