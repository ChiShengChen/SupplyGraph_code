#  "Multi-Modal Data Fusion Prediction"
# Load datasets for multi-modal data fusion prediction
import pandas as pd

# # Load the provided files
# edges_plant = pd.read_csv('/mnt/data/Edges (Plant).csv')
# edges_product_group = pd.read_csv('/mnt/data/Edges (Product Group).csv')
# edges_product_subgroup = pd.read_csv('/mnt/data/Edges (Product Sub-Group).csv')
# edges_storage_location = pd.read_csv('/mnt/data/Edges (Storage Location).csv')
# nodes = pd.read_csv('/mnt/data/Nodes.csv')
# nodes_index = pd.read_csv('/mnt/data/NodesIndex.csv')
# nodes_type_plant_storage = pd.read_csv('/mnt/data/Nodes Type (Plant & Storage).csv')
# nodes_type_product_group_subgroup = pd.read_csv('/mnt/data/Node Types (Product Group and Subgroup).csv')

# # Inspect the structure and initial data of each file
# edges_plant_info = edges_plant.info()
# edges_product_group_info = edges_product_group.info()
# edges_product_subgroup_info = edges_product_subgroup.info()
# edges_storage_location_info = edges_storage_location.info()
# nodes_info = nodes.info()
# nodes_index_info = nodes_index.info()
# nodes_type_plant_storage_info = nodes_type_plant_storage.info()
# nodes_type_product_group_subgroup_info = nodes_type_product_group_subgroup.info()

# edges_plant_head = edges_plant.head()
# edges_product_group_head = edges_product_group.head()
# edges_product_subgroup_head = edges_product_subgroup.head()
# edges_storage_location_head = edges_storage_location.head()
# nodes_head = nodes.head()
# nodes_index_head = nodes_index.head()
# nodes_type_plant_storage_head = nodes_type_plant_storage.head()
# nodes_type_product_group_subgroup_head = nodes_type_product_group_subgroup.head()

# (
#     edges_plant_info, edges_product_group_info, edges_product_subgroup_info, edges_storage_location_info, 
#     nodes_info, nodes_index_info, nodes_type_plant_storage_info, nodes_type_product_group_subgroup_info,
#     edges_plant_head, edges_product_group_head, edges_product_subgroup_head, edges_storage_location_head,
#     nodes_head, nodes_index_head, nodes_type_plant_storage_head, nodes_type_product_group_subgroup_head
# )


import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader

class MultiModalFusionDataset(Dataset):
    def __init__(self, node_features, edge_features, heterogeneity_features, window_size, target_column):
        """
        Args:
            node_features: DataFrame containing node-level features.
            edge_features: DataFrame containing edge-level features.
            heterogeneity_features: Columns in edge_features representing heterogeneity.
            window_size: Number of time steps in the sliding window (if temporal data exists).
            target_column: The column to be predicted.
        """
        self.node_features = node_features
        self.edge_features = edge_features
        self.heterogeneity_features = heterogeneity_features
        self.window_size = window_size
        self.target_column = target_column

        # Prepare features and targets
        self.features, self.targets = self._prepare_data()

    def _prepare_data(self):
        features, targets = [], []

        for _, edge in self.edge_features.iterrows():
            node1, node2 = edge['node1'], edge['node2']
            heterogeneity = edge[self.heterogeneity_features].values

            if node1 in self.node_features.index and node2 in self.node_features.index:
                node1_features = self.node_features.loc[node1].values
                node2_features = self.node_features.loc[node2].values

                feature_vector = np.hstack([node1_features, node2_features, heterogeneity])
                target = edge[self.target_column] if self.target_column in edge else 0

                features.append(feature_vector)
                targets.append(target)

        return np.array(features), np.array(targets)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]

# Load edge data
edges_plant = pd.read_csv('/mnt/data/Edges (Plant).csv')
edges_product_group = pd.read_csv('/mnt/data/Edges (Product Group).csv')
edges_product_subgroup = pd.read_csv('/mnt/data/Edges (Product Sub-Group).csv')
edges_storage_location = pd.read_csv('/mnt/data/Edges (Storage Location).csv')

# Combine edge data
edge_features = pd.concat([edges_plant, edges_product_group, edges_product_subgroup, edges_storage_location], ignore_index=True)

# Load node data
nodes = pd.read_csv('/mnt/data/Nodes.csv')
nodes_index = pd.read_csv('/mnt/data/NodesIndex.csv')
nodes_type_plant_storage = pd.read_csv('/mnt/data/Nodes Type (Plant & Storage).csv')
nodes_type_product_group_subgroup = pd.read_csv('/mnt/data/Node Types (Product Group and Subgroup).csv')

# Merge node data into a single DataFrame
node_features = nodes.set_index('Node')
node_features = node_features.join(nodes_index.set_index('Node'))
node_features = node_features.join(nodes_type_plant_storage.set_index('Node'))
node_features = node_features.join(nodes_type_product_group_subgroup.set_index('Node'))

# Parameters
heterogeneity_columns = ['Plant', 'GroupCode', 'SubGroupCode', 'Storage Location']
window_size = 5  # If temporal data exists, this is used

target_column = 'Storage Location'  # Example target column for demonstration

# Create dataset
fusion_dataset = MultiModalFusionDataset(node_features, edge_features, heterogeneity_columns, window_size, target_column)

# Create dataloader
batch_size = 32
data_loader = DataLoader(fusion_dataset, batch_size=batch_size, shuffle=True)

print("Multi-modal fusion dataloader created.")

