import pandas as pd
import os

# Paths to folders
edges_path = "/home/meow/SupplyGraph/Raw Dataset/Homogeneous/Edges"
nodes_path = "/home/meow/SupplyGraph/Raw Dataset/Homogeneous/Nodes"
temporal_data_path = "/home/meow/SupplyGraph/Raw Dataset/Homogeneous/Temporal Data"

# Helper function to remove duplicates and save
def clean_csv(file_path):
    df = pd.read_csv(file_path)
    df.drop_duplicates(inplace=True)
    df.to_csv(file_path, index=False)
    print(f"Cleaned: {file_path}")

# Step 1: Clean Edge Data
edge_files = [f for f in os.listdir(edges_path) if f.endswith('.csv')]
for edge_file in edge_files:
    clean_csv(os.path.join(edges_path, edge_file))

# Step 2: Clean Node Data
node_files = [f for f in os.listdir(nodes_path) if f.endswith('.csv')]
for node_file in node_files:
    clean_csv(os.path.join(nodes_path, node_file))

# Step 3: Temporal Data Cleaning
temporal_files = [f for f in os.listdir(temporal_data_path) if f.endswith('.csv')]
for temporal_file in temporal_files:
    file_path = os.path.join(temporal_data_path, temporal_file)
    df = pd.read_csv(file_path)

    # Remove rows with missing values
    df.dropna(inplace=True)

    # Normalize features (z-score)
    for col in df.columns[1:]:  # Skip the first column (e.g., time index)
        if pd.api.types.is_numeric_dtype(df[col]):
            df[col] = (df[col] - df[col].mean()) / df[col].std()

    # Save cleaned data
    df.to_csv(file_path, index=False)
    print(f"Cleaned and normalized: {file_path}")

# Step 4: Remove Low-Quality Nodes (Nodes with predominantly zero temporal features)
nodes_df = pd.read_csv(os.path.join(nodes_path, "Nodes.csv"))
cleaned_temporal_data = []

for temporal_file in temporal_files:
    file_path = os.path.join(temporal_data_path, temporal_file)
    df = pd.read_csv(file_path)

    # Identify nodes with mostly zero values
    zero_threshold = 0.9  # 90% of values are zero
    zero_nodes = df.loc[:, (df == 0).mean() > zero_threshold].columns

    # Remove these nodes from the temporal data
    df.drop(columns=zero_nodes, inplace=True, errors='ignore')
    df.to_csv(file_path, index=False)
    cleaned_temporal_data.append(df)
    print(f"Removed low-quality nodes from: {file_path}")

# Optional: Validate Node-Edge Mapping
nodes_df = pd.read_csv(os.path.join(nodes_path, "Nodes.csv"))
for edge_file in edge_files:
    edge_path = os.path.join(edges_path, edge_file)
    edges_df = pd.read_csv(edge_path)

    # Check if all nodes in edges exist in the node file
    invalid_nodes = set(edges_df['Source']).union(edges_df['Target']) - set(nodes_df['NodeID'])
    if invalid_nodes:
        print(f"Invalid nodes found in {edge_file}: {invalid_nodes}")
    else:
        print(f"Node-edge mapping validated for {edge_file}")

print("Data cleaning completed!")
