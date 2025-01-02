from task1_data_handler import DataHandler
from forecasting_models import MLPModel, GNNModel, GCNModel, train_model, train_gnn_model
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd

# File paths for the dataset
file_paths = {
    # 'nodes': '/home/meow/SupplyGraph/RawDataset/Homogenoeus/Nodes/Nodes.csv', # do not need node information
    'sales_order': '/home/meow/SupplyGraph/RawDataset/Homogenoeus/Temporal Data/Unit/Sales Order.csv',
    'delivery_to_distributor': '/home/meow/SupplyGraph/RawDataset/Homogenoeus/Temporal Data/Unit/Delivery To distributor.csv',
    'factory_issue': '/home/meow/SupplyGraph/RawDataset/Homogenoeus/Temporal Data/Unit/Factory Issue.csv',
    'production': '/home/meow/SupplyGraph/RawDataset/Homogenoeus/Temporal Data/Unit/Production .csv'
}



# Load the header from the processed_data.csv
processed_data_path = '/home/meow/SupplyGraph/RawDataset/Homogenoeus/task1_results/processed_data.csv'
# Initialize DataHandler
# target_column = 'SOS008L02P_delivery_to_distributor'

processed_data = pd.read_csv(processed_data_path)
headers = processed_data.columns.tolist()

# Loop through each target column in the header
for target_column in headers:
    if "date" in target_column.lower() or ".1" in target_column:
        print(f"Skipping target column {target_column} as it contains 'date'.")
        continue
    print(f"Training for target column: {target_column}")

    window_size = 5
    batch_size = 32

    # Initialize DataHandler
    data_handler = DataHandler(file_paths, target_column, window_size, batch_size)
    train_loader, val_loader, test_loader = data_handler.prepare_dataloaders()

    # Instantiate and train models
    input_size = len(train_loader.dataset[0][0])
    # print("input_size :", input_size)
    output_size = 1
    hidden_size = 64
    num_epochs = 50

    criterion = nn.MSELoss()

    # MLP Model
    mlp_model = MLPModel(input_size=window_size * 164, hidden_size=64, output_size=1)
    optimizer = optim.Adam(mlp_model.parameters(), lr=0.001)
    train_model(mlp_model, train_loader, val_loader, test_loader, optimizer, criterion, num_epochs, target_column, "mlp")

    # GNN Model
    num_nodes = 164 * window_size # 164 nodes
    adjacency_matrix = torch.eye(num_nodes)  # Identity matrix for window_size=5
    gnn_model = GNNModel(164 * window_size, hidden_size, output_size)
    optimizer_gnn = optim.Adam(gnn_model.parameters(), lr=0.001)
    train_gnn_model(gnn_model, train_loader, val_loader, test_loader, optimizer_gnn, criterion, num_epochs, target_column, "gnn", adjacency_matrix)


    # # GCN Model
    num_nodes = 164 * window_size # 164 nodes
    gcn_model = GCNModel(164 * window_size, hidden_size, output_size)
    adjacency_matrix = torch.eye(num_nodes)  # Dummy adjacency matrix for single node
    optimizer_gcn = optim.Adam(gcn_model.parameters(), lr=0.001)
    train_gnn_model(gcn_model, train_loader, val_loader, test_loader, optimizer_gcn, criterion, num_epochs, target_column, "gcn", adjacency_matrix)

    print(f"Training complete for target column: {target_column}\n")

print("Training complete.")