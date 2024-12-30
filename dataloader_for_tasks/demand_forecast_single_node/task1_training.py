from task1_data_handler import DataHandler
from forecasting_models import MLPModel, GNNModel, GCNModel, train_model
import torch
import torch.nn as nn
import torch.optim as optim

# File paths for the dataset
file_paths = {
    # 'nodes': '/home/meow/SupplyGraph/RawDataset/Homogenoeus/Nodes/Nodes.csv', # do not need node information
    'sales_order': '/home/meow/SupplyGraph/RawDataset/Homogenoeus/Temporal Data/Unit/Sales Order.csv',
    'delivery_to_distributor': '/home/meow/SupplyGraph/RawDataset/Homogenoeus/Temporal Data/Unit/Delivery To distributor.csv',
    'factory_issue': '/home/meow/SupplyGraph/RawDataset/Homogenoeus/Temporal Data/Unit/Factory Issue.csv',
    'production': '/home/meow/SupplyGraph/RawDataset/Homogenoeus/Temporal Data/Unit/Production .csv'
}

# Initialize DataHandler
target_column = 'SOS008L02P_delivery_to_distributor'
window_size = 5
batch_size = 32

data_handler = DataHandler(file_paths, target_column, window_size, batch_size)
train_loader, val_loader, test_loader = data_handler.prepare_dataloaders()


# Instantiate and train models
# input_size = train_dataset[0][0].shape[1]
input_size = len(train_loader.dataset[0][0])
output_size = 1
hidden_size = 64
num_epochs = 50

# Define your models and train as before
mlp_model = MLPModel(input_size=window_size * len(train_loader.dataset[0][0]), hidden_size=64, output_size=1)
optimizer = optim.Adam(mlp_model.parameters(), lr=0.001)
criterion = nn.MSELoss()
train_model(mlp_model, train_loader, val_loader, optimizer, criterion, num_epochs)

# GNN Model
gnn_model = GNNModel(input_size * window_size, hidden_size, output_size)
adjacency_matrix = torch.eye(input_size * window_size)  # Dummy adjacency matrix for single node
optimizer_gnn = optim.Adam(gnn_model.parameters(), lr=0.001)
train_model(gnn_model, train_loader, val_loader, optimizer_gnn, criterion, num_epochs)

# GCN Model
gcn_model = GCNModel(input_size * window_size, hidden_size, output_size)
adjacency_matrix_gcn = torch.eye(input_size * window_size)  # Dummy adjacency matrix for single node
optimizer_gcn = optim.Adam(gcn_model.parameters(), lr=0.001)
train_model(gcn_model, train_loader, val_loader, optimizer_gcn, criterion, num_epochs)

print("Training complete.")