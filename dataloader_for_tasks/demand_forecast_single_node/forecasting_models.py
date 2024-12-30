import torch
import torch.nn as nn
import torch.optim as optim

# Define MLP Model
class MLPModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLPModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x

# Define GNN Model
class GNNModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(GNNModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)

    def forward(self, x, adjacency_matrix):
        x = torch.mm(adjacency_matrix, x)  # Message passing
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x

# Define GCN Model
class GCNModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(GCNModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)

    def forward(self, x, adjacency_matrix):
        degree_matrix = torch.diag(torch.sum(adjacency_matrix, dim=1))
        laplacian = degree_matrix - adjacency_matrix
        x = torch.mm(torch.inverse(degree_matrix), torch.mm(laplacian, x))  # Graph convolution
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x



# Training loop
# def train_model(model, train_loader, val_loader, optimizer, criterion, num_epochs):
#     for epoch in range(num_epochs):
#         model.train()
#         train_loss = 0.0
#         for features, targets in train_loader:
#             for features, _ in train_loader:
#                 print(f"Feature shape: {features.shape}")  # Add this line to debug
#                 break

#             # input_size = features.shape[1]  # Ensure input size matches the actual data

#             features = features.float()
#             targets = targets.float().unsqueeze(1)
#             optimizer.zero_grad()
#             outputs = model(features)
#             loss = criterion(outputs, targets)
#             loss.backward()
#             optimizer.step()
#             train_loss += loss.item()

#         val_loss = 0.0
#         model.eval()
#         with torch.no_grad():
#             for features, targets in val_loader:
#                 features = features.float()
#                 targets = targets.float().unsqueeze(1)
#                 outputs = model(features)
#                 loss = criterion(outputs, targets)
#                 val_loss += loss.item()

#         print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss/len(train_loader)}, Val Loss: {val_loss/len(val_loader)}")


# import torch
# import torchmetrics

# def train_model(model, train_loader, val_loader, optimizer, criterion, num_epochs):
#     for epoch in range(num_epochs):
#         # Training Phase
#         model.train()
#         train_loss = 0.0
#         train_mse = 0.0
#         train_mae = 0.0
#         for features, targets in train_loader:
#             features = features.float().view(features.size(0), -1)  # Flatten features
#             targets = targets.float().unsqueeze(1)
#             optimizer.zero_grad()
#             outputs = model(features)
#             loss = criterion(outputs, targets)
#             loss.backward()
#             optimizer.step()
#             train_loss += loss.item()
            
#             # Metrics
#             train_mse += torch.mean((outputs - targets) ** 2).item()
#             train_mae += torch.mean(torch.abs(outputs - targets)).item()

#         # Validation Phase
#         val_loss = 0.0
#         val_mse = 0.0
#         val_mae = 0.0
#         total_targets = []
#         total_predictions = []
#         model.eval()
#         with torch.no_grad():
#             for features, targets in val_loader:
#                 features = features.float().view(features.size(0), -1)  # Flatten features
#                 targets = targets.float().unsqueeze(1)
#                 outputs = model(features)
#                 loss = criterion(outputs, targets)
#                 val_loss += loss.item()

#                 # Metrics
#                 val_mse += torch.mean((outputs - targets) ** 2).item()
#                 val_mae += torch.mean(torch.abs(outputs - targets)).item()
#                 total_targets.append(targets)
#                 total_predictions.append(outputs)
        
#         # Calculate R? (coefficient of determination)
#         total_targets = torch.cat(total_targets)
#         total_predictions = torch.cat(total_predictions)
#         ss_total = torch.sum((total_targets - total_targets.mean()) ** 2)
#         ss_residual = torch.sum((total_targets - total_predictions) ** 2)
#         val_r2 = 1 - (ss_residual / ss_total).item()

#         # Logging
#         print(f"Epoch {epoch+1}/{num_epochs}")
#         print(f"  Train Loss: {train_loss/len(train_loader):.4f}, Train MSE: {train_mse/len(train_loader):.4f}, Train MAE: {train_mae/len(train_loader):.4f}")
#         print(f"  Val Loss: {val_loss/len(val_loader):.4f}, Val MSE: {val_mse/len(val_loader):.4f}, Val MAE: {val_mae/len(val_loader):.4f}, Val R?: {val_r2:.4f}")






import torch
import torchmetrics

def train_model(model, train_loader, val_loader, optimizer, criterion, num_epochs, target_column, model_name):
    # Initialize a file for saving results
    results_filename = f"/home/meow/SupplyGraph/RawDataset/Homogenoeus/results_{model_name}_{target_column}.txt"
    with open(results_filename, "w") as results_file:
        for epoch in range(num_epochs):
            # Training Phase
            model.train()
            train_loss = 0.0
            train_mse = 0.0
            train_mae = 0.0
            for features, targets in train_loader:
                features = features.float().view(features.size(0), -1)  # Flatten features
                targets = targets.float().unsqueeze(1)
                optimizer.zero_grad()
                outputs = model(features)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
                
                # Metrics
                train_mse += torch.mean((outputs - targets) ** 2).item()
                train_mae += torch.mean(torch.abs(outputs - targets)).item()

            # Validation Phase
            val_loss = 0.0
            val_mse = 0.0
            val_mae = 0.0
            total_targets = []
            total_predictions = []
            model.eval()
            with torch.no_grad():
                for features, targets in val_loader:
                    features = features.float().view(features.size(0), -1)  # Flatten features
                    targets = targets.float().unsqueeze(1)
                    outputs = model(features)
                    loss = criterion(outputs, targets)
                    val_loss += loss.item()

                    # Metrics
                    val_mse += torch.mean((outputs - targets) ** 2).item()
                    val_mae += torch.mean(torch.abs(outputs - targets)).item()
                    total_targets.append(targets)
                    total_predictions.append(outputs)
            
            # Calculate R? (coefficient of determination)
            total_targets = torch.cat(total_targets)
            total_predictions = torch.cat(total_predictions)
            ss_total = torch.sum((total_targets - total_targets.mean()) ** 2)
            ss_residual = torch.sum((total_targets - total_predictions) ** 2)
            val_r2 = 1 - (ss_residual / ss_total).item()

            # Logging to console
            print(f"Epoch {epoch+1}/{num_epochs}")
            print(f"  Train Loss: {train_loss/len(train_loader):.4f}, Train MSE: {train_mse/len(train_loader):.4f}, Train MAE: {train_mae/len(train_loader):.4f}")
            print(f"  Val Loss: {val_loss/len(val_loader):.4f}, Val MSE: {val_mse/len(val_loader):.4f}, Val MAE: {val_mae/len(val_loader):.4f}, Val R?: {val_r2:.4f}")

            # Logging to file
            results_file.write(f"Epoch {epoch+1}/{num_epochs}\n")
            results_file.write(f"  Train Loss: {train_loss/len(train_loader):.4f}, Train MSE: {train_mse/len(train_loader):.4f}, Train MAE: {train_mae/len(train_loader):.4f}\n")
            results_file.write(f"  Val Loss: {val_loss/len(val_loader):.4f}, Val MSE: {val_mse/len(val_loader):.4f}, Val MAE: {val_mae/len(val_loader):.4f}, Val R?: {val_r2:.4f}\n\n")
        
        print(f"Results saved to {results_filename}")
