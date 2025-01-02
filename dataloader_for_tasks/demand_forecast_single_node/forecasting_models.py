import torch
import torchmetrics
import torch.nn as nn
import torch.optim as optim

# Normalize and denormalize helper functions
def normalize_targets(targets, min_val, max_val):
    return (targets - min_val) / (max_val - min_val)

def denormalize_targets(normalized, min_val, max_val):
    return normalized * (max_val - min_val) + min_val


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
        batch_size = x.size(0)  # Get batch size
        num_nodes = adjacency_matrix.size(0)  # Number of nodes
        
        # Reshape input for adjacency matrix multiplication
        x = x.view(batch_size, num_nodes, -1)  # (batch_size, num_nodes, features_per_node)

        # Expand adjacency matrix for batch processing
        adjacency_matrix = adjacency_matrix.unsqueeze(0).expand(batch_size, -1, -1)

        # Perform batch-wise matrix multiplication
        x = torch.bmm(adjacency_matrix, x)

        # Flatten back for fully connected layers
        x = x.view(batch_size, -1)

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
        batch_size = x.size(0)  # Get batch size
        num_nodes = adjacency_matrix.size(0)  # Number of nodes

        # Reshape x to (batch_size, num_nodes, features_per_node)
        x = x.view(batch_size, num_nodes, -1)

        # Compute degree matrix and laplacian for each batch
        adjacency_matrix = adjacency_matrix.unsqueeze(0).expand(batch_size, -1, -1)
        degree_matrix = torch.diag_embed(torch.sum(adjacency_matrix, dim=2))
        laplacian = degree_matrix - adjacency_matrix

        # Perform graph convolution: D^(-1) * L * x
        degree_matrix_inv = torch.linalg.pinv(degree_matrix)  # Use pseudo-inverse for stability
        x = torch.bmm(degree_matrix_inv, torch.bmm(laplacian, x))

        # Flatten x back for fully connected layers
        x = x.view(batch_size, -1)

        # Apply fully connected layers
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x



def train_model(model, train_loader, val_loader, test_loader, optimizer, criterion, num_epochs, target_column, model_name):
    # Calculate min and max values for target normalization
    all_targets = torch.cat([targets for _, targets in train_loader], dim=0)
    min_target = all_targets.min().item()
    max_target = all_targets.max().item()

    # Helper functions for normalization and denormalization
    def normalize_targets(targets):
        return (targets - min_target) / (max_target - min_target)

    def denormalize_targets(normalized):
        return normalized * (max_target - min_target) + min_target

    # Initialize a file for saving results
    results_filename = f"/home/meow/SupplyGraph/RawDataset/Homogenoeus/task1_results/{model_name}/results_{model_name}_{target_column}.txt"
    with open(results_filename, "w") as results_file:
        for epoch in range(num_epochs):
            # Training Phase
            model.train()
            train_loss = 0.0
            train_mse = 0.0
            train_mae = 0.0
            for features, targets in train_loader:
                features = features.float().view(features.size(0), -1)  # Flatten features
                targets = normalize_targets(targets.float().unsqueeze(1))  # Normalize targets
                optimizer.zero_grad()
                outputs = model(features)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
                
                # Metrics (using normalized values)
                train_mse += torch.mean((outputs - targets) ** 2).item()
                train_mae += torch.mean(torch.abs(outputs - targets)).item()

            # Validation Phase
            val_loss = 0.0
            val_mse_normalized = 0.0
            val_mae_normalized = 0.0
            val_mse_denormalized = 0.0
            val_mae_denormalized = 0.0
            total_targets = []
            total_predictions = []
            model.eval()
            with torch.no_grad():
                for features, targets in val_loader:
                    features = features.float().view(features.size(0), -1)  # Flatten features
                    targets = normalize_targets(targets.float().unsqueeze(1))  # Normalize targets
                    outputs = model(features)
                    loss = criterion(outputs, targets)
                    val_loss += loss.item()

                    # Denormalize outputs and targets for metrics
                    outputs_denorm = denormalize_targets(outputs)
                    targets_denorm = denormalize_targets(targets)

                    # Metrics (both normalized and denormalized)
                    val_mse_normalized += torch.mean((outputs - targets) ** 2).item()
                    val_mae_normalized += torch.mean(torch.abs(outputs - targets)).item()
                    val_mse_denormalized += torch.mean((outputs_denorm - targets_denorm) ** 2).item()
                    val_mae_denormalized += torch.mean(torch.abs(outputs_denorm - targets_denorm)).item()
                    total_targets.append(targets_denorm)
                    total_predictions.append(outputs_denorm)
            
            # Calculate R? (coefficient of determination)
            total_targets = torch.cat(total_targets)
            total_predictions = torch.cat(total_predictions)
            ss_total = torch.sum((total_targets - total_targets.mean()) ** 2)
            ss_residual = torch.sum((total_targets - total_predictions) ** 2)
            val_r2 = 1 - (ss_residual / ss_total).item()

            # Logging to console
            print(f"Epoch {epoch+1}/{num_epochs}")
            print(f"  Train Loss: {train_loss/len(train_loader):.4f}, Train MSE: {train_mse/len(train_loader):.4f}, Train MAE: {train_mae/len(train_loader):.4f}")
            print(f"  Val Loss: {val_loss/len(val_loader):.4f}")
            print(f"  Val MSE (Normalized): {val_mse_normalized/len(val_loader):.4f}, Val MAE (Normalized): {val_mae_normalized/len(val_loader):.4f}")
            print(f"  Val MSE (Denormalized): {val_mse_denormalized/len(val_loader):.4f}, Val MAE (Denormalized): {val_mae_denormalized/len(val_loader):.4f}, Val R?: {val_r2:.4f}")

            # Logging to file
            results_file.write(f"Epoch {epoch+1}/{num_epochs}\n")
            results_file.write(f"  Train Loss: {train_loss/len(train_loader):.4f}, Train MSE: {train_mse/len(train_loader):.4f}, Train MAE: {train_mae/len(train_loader):.4f}\n")
            results_file.write(f"  Val Loss: {val_loss/len(val_loader):.4f}\n")
            results_file.write(f"  Val MSE (Normalized): {val_mse_normalized/len(val_loader):.4f}, Val MAE (Normalized): {val_mae_normalized/len(val_loader):.4f}\n")
            results_file.write(f"  Val MSE (Denormalized): {val_mse_denormalized/len(val_loader):.4f}, Val MAE (Denormalized): {val_mae_denormalized/len(val_loader):.4f}, Val R?: {val_r2:.4f}\n\n")
        
        print(f"Results saved to {results_filename}")

       # Test Phase
        test_loss = 0.0
        test_mse_normalized = 0.0
        test_mae_normalized = 0.0
        test_mse_denormalized = 0.0
        test_mae_denormalized = 0.0
        model.eval()
        with torch.no_grad():
            for features, targets in test_loader:
                features = features.float().view(features.size(0), -1)  # Flatten features
                targets = normalize_targets(targets.float().unsqueeze(1))  # Normalize targets
                outputs = model(features)
                outputs_denorm = denormalize_targets(outputs)
                targets_denorm = denormalize_targets(targets)

                # Metrics (both normalized and denormalized)
                test_loss += criterion(outputs, targets).item()
                test_mse_normalized += torch.mean((outputs - targets) ** 2).item()
                test_mae_normalized += torch.mean(torch.abs(outputs - targets)).item()
                test_mse_denormalized += torch.mean((outputs_denorm - targets_denorm) ** 2).item()
                test_mae_denormalized += torch.mean(torch.abs(outputs_denorm - targets_denorm)).item()

        print(f"Test Results")
        print(f"  Test Loss: {test_loss/len(test_loader):.4f}")
        print(f"  Test MSE (Normalized): {test_mse_normalized/len(test_loader):.4f}, Test MAE (Normalized): {test_mae_normalized/len(test_loader):.4f}")
        print(f"  Test MSE (Denormalized): {test_mse_denormalized/len(test_loader):.4f}, Test MAE (Denormalized): {test_mae_denormalized/len(test_loader):.4f}")
        results_file.write(f"Test Results\n")
        results_file.write(f"  Test Loss: {test_loss/len(test_loader):.4f}\n")
        results_file.write(f"  Test MSE (Normalized): {test_mse_normalized/len(test_loader):.4f}, Test MAE (Normalized): {test_mae_normalized/len(test_loader):.4f}\n")
        results_file.write(f"  Test MSE (Denormalized): {test_mse_denormalized/len(test_loader):.4f}, Test MAE (Denormalized): {test_mae_denormalized/len(test_loader):.4f}\n")




def train_gnn_model(
    model, train_loader, val_loader, test_loader, optimizer, criterion, num_epochs, target_column, model_name, adjacency_matrix):
    # Calculate min and max values for target normalization
    all_targets = torch.cat([targets for _, targets in train_loader], dim=0)
    min_target = all_targets.min().item()
    max_target = all_targets.max().item()

    # Helper functions for normalization and denormalization
    def normalize_targets(targets):
        return (targets - min_target) / (max_target - min_target)

    def denormalize_targets(normalized):
        return normalized * (max_target - min_target) + min_target

    # Initialize a file for saving results
    results_filename = f"/home/meow/SupplyGraph/RawDataset/Homogenoeus/task1_results/{model_name}/results_{model_name}_{target_column}.txt"
    with open(results_filename, "w") as results_file:
        for epoch in range(num_epochs):
            # Training Phase
            model.train()
            train_loss = 0.0
            train_mse = 0.0
            train_mae = 0.0
            for features, targets in train_loader:
                features = features.float().view(features.size(0), -1)  # Flatten features
                targets = normalize_targets(targets.float().unsqueeze(1))  # Normalize targets
                optimizer.zero_grad()
                outputs = model(features, adjacency_matrix)  # Pass adjacency_matrix here
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
                
                # Metrics (using normalized values)
                train_mse += torch.mean((outputs - targets) ** 2).item()
                train_mae += torch.mean(torch.abs(outputs - targets)).item()

            # Validation Phase
            val_loss = 0.0
            val_mse_normalized = 0.0
            val_mae_normalized = 0.0
            val_mse_denormalized = 0.0
            val_mae_denormalized = 0.0
            total_targets = []
            total_predictions = []
            model.eval()
            with torch.no_grad():
                for features, targets in val_loader:
                    features = features.float().view(features.size(0), -1)  # Flatten features
                    targets = normalize_targets(targets.float().unsqueeze(1))  # Normalize targets
                    outputs = model(features, adjacency_matrix)  # Pass adjacency_matrix here
                    loss = criterion(outputs, targets)
                    val_loss += loss.item()

                    # Denormalize outputs and targets for metrics
                    outputs_denorm = denormalize_targets(outputs)
                    targets_denorm = denormalize_targets(targets)

                    # Metrics
                    val_mse_normalized += torch.mean((outputs - targets) ** 2).item()
                    val_mae_normalized += torch.mean(torch.abs(outputs - targets)).item()
                    val_mse_denormalized += torch.mean((outputs_denorm - targets_denorm) ** 2).item()
                    val_mae_denormalized += torch.mean(torch.abs(outputs_denorm - targets_denorm)).item()

                    total_targets.append(targets_denorm)
                    total_predictions.append(outputs_denorm)
            
            # Calculate R? (coefficient of determination)
            total_targets = torch.cat(total_targets)
            total_predictions = torch.cat(total_predictions)
            ss_total = torch.sum((total_targets - total_targets.mean()) ** 2)
            ss_residual = torch.sum((total_targets - total_predictions) ** 2)
            val_r2 = 1 - (ss_residual / ss_total).item()

            # Logging to console
            print(f"Epoch {epoch+1}/{num_epochs}")
            print(f"  Train Loss: {train_loss/len(train_loader):.4f}, Train MSE: {train_mse/len(train_loader):.4f}, Train MAE: {train_mae/len(train_loader):.4f}")
            print(f"  Val Loss: {val_loss/len(val_loader):.4f}, Val MSE (Normalized): {val_mse_normalized/len(val_loader):.4f}, Val MAE (Normalized): {val_mae_normalized/len(val_loader):.4f}")
            print(f"  Val MSE (Denormalized): {val_mse_denormalized/len(val_loader):.4f}, Val MAE (Denormalized): {val_mae_denormalized/len(val_loader):.4f}, Val R?: {val_r2:.4f}")

            # Logging to file
            results_file.write(f"Epoch {epoch+1}/{num_epochs}\n")
            results_file.write(f"  Train Loss: {train_loss/len(train_loader):.4f}, Train MSE: {train_mse/len(train_loader):.4f}, Train MAE: {train_mae/len(train_loader):.4f}\n")
            results_file.write(f"  Val Loss: {val_loss/len(val_loader):.4f}, Val MSE (Normalized): {val_mse_normalized/len(val_loader):.4f}, Val MAE (Normalized): {val_mae_normalized/len(val_loader):.4f}\n")
            results_file.write(f"  Val MSE (Denormalized): {val_mse_denormalized/len(val_loader):.4f}, Val MAE (Denormalized): {val_mae_denormalized/len(val_loader):.4f}, Val R?: {val_r2:.4f}\n\n")
        
        print(f"Results saved to {results_filename}")

        # Test Phase
        test_loss = 0.0
        test_mse_normalized = 0.0
        test_mae_normalized = 0.0
        test_mse_denormalized = 0.0
        test_mae_denormalized = 0.0
        model.eval()
        with torch.no_grad():
            for features, targets in test_loader:
                features = features.float().view(features.size(0), -1)  # Flatten features
                targets = normalize_targets(targets.float().unsqueeze(1))  # Normalize targets
                outputs = model(features, adjacency_matrix)  # Pass adjacency_matrix here
                loss = criterion(outputs, targets)
                test_loss += loss.item()

                # Denormalize outputs and targets for metrics
                outputs_denorm = denormalize_targets(outputs)
                targets_denorm = denormalize_targets(targets)

                # Metrics (both normalized and denormalized)
                test_mse_normalized += torch.mean((outputs - targets) ** 2).item()
                test_mae_normalized += torch.mean(torch.abs(outputs - targets)).item()
                test_mse_denormalized += torch.mean((outputs_denorm - targets_denorm) ** 2).item()
                test_mae_denormalized += torch.mean(torch.abs(outputs_denorm - targets_denorm)).item()

        print(f"Test Results")
        print(f"  Test Loss: {test_loss/len(test_loader):.4f}")
        print(f"  Test MSE (Normalized): {test_mse_normalized/len(test_loader):.4f}, Test MAE (Normalized): {test_mae_normalized/len(test_loader):.4f}")
        print(f"  Test MSE (Denormalized): {test_mse_denormalized/len(test_loader):.4f}, Test MAE (Denormalized): {test_mae_denormalized/len(test_loader):.4f}")
        results_file.write(f"Test Results\n")
        results_file.write(f"  Test Loss: {test_loss/len(test_loader):.4f}\n")
        results_file.write(f"  Test MSE (Normalized): {test_mse_normalized/len(test_loader):.4f}, Test MAE (Normalized): {test_mae_normalized/len(test_loader):.4f}\n")
        results_file.write(f"  Test MSE (Denormalized): {test_mse_denormalized/len(test_loader):.4f}, Test MAE (Denormalized): {test_mae_denormalized/len(test_loader):.4f}\n")
