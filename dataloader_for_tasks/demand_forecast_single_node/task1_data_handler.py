import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class TemporalDataset(Dataset):
    def __init__(self, data, target_column, window_size):
        # Ensure the DataFrame is not empty
        if data.empty:
            raise ValueError("The input DataFrame is empty.")
        
        # Check if target column exists
        if target_column not in data.columns:
            raise ValueError(f"Target column '{target_column}' not found in the DataFrame.")
        
        # Ensure target column is numeric
        # data[target_column] = pd.to_numeric(data[target_column], errors='coerce')
        
        # Drop rows with missing values in the target column
        # data.dropna(subset=[target_column], inplace=True)
        data = data.dropna(subset=[target_column])

        
        # Keep only numeric columns for features
        self.data = data.select_dtypes(include=[np.number])
        self.target_column = target_column
        self.window_size = window_size

        # Prepare features and targets
        self.features, self.targets = self._prepare_data()

    def _prepare_data(self):
        features, targets = [], []
        for i in range(len(self.data) - self.window_size):
            feature_window = self.data.iloc[i:i + self.window_size].values.astype(np.float32)
            target = self.data.iloc[i + self.window_size][self.target_column]
            features.append(feature_window)
            targets.append(target)
        return np.array(features, dtype=np.float32), np.array(targets, dtype=np.float32)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.features[idx], dtype=torch.float32),
            torch.tensor(self.targets[idx], dtype=torch.float32)
        )


class DataHandler:
    def __init__(self, file_paths, target_column, window_size, batch_size, train_ratio=0.7, val_ratio=0.2):
        self.file_paths = file_paths
        self.target_column = target_column
        self.window_size = window_size
        self.batch_size = batch_size
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio

    def load_data(self):
        # Load the datasets
        datasets = {}
        for key, path in self.file_paths.items():
            datasets[key] = pd.read_csv(path)
            datasets[key].set_index('Date', inplace=True)

        # Combine all datasets into a single DataFrame
        data = datasets['sales_order']
        for key in ['delivery_to_distributor', 'factory_issue', 'production']:
            data = data.join(datasets[key], how='inner', lsuffix=f'_{key}', rsuffix=f'_{key}')

        data.reset_index(inplace=True)
        return data

    def save_to_csv(self, data, output_path):
        # Save the processed data to a CSV file
        data.to_csv(output_path, index=False)
        print(f"Data saved to {output_path}")

    def prepare_dataloaders(self):
        data = self.load_data()

        # self.save_to_csv(data, "/home/meow/SupplyGraph/RawDataset/Homogenoeus/processed_data.csv")

        # Split the dataset into train, validation, and test sets
        train_size = int(len(data) * self.train_ratio)
        val_size = int(len(data) * self.val_ratio)
        test_size = len(data) - train_size - val_size

        train_data = data.iloc[:train_size]
        val_data = data.iloc[train_size:train_size + val_size]
        test_data = data.iloc[train_size + val_size:]

        # Create PyTorch datasets
        train_dataset = TemporalDataset(train_data, self.target_column, self.window_size)
        val_dataset = TemporalDataset(val_data, self.target_column, self.window_size)
        test_dataset = TemporalDataset(test_data, self.target_column, self.window_size)

        # Create DataLoaders
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)

        return train_loader, val_loader, test_loader
