import torch
import torch.nn as nn
from torch.utils.data import Dataset,DataLoader, TensorDataset, RandomSampler, random_split
from torch.optim import Adam, LBFGS

class Cus_Dataset(Dataset):
    def __init__(self,df, feature_columns,target_column,train_ratio=0.8,\
                    val_ratio=0.1, test_ratio=0.1,split='train'):
        """
        Initialize the dataset with a pandas DataFrame
        
        Parameters:
        -----------
        df : pandas.DataFrame
            Input DataFrame
        feature_columns : list
            List of column names to use as features
        target_column : str
            Name of the column to use as target
        train : bool
            Whether to use training or testing split
        train_ratio : float
            Ratio of data to use for training (default: 0.8)
        val_ratio : float
            Ratio of data for validation
        test_ratio : float
            Ratio of data to use for test
        random_seed : int
            Random seed for reproducibility (default: 42)
        """
        # Extract features and labels from DataFrame
        self.features = torch.FloatTensor(df[feature_columns].values)
        self.labels = torch.FloatTensor(df[target_column].values)
        
        # Calculate split sizes
        total_size = len(self.features)
        train_size = int(train_ratio * total_size)
        test_size = int(total_size * test_ratio)
        val_size = int(val_ratio * total_size)
        
        # Create the full dataset
        full_dataset = torch.utils.data.TensorDataset(self.features, self.labels)
        
        # Split into train and test
        train_dataset, test_dataset, val_dataset = random_split(full_dataset, [train_size, test_size,val_size])
        
        # Store the appropriate dataset
        if split == 'train':
            self.dataset =train_dataset
        elif split == 'val':
            self.dataset = val_dataset
        else:
            self.dataset = test_dataset
        
    def __len__(self):
        """
        Return the total number of samples
        """
        return len(self.dataset)
    
    def __getitem__(self,idx):
        """
        Return a single sample and its label
        
        Parameters:
        -----------
        idx : int
            Index of the sample to return
            
        Returns:
        --------
        tuple
            (feature, label) at the given index
        """
        return self.dataset[idx]


# Define the neural network architecture
class SimpleNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, hidden_layers):  # Constructor initializes the network
        super(SimpleNN, self).__init__()  # Call the parent class (nn.Module) constructor
        layers = []  # Initialize an empty list to store the network layers

        # Input layer: Takes input features and maps them to the hidden layer size
        layers.append(nn.Linear(input_size, hidden_size))  # Add the first linear layer
        layers.append(nn.Tanh())  # Apply the activation function (Tanh)

        # Hidden layers: Create a series of hidden layers with activation functions
        for _ in range(hidden_layers):  # Loop for creating multiple hidden layers
            layers.append(nn.Linear(hidden_size, hidden_size))  # Add a hidden linear layer
            layers.append(nn.Tanh())  # Add an activation function (Tanh)

        # Output layer: Maps the final hidden layer outputs to the desired output size
        layers.append(nn.Linear(hidden_size, output_size))  # Add the final linear layer
        self.base = nn.Sequential(*layers)  # Create a sequential container with all layers
        self._init_weights()  # Initialize the network weights  

    def _init_weights(self):
        for layer in self.base:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(self, x,):  # Define the forward pass of the network
        input_features = torch.cat([x], dim=1)  # Concatenate input tensors along dimension 1
        m = self.base(input_features)  # Pass the concatenated input through the network
        return m  # Return the network output
    
    