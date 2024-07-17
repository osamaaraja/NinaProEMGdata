from utilities import *
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import random
import seaborn as sns
import warnings
from numba import NumbaDeprecationWarning
warnings.simplefilter("ignore", category=NumbaDeprecationWarning)
warnings.simplefilter("ignore", category=UserWarning)
import umap.umap_ as umap

class Autoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(Autoencoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, latent_dim),
            nn.LeakyReLU()
        )

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded, encoded

df_data = pd.read_csv('nina_pro_transradial_data/S1_E1_A1_filtered_EMG_with_labels.csv', nrows=500000)
df_data = df_data.drop('original_label', axis=1) # checking for only refined labels
df_EMG = df_data.iloc[:,0:13]
print(df_EMG.columns)
# Separate EMG columns and label column
emg_columns = df_EMG.columns[:-1]
label_column = df_EMG.columns[-1]

# Randomly shuffle the EMG column indices
shuffled_indices = np.random.permutation(len(emg_columns))

# Rearrange the EMG columns based on the shuffled indices
shuffled_emg_columns = emg_columns[shuffled_indices]

# Reconstruct the shuffled DataFrame
shuffled_df_EMG = df_EMG[shuffled_emg_columns]
shuffled_df_EMG[label_column] = df_EMG[label_column]

print(shuffled_df_EMG.columns)
df_label = shuffled_df_EMG['refined_label']

channels = shuffled_df_EMG.columns[:-1].tolist()

print(channels)


input_dim = len(channels)
hidden_dim = 16
latent_dim = 8
learning_rate = 0.001
num_epochs = 20
batch_size = 32

autoencoder = Autoencoder(input_dim, hidden_dim, latent_dim)
criterion = nn.MSELoss()
optimizer = optim.Adam(autoencoder.parameters(), lr=learning_rate)

all_features = []
all_labels = []

# Training loop
for epoch in range(num_epochs):
    running_loss = 0.0

    # Shuffle the data
    #df_EMG = df_EMG.sample(frac=1).reset_index(drop=True)

    for i in range(0, len(shuffled_df_EMG), batch_size):
        batch_data = shuffled_df_EMG.loc[i:i + batch_size - 1, channels].values

        # Normalize the input data
        batch_data = (batch_data - np.mean(batch_data)) / np.std(batch_data)

        batch_tensor = torch.tensor(batch_data, dtype=torch.float)

        # Forward pass
        outputs, encoded = autoencoder(batch_tensor)

        # Reconstruction loss
        loss = criterion(outputs, batch_tensor)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        # Store the encoded features and labels
        all_features.append(encoded.detach().numpy())
        all_labels.extend(df_EMG.loc[i:i + batch_size - 1, 'refined_label'])


    epoch_loss = running_loss / (len(df_EMG) // batch_size)
    print(f"Epoch: {epoch + 1}, Loss: {epoch_loss:.6f}")

all_features = np.concatenate(all_features) # all_features contains the latent information
all_labels = np.array(all_labels)

flag = 0

if flag == 0:

    pca = PCA(n_components=2, random_state=42)
    pca_features = pca.fit_transform(all_features)

    unique_labels = np.unique(all_labels)

    num_labels = 18
    '''
    colors = []
    for _ in range(num_labels):
        color = "#" + ''.join([random.choice('0123456789ABCDEF') for _ in range(6)])
        colors.append(color)
    '''
    colors = sns.color_palette('bright', num_labels)

    plt.figure(figsize=(10, 8))
    for label, color in zip(unique_labels, colors):
        label_indices = np.where(all_labels == label)[0]
        plt.scatter(pca_features[label_indices, 0], pca_features[label_indices, 1], c=color, label=label)

    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.legend()
    plt.title('PCA Visualization')
    plt.show()

else:
    # Perform UMAP dimensionality reduction
    umap_reducer = umap.UMAP(n_components=2, random_state=42)
    umap_features = umap_reducer.fit_transform(all_features)

    unique_labels = np.unique(all_labels)
    num_labels = len(unique_labels)

    # Generate colors using seaborn color palette
    colors = sns.color_palette('bright', num_labels)

    plt.figure(figsize=(10, 8))
    for label, color in zip(unique_labels, colors):
        label_indices = np.where(all_labels == label)[0]
        plt.scatter(umap_features[label_indices, 0], umap_features[label_indices, 1], color=color, label=label)

    plt.xlabel('UMAP Dimension 1')
    plt.ylabel('UMAP Dimension 2')
    plt.legend()
    plt.title('UMAP Visualization')
    plt.show()

