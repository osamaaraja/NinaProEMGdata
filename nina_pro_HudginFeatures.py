from utilities import *
import torch
import pandas as pd
import torch.optim as optim

df_S1_E1 = pd.read_csv('nina_pro_transradial_data/S1_E1_A1_filtered_EMG_with_labels.csv')
print(df_S1_E1.info())

df_EMG = df_S1_E1.iloc[:,0:12]
print(f'df_EMG columns: {df_EMG.columns}')
df_original_label = df_S1_E1.iloc[:,12]
print(f'df_original_label columns: {df_original_label.name}')
print(f'df_original_label unique values: {df_original_label.unique()}')
df_refined_label = df_S1_E1.iloc[:,13]
print(f'df_refined_label columns: {df_refined_label.name}')
print(f'df_refined_label unique values: {df_refined_label.unique()}')

channels = df_EMG.columns.tolist()

fs = 200
window_duration = 50e-3
window_size = int(window_duration * fs)

stride = window_size
thresh = 0.02

num_windows = (len(df_EMG) - window_size) // stride + 1
features = np.zeros((num_windows, 16, 5))

for i in range(num_windows):
    window_start = i * stride
    window_end = window_start + window_size

    if window_end > len(df_EMG):
        break

    window_data = df_EMG.loc[window_start:window_end, channels]

    for j, channel in enumerate(channels):
        mav = mean_absolute_value(window_data[channel])
        mas = mean_absolute_slope(window_data[channel].values[:-1], window_data[channel].values[1:])
        zc = zero_crossings(window_data[channel], thresh)
        ssc = slope_sign_change(window_data[channel], thresh)
        wl = waveform_length(window_data[channel])

        features[i, j, :] = [mav, mas, zc, ssc, wl]

print(features.shape)

num_windows = features.shape[0]
num_channels = features.shape[1]
num_features = features.shape[2]
feature_vector = features.reshape(num_windows, num_channels * num_features)
print(f'feature_vector.shape: {feature_vector.shape}')

input_dim = num_channels * num_features
hidden_dim = 16
latent_dim = 8
learning_rate = 0.001

autoencoder = Autoencoder(input_dim, hidden_dim, latent_dim)
criterion = nn.MSELoss()
optimizer = optim.Adam(autoencoder.parameters(), lr=learning_rate)
feature_vector = torch.tensor(feature_vector, dtype=torch.float)

num_epochs = 10
batch_size = 32

for epoch in range(num_epochs):
    for i in range(0, len(feature_vector), batch_size):

        batch = feature_vector[i:i + batch_size]
        outputs = autoencoder(batch)
        loss = criterion(outputs, batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch: {epoch + 1}, Loss: {loss.item()}")

encoded_features = autoencoder.encoder(feature_vector)
encoded_array = encoded_features.detach().numpy()

PCA_result = apply_PCA(encoded_array, window_size)
apply_Kmeans(PCA_result, 18, window_size)


