import numpy as np
import torch.nn as nn
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

def fill_NaN_with_zero(df, EMG, col):

    indexes_with_zero = df.loc[df[col] == 0].index

    if len(indexes_with_zero) == 0:
        print(f'no indexes with the value of 0 found in {EMG}')

    else:
        print(indexes_with_zero)

    df[col] = df[col].fillna('0')
    print(df.info)
    print(f'NaN filled with zeros in {EMG} dataset')
    return df

def first_nonzero_index(df, EMG,col):

    for index, value in df[col].items():
        if value != 0:
            first_non_zero_index = index
            break
    print(f'first_non_zero_index for EMG data in {EMG} df is {first_non_zero_index}')
    return first_non_zero_index

def drop_rows_till_first_nonzero(df, value):

    df = df.drop(df.index[:value])
    print(f'after row drop, df.shape: {df.shape}')
    return df

def fetch_labels(df, EMG, col):

    if not df[col].isnull().all():
        x = df[col].dropna().values
        print(f'list of labels in {EMG}: {x}')
        print(x.shape)

    unique_strings = df[col].dropna().unique()
    for string in unique_strings:
        print(string)

    nan_count = df[col].isnull().sum()
    print(f'Number of NaN values in {EMG} col {col}: {nan_count}')

def check_column_similarity(df, col1, col2):

    same_columns = df[col1].equals(df[col2])
    if same_columns:
        print(f"The columns {col1} and {col2} are the same.")
        return True
    else:
        #print(f"The columns {col1} and {col2} are different. But have {num_matches / len} same rows")
        print(f"The columns {col1} and {col2} are different")

        return False

def mean_absolute_value(v):

    abs_vec = np.abs(np.array(v))
    mav = abs_vec.mean()
    return mav

def mean_absolute_slope(v_0, v_plus1):

    v_0 = np.array(v_0)
    v_plus1 = np.array(v_plus1)
    mas = mean_absolute_value(v_plus1) - mean_absolute_value(v_0)
    return mas

def zero_crossings(v,thresh):

    v = np.array(v)
    count = 0
    for i in range(len(v) - 1):
        if((v[i]<0 and v[i+1]>0) or (v[i]>0 and v[i+1]<0)):
            if(abs(v[i]-v[i+1])>thresh):
                count += 1
    return count

def slope_sign_change(v, thresh):

    v = np.array(v)
    count = 0
    for i in range(len(v)-2):
        for i in range(len(v) - 2):
            if ((v[i + 1] > v[i] and v[i + 1] > v[i + 2]) or (v[i + 1] < v[i] and v[i + 1] < v[i + 2])):
                if (abs(v[i + 1] - v[i + 2]) >= thresh or abs(v[i + 1] - v[i]) >= thresh):
                    count += 1
    return count

def waveform_length(v):

    v = np.array(v)
    abs_diffs = np.abs(np.diff(v, axis=0))
    l_0 = np.sum(abs_diffs)
    return l_0

def calculate_hudgins_features(window):
    features = []
    for i in range(window.shape[1]):
        channel_features = []
        for j in range(window.shape[0] - 1):
            channel_features.append(mean_absolute_value(window[j]))
            channel_features.append(mean_absolute_slope(window[j], window[j + 1]))
            channel_features.append(zero_crossings(window[j], 0))
            channel_features.append(slope_sign_change(window[j], 0))
            channel_features.append(waveform_length(window[j]))
        features.append(channel_features)
    return features


class Autoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(Autoencoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
            nn.ReLU()
        )

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

def apply_tSNE(encoded_array, window_size=None):
    tsne = TSNE(n_components=2, random_state=42)
    tsne_result = tsne.fit_transform(encoded_array)

    tsne_x = tsne_result[:, 0]
    tsne_y = tsne_result[:, 1]

    plt.scatter(tsne_x, tsne_y)
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    if window_size == None:
        plt.title('t-SNE Visualization of Encoded Features')
    else:
        plt.title(f't-SNE Visualization of Encoded Features (unlabelled), window_length = {window_size} samples')
    plt.show()

    return tsne_result

def apply_Kmeans(result,k, window_size=None):
    num_clusters = k
    kmeans = KMeans(n_clusters=k, random_state=42)
    cluster_labels = kmeans.fit_predict(result)
    res_x = result[:, 0]
    res_y = result[:, 1]

    plt.scatter(res_x, res_y, c=cluster_labels)
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    if window_size == None:
        plt.title(f'K-means n_cluster = {num_clusters}')
    else:
        plt.title(f'K-means n_cluster = {num_clusters}, window_length = {window_size} samples')
    plt.show()

def apply_PCA(result, window_size=None):
    pca = PCA(n_components=2, random_state=42)
    pca_result = pca.fit_transform(result)
    pca_x = pca_result[:, 0]
    pca_y = pca_result[:, 1]

    plt.scatter(pca_x, pca_y)
    plt.xlabel('PCA Dimension 1')
    plt.ylabel('PCA Dimension 2')
    if window_size == None:
        plt.title('PCA Visualization of Encoded Features')
    else:
        plt.title(f'PCA Visualization of Encoded Features (unlabelled), window_length = {window_size} samples')
    plt.show()

    return pca_result