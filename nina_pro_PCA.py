
from utilities import *
import pandas as pd

df_S1_E1 = pd.read_csv('nina_pro_transradial_data/S1_E1_A1_filtered_EMG_with_labels.csv')
print(df_S1_E1.info())

df_EMG = df_S1_E1.iloc[:,0:12]
print(f'df_EMG columns: {df_EMG.columns}')
df_original_label = df_S1_E1.iloc[:,12]
print(f'df_original_label columns: {df_original_label.name}')
df_refined_label = df_S1_E1.iloc[:,13]
print(f'df_refined_label columns: {df_refined_label.name}')

cov_mat = np.cov(df_EMG, rowvar=False, ddof=1)
eig_vals, eig_vecs = np.linalg.eig(cov_mat)
idx = eig_vals.argsort()[::-1]
eig_vals_sort = eig_vals[idx]
eig_vecs_sort = eig_vecs[:,idx]
T = eig_vecs_sort[:,:2]
print('sorted eigen vectors:')
print(T)
explained_var = np.sum(eig_vals[:2]/np.sum(eig_vals))
print(f'explained variance based on eigen values: {explained_var*100:.2f}%')

pca = PCA(n_components=2)
principal_components = pca.fit_transform(df_EMG)
df_principal = pd.DataFrame(data=principal_components,columns=['principal component 1', 'principal component 2'])

plt.figure(figsize=(10, 6))
plt.scatter(df_principal['principal component 1'], df_principal['principal component 2'])
plt.title('PCA:- n_components = 2')
plt.show()

variance = pca.explained_variance_ratio_.sum()
print(f'explained variance with built-in method: {variance*100:.2f}%')

