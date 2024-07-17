
import scipy
from mlxtend.preprocessing import standardize
from scipy import signal
from utilities import *

S1_E1 = scipy.io.loadmat('nina_pro_transradial_data/S1_E1_A1.mat')

original_label = S1_E1['stimulus']
df_original_label = pd.DataFrame(original_label)

refined_label = S1_E1['restimulus']
df_refined_label = pd.DataFrame(refined_label)

raw_EMG= S1_E1['emg']

df_raw_EMG = pd.DataFrame(raw_EMG)
column_names = ['EMG' + str(i) for i in range(12)]
df_raw_EMG.columns = column_names
print(df_raw_EMG.columns)
print(f'df_raw_EMG.shape_: {df_raw_EMG.shape}')

df_raw_EMG = standardize(df_raw_EMG,ddof=1)
has_nan = df_raw_EMG.isnull().values.any()
if has_nan:
    print("The DataFrame contains NaN values.")
else:
    print("The DataFrame does not contain any NaN values.")

for i in range(1,len(df_raw_EMG.columns)):
    plt.plot(df_raw_EMG.iloc[:,i] + 25*i, linewidth=0.1)
plt.title('raw EMG')
plt.show()

df_raw_EMG = df_raw_EMG.abs()
for i in range(1,len(df_raw_EMG.columns)):
    plt.plot(df_raw_EMG.iloc[:,i] + 25*i, linewidth=0.1)
plt.title('raw EMG - abs')
plt.show()

f_s = 2000
f_c = 1
N = 2
Wn = 2 * f_c/f_s
b,a = signal.butter(N=2, Wn=Wn, btype='low')
EMG_lp = signal.lfilter(b,a,df_raw_EMG.to_numpy(), axis=0)

for i in range(1, EMG_lp.shape[1]):
    plt.plot(EMG_lp[:,i] + 25*i, linewidth=0.5)

plt.title(f'EMG_LP - butterwoth LPF: fs = {f_s}Hz (samp. freq), N={N} (order), fc ={f_c}Hz (cutoff)')
plt.show()

df_EMG_LP = pd.DataFrame(EMG_lp)
df_EMG_LP.columns = ['EMG0','EMG1','EMG2','EMG3','EMG4','EMG5','EMG6','EMG7','EMG8',
                     'EMG9','EMG10','EMG11']

df_EMG_LP = standardize(df_EMG_LP,ddof=1)

df_normalized_EMG_only = df_EMG_LP.copy()
df_normalized_EMG_only.reset_index(drop=True, inplace=True)

print(f'df_normalized_EMG_only.shape: {df_normalized_EMG_only.shape}')

df_S1_E1_A1 = pd.concat([df_normalized_EMG_only, df_original_label], axis=1)
df_S1_E1_A1.rename(columns={df_S1_E1_A1.columns[-1]: 'original_label'}, inplace=True)
df_S1_E1_A1 = pd.concat([df_S1_E1_A1, df_refined_label], axis=1)
df_S1_E1_A1.rename(columns={df_S1_E1_A1.columns[-1]: 'refined_label'}, inplace=True)


df_S1_E1_A1.to_csv('nina_pro_transradial_data/S1_E1_A1_filtered_EMG_with_labels.csv', index=False)
print('file saved.')








