import wfdb
import numpy as np
from scipy.signal import savgol_filter, find_peaks
import pywt
import matplotlib.pyplot as plt
import random
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import RandomOverSampler
from imblearn.over_sampling import SMOTE
from collections import Counter
from sklearn.model_selection import train_test_split

#Helper function to convert annotation symbol, if you want to visualize the data from a section of ecg signal
def converted_array(ann_array, val):
    converted = [ann_array[0] - val]
    for j in range(len(ann_array)-1):
        converted.append(ann_array[j+1] - ann_array[j] + converted[j])

    return converted



record = wfdb.rdrecord('109')

ann = wfdb.rdann('109', 'atr')
#Reading signals
signal = record.p_signal

filter_signal = savgol_filter(signal, window_length=20, polyorder=2, axis=0, mode='mirror')



# Access the annotations
annotation_sample = ann.sample # Annotation sample locations (positions of annotations)
annotation_symbol = ann.symbol

#Remove first entry, as it is redundant
annotation_sample = annotation_sample[1:]
annotation_symbol = annotation_symbol[1:]




#Load previously saved data (if any) to add it into the data which is extracted from new file
loaded_data = np.load('model_features.npz')

# Load ECG signal (Lead II) from MIT-BIH Arrhythmia dataset

ecg_signal = filter_signal[:, 0] # Assuming Lead II is the second column


# r_peaks = annotation_sample[(annotation_sample>=0) & (annotation_sample<=3000)]
r_peaks = annotation_sample



# Step 4: Detect Q and S points
q_points = []
s_points = []

for r in r_peaks:
    # Search for Q point to the left of the R peak
  
    q_region = ecg_signal[max(0, r - 40):r]

    q_index = np.argmin(q_region)
    q_points.append(r - 40 + q_index)
    
    # Search for S point to the right of the R peak
    s_region = ecg_signal[r:min(len(ecg_signal), r + 80)]
   
    s_index = np.argmin(s_region)
    s_points.append(r + s_index)

# Step 5: Detect P and T points
p_points = []
t_points = []


for s in s_points:
    # Search for T wave in a region after S
    t_region = ecg_signal[s:min(len(ecg_signal), s + 140)]
    t_index = np.argmax(t_region)
    
    t_points.append(s + t_index)

for q in q_points:
    # Search for P wave in a region before Q
    p_region = ecg_signal[max(0, q - 70):q]
    p_index = np.argmax(p_region)

    if(q-70 + p_index) in t_points:
        p_points.append(0)
    else:
        p_points.append(q - 70 + p_index)


#PLot the peaks
plt.figure(figsize=(10, 6))
plt.plot(ecg_signal, label='ECG Signal')
plt.scatter(r_peaks, ecg_signal[r_peaks], color='red', label='R Peaks')
plt.scatter(q_points, ecg_signal[q_points], color='blue', label='Q Points')
plt.scatter(s_points, ecg_signal[s_points], color='green', label='S Points')
plt.scatter(p_points, ecg_signal[p_points], color='purple', label='P Points')
plt.scatter(t_points, ecg_signal[t_points], color='orange', label='T Points')
plt.legend()
plt.title("ECG Signal with Detected P, Q, R, S, T Points")
plt.xlabel("Samples")
plt.ylabel("Normalized Amplitude")
plt.show()


#Extract features from signal
r_r_distance = []
for ind in range(len(r_peaks)-1):
    r_r_distance.append(r_peaks[ind+1] - r_peaks[ind])

q_s_distance = np.array(np.array(s_points[1:]) - np.array(q_points[1:]))

r_r_distance = np.array(r_r_distance)
p_points = np.array(ecg_signal[p_points[1:]])
q_points = np.array(ecg_signal[q_points[1:]])
r_peaks = np.array(ecg_signal[r_peaks[1:]])
s_points = np.array(ecg_signal[s_points[1:]])
t_points = np.array(ecg_signal[t_points[1:]])
annotation_symbol = annotation_symbol[1:]


#Load previously saved points

p_points_loaded = loaded_data['p_points']
q_points_loaded = loaded_data['q_points']
r_peaks_loaded = loaded_data['r_peaks']
s_points_loaded = loaded_data['s_points']
t_points_loaded = loaded_data['t_points']
r_r_distance_loaded = loaded_data['r_r_distance']
q_s_distance_loaded = loaded_data['q_s_distance']
annotation_symbol_loaded = loaded_data['annotation_symbol']


#Add it into previously daved points
p_points = np.concatenate((p_points_loaded, p_points))
q_points = np.concatenate((q_points_loaded, q_points))
r_peaks = np.concatenate((r_peaks_loaded, r_peaks))
s_points = np.concatenate((s_points_loaded, s_points))
t_points = np.concatenate((t_points_loaded, t_points))
r_r_distance = np.concatenate((r_r_distance_loaded, r_r_distance ))
q_s_distance = np.concatenate((q_s_distance_loaded, q_s_distance))
annotation_symbol = np.concatenate((annotation_symbol_loaded, annotation_symbol))


#Save the points in a file to load it into later for training model
np.savez('model_features.npz', p_points=p_points_loaded, r_peaks=r_peaks_loaded, q_points=q_points_loaded, s_points=s_points_loaded
,t_points=t_points_loaded, r_r_distance=r_r_distance_loaded, q_s_distance=q_s_distance_loaded, annotation_symbol=annotation_symbol_loaded)

