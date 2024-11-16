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






loaded_data = np.load('model_features.npz')



p_points_loaded = loaded_data['p_points']
q_points_loaded = loaded_data['q_points']
r_peaks_loaded = loaded_data['r_peaks']
s_points_loaded = loaded_data['s_points']
t_points_loaded = loaded_data['t_points']
r_r_distance_loaded = loaded_data['r_r_distance']
q_s_distance_loaded = loaded_data['q_s_distance']
annotation_symbol_loaded = loaded_data['annotation_symbol']




features = np.column_stack(( p_points_loaded, q_points_loaded, r_peaks_loaded, s_points_loaded, t_points_loaded,r_r_distance_loaded, q_s_distance_loaded))


################################# CNN Model ######################################


import tensorflow as tf
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import classification_report




# Encode the output labels
encoder = OneHotEncoder(sparse_output=False)
Y_encoded = encoder.fit_transform(annotation_symbol_loaded.reshape(-1, 1))

# Normalize input features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(features)

# Split the dataset into training and testing
X_train, X_test, Y_train, Y_test = train_test_split(X_scaled, Y_encoded, test_size=0.2, random_state=42)

smote = SMOTE(sampling_strategy='auto', random_state=42)

# Apply SMOTE to the training data
X_train, Y_train = smote.fit_resample(X_train, Y_train)


# Reshape the data for CNN (if required, for example adding a channel dimension)
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

# Define the CNN model
model = tf.keras.Sequential([
    tf.keras.layers.Conv1D(filters=32, kernel_size=2, activation='relu', input_shape=(7, 1)),
    tf.keras.layers.MaxPooling1D(pool_size=2),
    tf.keras.layers.Conv1D(filters=64, kernel_size=2, activation='relu'),
    tf.keras.layers.MaxPooling1D(pool_size=2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(4, activation='softmax')  # 4 output categories
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(X_train, Y_train, epochs=100, batch_size=32, validation_split=0.2, verbose=1)

# Evaluate the model on the test data
test_loss, test_accuracy = model.evaluate(X_test, Y_test, verbose=0)
print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f}")

# Predictions and classification report
Y_pred = model.predict(X_test)
Y_pred_classes = np.argmax(Y_pred, axis=1)
Y_test_classes = np.argmax(Y_test, axis=1)
print(classification_report(Y_test_classes, Y_pred_classes, target_names=encoder.categories_[0]))

#Save the model
model.save('ecg_model.keras')

