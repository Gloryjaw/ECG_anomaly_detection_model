# Real-Time ECG Arrhythmia Detection System

This project is an edge-computing healthcare application designed to monitor and detect cardiac anomalies in real-time. By combining dedicated hardware sensors with a Convolutional Neural Network (CNN), the system captures live electrocardiogram (ECG) readings and performs instant classification to identify arrhythmic heartbeat patterns.

## How It Works

The system utilizes a continuous data pipeline, transitioning from raw analog signals to advanced machine learning inference:

1. **Signal Acquisition:** A physical ECG sensor module captures the raw electrical activity of the heart via electrodes attached to the subject.
2. **Signal Digitization:** An Arduino microcontroller reads the analog voltage from the sensor, digitizing the data and ensuring a consistent sampling rate.
3. **Serial Transmission:** The Arduino transmits the digitized ECG time-series data over a serial connection to the central processing hub.
4. **Edge Processing:** A Raspberry Pi receives and parses the incoming serial data stream, buffering it into appropriate time windows.
5. **Real-Time Inference:** A pre-trained Convolutional Neural Network (CNN) deployed on the Raspberry Pi analyzes the data windows, classifying the waveforms and flagging arrhythmic anomalies as they occur.

## Dataset & Model Training

The CNN powering this system was trained and validated using the **MIT-BIH Arrhythmia Database**, a standard reference repository of recorded electrocardiograms. 

The model is designed to recognize complex morphological features in the ECG waveforms, allowing it to distinguish between normal sinus rhythms and various classes of arrhythmias based on the MIT-BIH annotations.

## Technology Stack

* **Machine Learning:** Python, TensorFlow/Keras, NumPy, Pandas
* **Data Integration:** PySerial (for hardware-to-software communication)
* **Embedded Software:** C++ / Arduino IDE for data sampling and transmission
* **Hardware Platforms:** * Raspberry Pi (Inference & Edge Computing)
  * Arduino Microcontroller (Signal Digitization)

## System Architecture

The architecture is explicitly decoupled to optimize performance:
* **The Sensor Node (Arduino):** Acts as a dedicated Analog-to-Digital Converter (ADC). By offloading the raw data collection to a real-time microcontroller, the system avoids the timing inconsistencies that can occur if an operating system tries to sample analog pins directly.
* **The Inference Node (Raspberry Pi):** Acts as the edge server. It possesses the necessary computational power to run the CNN, handle the Python environment, and potentially host a dashboard or alert system without interrupting the continuous hardware readings.

## Hardware Requirements
* Raspberry Pi (Model 3B+ or newer recommended for ML inference)
* Arduino Microcontroller (e.g., Uno, Nano)
* Analog ECG Sensor Module (e.g., AD8232)
* Biomedical sensor pads/electrodes
* USB cable (for Arduino to Pi serial connection and power)
