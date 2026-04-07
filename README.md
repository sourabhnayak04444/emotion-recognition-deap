# DEAP Dataset Spectrogram Generator

This script converts raw EEG data from the **DEAP dataset** into 224x224 RGB spectrograms, pre-labeled and organized for Deep Learning training.

---

### 🛠 What This Code Does
* **Downloads Data:** Automatically fetches the DEAP dataset using `kagglehub`.
* **Processes Signals:** Extracts the last 30 seconds of 32-channel EEG data.
* **Generates Spectrograms:** Uses **STFT** (Short-Time Fourier Transform) to create frequency-based images.
* **Auto-Labels:** Sorts images into **High/Low Valence** and **High/Low Arousal** based on a threshold of **4.5**.
* **Organizes Folders:** Creates a clean directory structure compatible with most AI frameworks.

---

### 📂 Folder Structure
The output is saved in `./spectrograms_output/` as follows:
* **Subject ID** (e.g., s01)
    * **Valence** $\rightarrow$ HV (High) / LV (Low)
    * **Arousal** $\rightarrow$ HA (High) / LA (Low)

---

### 🚀 How to Use
1. **Install Requirements:**
   ```bash
   pip install numpy opencv-python scipy tqdm kagglehub

# DEAP Dataset Spectrogram Generator

This script converts raw EEG data from the **DEAP dataset** into 224x224 RGB spectrograms, pre-labeled and organized for Deep Learning training.

---

### 🛠 What This Code Does
* **Data Source:** Automatically fetches the DEAP dataset using `kagglehub`.
* **Signal Extraction:** Processes the last 30 seconds of 32-channel EEG data.
* **Transformation:** Uses **STFT** (Short-Time Fourier Transform) to create frequency-based images.
* **Auto-Labeling:** Binarizes **Valence** and **Arousal** using a 4.5 threshold.
* **Organization:** Creates a directory structure ready for `ImageFolder` or `ImageDataGenerator`.

---

### 📂 Folder Structure
```text
spectrograms_output/
└── [Subject_ID]/
    ├── Valence/
    │   ├── HV (High)
    │   └── LV (Low)
    └── Arousal/
        ├── HA (High)
        └── LA (Low)
