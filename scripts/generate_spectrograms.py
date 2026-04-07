import os
import glob
import pickle
import numpy as np
import cv2
import shutil
from scipy.signal import stft
from tqdm.auto import tqdm
import kagglehub
from IPython.display import FileLink, display

# ==========================================
# 1. DOWNLOAD & SETUP PATHS
# ==========================================
print(" Contacting Kaggle to download the DEAP dataset...")
# Downloads to your local cache. If already downloaded, it loads instantly.
DATASET_DIR = kagglehub.dataset_download("manh123df/deap-dataset")
print(f" Dataset located locally at: {DATASET_DIR}")

# Define where to build the image folders inside your current Jupyter directory
OUTPUT_DIR = "./spectrograms_output"
ZIP_TARGET = "final_spectrograms_dataset" # Do not add .zip here

# Find all .dat files
dat_files = sorted(glob.glob(os.path.join(DATASET_DIR, "**", "*.dat*"), recursive=True))

if not dat_files:
    raise FileNotFoundError(" Could not find any .dat files. Check the dataset structure.")
print(f" Found {len(dat_files)} subject data files.")

os.makedirs(OUTPUT_DIR, exist_ok=True)

def create_subject_folders(subject_id):
    dirs_to_make = [
        f"{OUTPUT_DIR}/{subject_id}/Valence/HV",
        f"{OUTPUT_DIR}/{subject_id}/Valence/LV",
        f"{OUTPUT_DIR}/{subject_id}/Arousal/HA",
        f"{OUTPUT_DIR}/{subject_id}/Arousal/LA"
    ]
    for d in dirs_to_make:
        os.makedirs(d, exist_ok=True)

# ==========================================
# 2. FAST STFT & IMAGE GENERATION
# ==========================================
def generate_and_save_spectrogram(eeg_signal, save_path):
    # Apply STFT based on the research paper parameters
    f, t, Zxx = stft(eeg_signal, fs=128, window='hann', nperseg=50, noverlap=25, nfft=512)
    mag = np.abs(Zxx)
    mag_db = 20 * np.log10(mag + 1e-6)
    
    # Normalize and apply color map using fast OpenCV operations
    mag_db_norm = cv2.normalize(mag_db, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    img_color = cv2.applyColorMap(mag_db_norm, cv2.COLORMAP_JET)
    img_color = cv2.flip(img_color, 0)
    
    # Resize to 224x224x3 for MobileNet/ResNet
    img_resized = cv2.resize(img_color, (224, 224))
    cv2.imwrite(save_path, img_resized)

# ==========================================
# 3. PROCESSING LOOP WITH COUNTERS
# ==========================================
print("\n Starting Spectrogram Generation...")

class_counts = []
global_hv, global_lv, global_ha, global_la = 0, 0, 0, 0

for file_path in tqdm(dat_files, desc="Processing Subjects"):
    filename = os.path.basename(file_path)
    subject_id = filename.split('.')[0]
    create_subject_folders(subject_id)
    
    # Load pickle file (Python 3 requires 'latin1' encoding for DEAP)
    with open(file_path, 'rb') as f:
        subject_data = pickle.load(f, encoding='latin1')
        
    data = subject_data['data']     
    labels = subject_data['labels'] 
    
    subj_hv, subj_lv, subj_ha, subj_la = 0, 0, 0, 0
    
    for trial in range(40):
        v_class = "HV" if labels[trial, 0] >= 4.5 else "LV"
        a_class = "HA" if labels[trial, 1] >= 4.5 else "LA"
        
        if v_class == "HV": subj_hv += 32
        else: subj_lv += 32
            
        if a_class == "HA": subj_ha += 32
        else: subj_la += 32
        
        # Process the 32 EEG channels (ignoring peripheral channels)
        for channel in range(32):
            # Second half of the signal starts at index 4032
            eeg_signal = data[trial, channel, 4032:]
            img_filename = f"trial_{trial+1:02d}_ch_{channel+1:02d}.png"
            
            v_save_path = f"{OUTPUT_DIR}/{subject_id}/Valence/{v_class}/{img_filename}"
            a_save_path = f"{OUTPUT_DIR}/{subject_id}/Arousal/{a_class}/{img_filename}"
            
            generate_and_save_spectrogram(eeg_signal, v_save_path)
            shutil.copy(v_save_path, a_save_path)
            
    class_counts.append({
        'subject': subject_id,
        'HV': subj_hv, 'LV': subj_lv,
        'HA': subj_ha, 'LA': subj_la
    })
    
    global_hv += subj_hv
    global_lv += subj_lv
    global_ha += subj_ha
    global_la += subj_la

# ==========================================
# 4. PRINT SUMMARY REPORT
# ==========================================
print("\n" + "="*60)
print(f"{'SUBJECT':<10} | {'HIGH VALENCE':<14} | {'LOW VALENCE':<14} | {'HIGH AROUSAL':<14} | {'LOW AROUSAL':<14}")
print("-" * 60)
for row in class_counts:
    print(f"{row['subject']:<10} | {row['HV']:<14} | {row['LV']:<14} | {row['HA']:<14} | {row['LA']:<14}")
print("-" * 60)
print(f"{'TOTALS':<10} | {global_hv:<14} | {global_lv:<14} | {global_ha:<14} | {global_la:<14}")
print("="*60 + "\n")
