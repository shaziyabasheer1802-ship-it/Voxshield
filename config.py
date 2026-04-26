# ============================================================
# VoxGuard — Configuration
# All settings are here. Change these to experiment!
# ============================================================

# --- Audio Settings ---
SAMPLE_RATE = 16000        # Standard sample rate for speech (16kHz)
DURATION    = 4            # Clip all audio to 4 seconds
N_MFCC      = 40           # Number of MFCC features to extract
HOP_LENGTH  = 512          # Frames between each MFCC window
N_FFT       = 1024         # FFT window size

# --- Dataset Paths ---
# Download ASVspoof 2019 LA dataset and point these to your folders
REAL_AUDIO_DIR  = "data/real"    # Folder with genuine/real voice files
FAKE_AUDIO_DIR  = "data/fake"    # Folder with spoofed/fake voice files

# Processed features will be saved here
FEATURES_PATH   = "data/features.npz"

# --- Model Settings ---
CNN_FILTERS     = [32, 64]       # Number of filters in each CNN layer
LSTM_UNITS      = 64             # Units in the LSTM layer
DROPOUT_RATE    = 0.3            # Dropout to prevent overfitting

# --- Training Settings ---
EPOCHS          = 30
BATCH_SIZE      = 32
LEARNING_RATE   = 0.001
VALIDATION_SPLIT = 0.2           # 20% of data used for validation

# --- Save Paths ---
MODEL_PATH      = "voxguard_model.h5"
PLOT_PATH       = "results/training_curves.png"
CM_PATH         = "results/confusion_matrix.png"

# --- Labels ---
LABEL_REAL = 0    # Genuine/real voice
LABEL_FAKE = 1    # Fake/spoofed voice
