"""
extract_features.py
-------------------
Reads audio files from data/real and data/fake folders,
extracts MFCC features using librosa, and saves them to a .npz file.

Run this FIRST before training:
    python extract_features.py
"""

import os
import numpy as np
import librosa
from tqdm import tqdm   # shows a nice progress bar
import config


def extract_mfcc(file_path):
    """
    Load an audio file and extract its MFCC features.
    
    MFCCs (Mel-Frequency Cepstral Coefficients) capture the "shape"
    of the sound spectrum — they're widely used in voice/speech tasks.

    Returns:
        numpy array of shape (time_steps, N_MFCC)
        or None if the file can't be loaded.
    """
    try:
        # Load audio, force it to mono, clip/pad to fixed DURATION
        audio, sr = librosa.load(
            file_path,
            sr=config.SAMPLE_RATE,
            duration=config.DURATION,
            mono=True
        )

        # Pad with zeros if audio is shorter than DURATION
        target_length = config.SAMPLE_RATE * config.DURATION
        if len(audio) < target_length:
            audio = np.pad(audio, (0, target_length - len(audio)))

        # Extract MFCCs → shape: (N_MFCC, time_frames)
        mfcc = librosa.feature.mfcc(
            y=audio,
            sr=sr,
            n_mfcc=config.N_MFCC,
            hop_length=config.HOP_LENGTH,
            n_fft=config.N_FFT
        )

        # Transpose → shape: (time_frames, N_MFCC)
        # This is the format the model expects: (timesteps, features)
        mfcc = mfcc.T

        return mfcc

    except Exception as e:
        print(f"  [Warning] Could not process {file_path}: {e}")
        return None


def load_all_files(folder, label):
    """
    Go through all .wav/.flac files in a folder,
    extract features, and assign the given label (0=real, 1=fake).
    """
    features = []
    labels   = []

    # Collect all supported audio files
    audio_files = [
        f for f in os.listdir(folder)
        if f.endswith((".wav", ".flac", ".mp3"))
    ]

    print(f"\nProcessing {len(audio_files)} files from: {folder}")

    for filename in tqdm(audio_files):
        file_path = os.path.join(folder, filename)
        mfcc = extract_mfcc(file_path)

        if mfcc is not None:
            features.append(mfcc)
            labels.append(label)

    return features, labels


def main():
    print("=" * 50)
    print("  VoxGuard — Feature Extraction")
    print("=" * 50)

    # ---- Load real voices ----
    real_features, real_labels = load_all_files(
        config.REAL_AUDIO_DIR,
        label=config.LABEL_REAL
    )

    # ---- Load fake/spoofed voices ----
    fake_features, fake_labels = load_all_files(
        config.FAKE_AUDIO_DIR,
        label=config.LABEL_FAKE
    )

    # ---- Combine real + fake ----
    all_features = real_features + fake_features
    all_labels   = real_labels   + fake_labels

    if len(all_features) == 0:
        print("\n[ERROR] No audio files found! Check your data/ folder.")
        return

    # ---- Normalize feature lengths (in case clips differ slightly) ----
    # Find the minimum time_steps across all samples
    min_len = min(f.shape[0] for f in all_features)
    all_features = [f[:min_len] for f in all_features]

    X = np.array(all_features)   # shape: (num_samples, time_steps, N_MFCC)
    y = np.array(all_labels)     # shape: (num_samples,)

    print(f"\n✅ Feature extraction complete!")
    print(f"   Total samples : {len(X)}")
    print(f"   Real samples  : {real_labels.count(config.LABEL_REAL)}")
    print(f"   Fake samples  : {fake_labels.count(config.LABEL_FAKE)}")
    print(f"   Feature shape : {X.shape}  (samples, time_steps, mfcc_coeffs)")

    # ---- Save to disk ----
    os.makedirs(os.path.dirname(config.FEATURES_PATH), exist_ok=True)
    np.savez(config.FEATURES_PATH, X=X, y=y)
    print(f"\n💾 Saved features to: {config.FEATURES_PATH}")


if __name__ == "__main__":
    main()
