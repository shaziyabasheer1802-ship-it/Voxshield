# 🎙️ VoxGuard — Audio Deepfake Detection

> Detecting AI-generated fake voices using CNN-LSTM and spectrogram analysis.

[![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.12+-FF6F00?style=flat-square&logo=tensorflow&logoColor=white)](https://tensorflow.org)
[![Librosa](https://img.shields.io/badge/Librosa-Audio_Processing-green?style=flat-square)](https://librosa.org)

---

## 📌 What is VoxGuard?

With the rise of AI voice cloning tools (ElevenLabs, VALL-E, etc.), it has become easy to generate fake audio that sounds exactly like a real person. VoxGuard is a deep learning system that **detects whether a voice recording is genuine or AI-generated**.

It works by:
1. **Extracting MFCC features** from the audio (a compact representation of the sound spectrum)
2. **Passing them through a CNN** to detect local patterns in the sound
3. **Passing through an LSTM** to analyse how those patterns change over time
4. **Outputting a probability** — how likely the voice is fake

---

## 🧠 Architecture

```
Audio File (.wav / .flac)
        │
        ▼
  MFCC Extraction (librosa)
  → shape: (time_steps × 40 coefficients)
        │
        ▼
  ┌─────────────────┐
  │   CNN Block 1   │  Conv2D(32) → BN → MaxPool → Dropout
  │   CNN Block 2   │  Conv2D(64) → BN → MaxPool → Dropout
  └─────────────────┘
        │
        ▼
  LSTM Layer (64 units)
  → Reads temporal patterns across the MFCC sequence
        │
        ▼
  Dense(32) → Dense(1, sigmoid)
        │
        ▼
  Output: probability of being FAKE
  > 0.5 → FAKE  |  < 0.5 → REAL
```

---

## 📁 Project Structure

```
VoxGuard/
├── config.py            # All settings — change hyperparameters here
├── extract_features.py  # Step 1: Extract MFCC features from audio files
├── model.py             # CNN-LSTM model definition
├── train.py             # Step 2: Train the model
├── evaluate.py          # Step 3: Print metrics + confusion matrix
├── predict.py           # Step 4: Check any audio file
├── requirements.txt
├── data/
│   ├── real/            # Put genuine voice files here (.wav / .flac)
│   ├── fake/            # Put AI-spoofed voice files here
│   └── README.md        # Dataset download instructions
└── results/
    ├── training_curves.png
    └── confusion_matrix.png
```

---

## 🚀 Getting Started

### 1. Clone the repo

```bash
git clone https://github.com/ramlasyaa/VoxGuard.git
cd VoxGuard
```

### 2. Set up environment

```bash
python3 -m venv venv
source venv/bin/activate        # macOS/Linux
# venv\Scripts\activate         # Windows

pip install -r requirements.txt
```

### 3. Add your dataset

See [`data/README.md`](data/README.md) for instructions.  
Recommended: **ASVspoof 2019 LA** partition.

```
data/
├── real/   ← copy genuine .wav files here
└── fake/   ← copy spoofed .wav files here
```

### 4. Extract features

```bash
python extract_features.py
```

### 5. Train the model

```bash
python train.py
```

### 6. Evaluate

```bash
python evaluate.py
```

### 7. Predict any audio file

```bash
python predict.py path/to/voice.wav
# or an entire folder
python predict.py path/to/audio_folder/
```

---

## 📊 Results

Results on the **ASVspoof 2019 LA** evaluation set:

| Metric    | Score  |
|-----------|--------|
| Accuracy  | ~91%   |
| Precision | ~89%   |
| Recall    | ~93%   |
| F1-Score  | ~91%   |
| ROC-AUC   | ~0.96  |

> *Results may vary depending on dataset size and split.*

---

## 🛠️ Tech Stack

| Component        | Tool / Library         |
|------------------|------------------------|
| Language         | Python 3.9+            |
| Deep Learning    | TensorFlow / Keras     |
| Audio Processing | Librosa                |
| Features         | MFCC (40 coefficients) |
| ML Metrics       | Scikit-learn           |
| Visualization    | Matplotlib, Seaborn    |

---

## 🔑 Key Concepts

- **MFCC** — Mel-Frequency Cepstral Coefficients. Compact audio features that capture how the human ear perceives sound.
- **CNN** — Detects local spatial patterns in the MFCC "image".
- **LSTM** — Captures how those patterns evolve over time (temporal context).
- **Binary Cross-Entropy** — Loss function for real/fake binary classification.
- **EarlyStopping** — Prevents overfitting by stopping training when validation loss plateaus.

---

## 📄 Related Research

This project is based on our paper:

> *"VoxGuard: Fake Audio (Deepfake Voice) Detection Using Spectrogram Analysis"*  
> Ram Lasya et al. — **CONIT 2026 (IEEE)**

---

## 🔗 References

- [ASVspoof 2019 Dataset](https://datashare.ed.ac.uk/handle/10283/3336)
- [WaveFake Dataset](https://github.com/RUB-SysSec/WaveFake)
- [Librosa Documentation](https://librosa.org/doc/)
- [ASVspoof Challenge](https://www.asvspoof.org/)

---

*Built by [Ram Lasya](https://www.linkedin.com/in/ram-lasya-405035336) · Amrita Vishwa Vidyapeetham*
