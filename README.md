🛡️ VoiceShield — AI Voice Fraud Detection & Complaint System

Detecting AI-generated fake voices in real time, and turning detections into actionable, trackable complaints against impersonators.

Python TensorFlow Librosa

📌 What is VoiceShield?

With the rise of AI voice cloning tools (ElevenLabs, VALL-E, etc.), it has become easy to generate fake audio that sounds exactly like a real person — enabling scam calls that impersonate banks, officials, and trusted contacts.

VoiceShield is an end-to-end system that:

Detects whether a voice recording is genuine or AI-generated, using a CNN-LSTM deep learning model
Identifies which organization the caller claims to represent
Generates and routes a complaint to a verified official channel when a call crosses a strict fake-probability threshold
Tracks the complaint through to resolution

The detection engine is built on and extends VoxGuard by ramlasyaa — an open-source deepfake voice detector. VoiceShield fixes a class-imbalance issue found during training and adds the full complaint-escalation layer on top.

🧠 Detection Architecture
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

On top of detection, VoiceShield adds:

Fake Probability
        │
        ▼
  Escalation Threshold Check (0.97)
        │
        ▼
  Claimed Organization Identification
        │
        ▼
  Complaint Generation
        │
        ▼
  Routing → Verified Official Channel
        │
        ▼
  Complaint Tracking (status, updates)
📁 Project Structure
VoiceShield/
├── config.py            # All settings — hyperparameters, thresholds
├── extract_features.py  # Step 1: Extract MFCC features from audio files
├── model.py              # CNN-LSTM model definition
├── train.py               # Step 2: Train the detection model
├── evaluate.py           # Step 3: Print metrics + confusion matrix
├── predict.py             # Step 4: Check any audio file
├── requirements.txt
├── data/
│   ├── real/               # Genuine voice files (.wav / .flac)
│   ├── fake/               # AI-spoofed voice files
│   └── README.md         # Dataset download instructions
└── results/
    ├── training_curves.png
    └── confusion_matrix.png
🚀 Getting Started
1. Clone the repo
bash
git clone https://github.com/YOUR_USERNAME/VoiceShield.git
cd VoiceShield
2. Set up environment
bash
python3 -m venv venv
source venv/bin/activate        # macOS/Linux
# venv\Scripts\activate         # Windows

pip install -r requirements.txt
3. Add your dataset

See data/README.md for instructions. Recommended: ASVspoof 2019 LA partition.

data/
├── real/   ← genuine .wav files
└── fake/   ← spoofed .wav files
4. Extract features
bash
python extract_features.py
5. Train the model
bash
python train.py
6. Evaluate
bash
python evaluate.py
7. Predict any audio file
bash
python predict.py path/to/voice.wav
📊 Results

Evaluated on the held-out test split of ASVspoof 2019 LA:

Metric	Score
Overall Accuracy	97.24%
Real — Precision	0.86
Real — Recall	0.88
Real — F1-score	0.87
Fake — Precision	0.99
Fake — Recall	0.98
Fake — F1-score	0.98
ROC-AUC	0.9888

Engineering note: An earlier version of this model reported 89.83% "accuracy" — but a full per-class evaluation revealed it was simply predicting "Fake" for every sample, exploiting the dataset's natural 9:1 fake-to-real imbalance (ROC-AUC was only 0.57, near random). This was fixed using class-weighted training, which produced the genuine, balanced results above. This is documented as part of the project's development process.

🛠️ Tech Stack
Component	Tool / Library
Language	Python 3.9+
Deep Learning	TensorFlow / Keras
Audio Processing	Librosa
Features	MFCC (40 coefficients)
ML Metrics	Scikit-learn
Visualization	Matplotlib, Seaborn
Cloud Training	Google Colab
Dataset Source	Kaggle / ASVspoof 2019
🔑 Key Concepts
MFCC — Mel-Frequency Cepstral Coefficients. Compact audio features that capture how the human ear perceives sound.
CNN — Detects local spatial patterns in the MFCC "image".
LSTM — Captures how those patterns evolve over time (temporal context).
Class Weighting — Corrects for imbalanced training data so the model doesn't just learn to predict the majority class.
EarlyStopping — Prevents overfitting by stopping training when validation loss plateaus.
🗺️ Roadmap
 Detection model trained and validated on real data
 Class imbalance identified and fixed
 Live prediction verified on held-out samples
 Backend API wrapping detection + complaint logic
 Mobile app (Flutter) for recording/uploading calls
 Firebase-backed complaint tracking dashboard
🔗 References
ASVspoof 2019 Dataset
Original VoxGuard detector by ramlasyaa
Librosa Documentation
ASVspoof Challenge

VoiceShield extends the VoxGuard detection model with a complaint-escalation system for handling detected voice fraud.
