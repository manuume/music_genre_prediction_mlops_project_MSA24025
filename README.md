# 🎵 Music Genre Prediction MLOps Project

An end-to-end machine learning pipeline for music genre classification using CNN and MFCC features.

## 📋 Project Overview

This project implements a complete MLOps pipeline for music genre classification using Convolutional Neural Networks. The system extracts Mel-Frequency Cepstral Coefficients (MFCCs) from audio tracks and trains a CNN model to classify music into different genres.

## 📁 Project Structure

```
music_genre_prediction_mlops_project_MSA24025/
│
├── data/
│   ├── Data/genres_original/ # Raw audio files
│   ├── data.json             # Extracted MFCC features
│   └── preprocessed_data.npz # Preprocessed train/test data
│
├── models/                   # Trained CNN model (.h5)
│
├── src/                      # All source code files
│   ├── data_loader.py
│   ├── preprocess.py
│   ├── model.py
│   ├── train.py
│   └── model_evaluation.py
│
├── config.yaml               # Central configuration file
├── requirements.txt          # Python dependencies
└── README.md                 # Project documentation
```

## 🔧 Setup Instructions

1. **Clone the repository**
   ```bash
   git clone <your_repo_url>
   cd music_genre_prediction_mlops_project_MSA24025
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Add dataset**
   Download the GTZAN dataset and place it under:
   ```bash
   data/Data/genres_original/
   ```

## 🚀 Run the Pipeline

All parameters are controlled through `config.yaml`.

1️⃣ **Extract MFCCs**
   ```bash
   python src/data_loader.py
   ```

2️⃣ **Preprocess and Split Data**
   ```bash
   python src/preprocess.py
   ```

3️⃣ **Train CNN Model**
   ```bash
   python src/train.py
   ```

4️⃣ **Evaluate Model**
   ```bash
   python src/model_evaluation.py
   ```

## 📊 MLflow Tracking

Run the MLflow UI to see experiment logs:
```bash
mlflow ui
```
Visit http://localhost:5000 in your browser.

## ⚙️ Example Config File (`config.yaml`)

```yaml
data_loader:
  dataset_path: "data/Data/genres_original"
  output_path: "data/data.json"
  sample_rate: 22050
  duration: 30
  num_mfcc: 13
  n_fft: 2048
  hop_length: 512
  num_segments: 10

preprocess:
  test_size: 0.2
  random_seed: 42

training:
  batch_size: 32
  epochs: 30
  model_output_path: "models/music_genre_cnn_model.h5"
```

## 🧰 Tech Stack

* Python 3.8+
* TensorFlow / Keras
* Librosa (audio processing)
* scikit-learn
* MLflow
* PyYAML

Install all with:
```bash
pip install -r requirements.txt
```

## ✅ Features

* 🎶 Audio feature extraction using MFCC
* 🧠 CNN architecture for multi-class genre classification
* 📈 MLflow logging for metrics and models
* 🔌 Modular code with config-based control
