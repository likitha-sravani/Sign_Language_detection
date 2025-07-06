# Sign Language Detection Project

A desktop application for real-time sign language detection using webcam input. This project uses TensorFlow and OpenCV to detect American Sign Language (ASL) letters and words.

## Features

- Real-time webcam sign language detection
- Support for ASL alphabet (A-Z) and common words
- Desktop GUI application
- Model training and evaluation tools

## Setup Instructions

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Dataset Setup

1. Download the ASL Alphabet Dataset from Kaggle: https://www.kaggle.com/datasets/grassknoted/asl-alphabet
2. Extract the dataset to the `data/` folder
3. The structure should look like:
   ```
   data/
   ├── asl_alphabet_train/
   │   ├── A/
   │   ├── B/
   │   ├── ...
   │   └── Z/
   └── asl_alphabet_test/
       ├── A/
       ├── B/
       ├── ...
       └── Z/
   ```

### 3. Project Structure

```
Sign_Language_detection/
│
├── data/                    # Dataset directory
├── models/                  # Saved trained models
├── src/                     # Source code
│   ├── train.py            # Model training script
│   ├── detect.py           # Webcam detection script
│   ├── gui.py              # Desktop app GUI
│   └── utils.py            # Utility functions
├── requirements.txt
└── README.md
```

## Usage

### Training the Model

```bash
python src/train.py
```

### Running the Desktop App

```bash
python src/gui.py
```

### Real-time Detection (Command Line)

```bash
python src/detect.py
```

## Model Architecture

- CNN-based image classification model
- Input: 64x64x3 RGB images
- Output: 29 classes (A-Z, space, delete, nothing)

## Requirements

- Python 3.8+
- Webcam
- GPU (optional, for faster training)

## License

This project is for educational purposes. 