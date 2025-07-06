# Getting Started with ASL Sign Language Detection

## Quick Start

1. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the Quick Start Script:**
   ```bash
   python quick_start.py
   ```

## Detailed Setup Instructions

### 1. Prerequisites

- Python 3.8 or higher
- Webcam
- GPU (optional, for faster training)

### 2. Dataset Setup

1. Download the ASL Alphabet Dataset from Kaggle:
   - Go to: https://www.kaggle.com/datasets/grassknoted/asl-alphabet
   - Download the dataset (you'll need a Kaggle account)

2. Extract the dataset to the `data/` folder:
   ```
   data/
   ├── asl_alphabet_train/
   │   ├── A/
   │   ├── B/
   │   ├── C/
   │   ├── ...
   │   └── Z/
   └── asl_alphabet_test/
       ├── A/
       ├── B/
       ├── C/
       ├── ...
       └── Z/
   ```

### 3. Training the Model

Run the training script:
```bash
python src/train.py
```

This will:
- Load and preprocess the dataset
- Create a CNN model using transfer learning (MobileNetV2)
- Train the model for 20 epochs (with early stopping)
- Save the best model to `models/asl_detection_model.h5`
- Save class indices to `models/class_indices.json`

**Training time:** 30-60 minutes on CPU, 10-20 minutes on GPU

### 4. Running the Application

#### Option A: Desktop GUI Application
```bash
python src/gui.py
```

Features:
- Real-time webcam feed
- Start/Stop detection buttons
- Live prediction display
- Confidence scores
- User-friendly interface

#### Option B: Command-line Detection
```bash
python src/detect.py
```

Features:
- Real-time webcam detection
- Press 'q' to quit
- Press 's' to save current frame
- Terminal output

### 5. Project Structure

```
Sign_Language_detection/
│
├── data/                    # Dataset directory
│   ├── asl_alphabet_train/  # Training images
│   └── asl_alphabet_test/   # Test images
│
├── models/                  # Saved models
│   ├── asl_detection_model.h5
│   └── class_indices.json
│
├── src/                     # Source code
│   ├── train.py            # Model training
│   ├── detect.py           # Command-line detection
│   ├── gui.py              # Desktop GUI app
│   └── utils.py            # Utility functions
│
├── requirements.txt         # Python dependencies
├── setup.py                # Setup script
├── quick_start.py          # Quick start launcher
├── README.md               # Project overview
└── GETTING_STARTED.md      # This file
```

## How It Works

### 1. Hand Detection
- Uses color-based segmentation to detect skin color
- Applies morphological operations to clean up the mask
- Finds the largest contour (assumed to be the hand)
- Extracts the hand region with padding

### 2. Model Architecture
- **Base Model:** MobileNetV2 (pre-trained on ImageNet)
- **Transfer Learning:** Freezes base model, adds custom layers
- **Custom Layers:**
  - Global Average Pooling
  - Dense layers (512 → 256 → num_classes)
  - Dropout for regularization
- **Output:** 29 classes (A-Z, space, delete, nothing)

### 3. Real-time Processing
- Captures webcam frames at 30 FPS
- Extracts hand region from each frame
- Preprocesses image (resize, normalize)
- Runs inference on the model
- Uses prediction history for stability
- Displays results in real-time

## Troubleshooting

### Common Issues

1. **"Could not open webcam"**
   - Make sure your webcam is connected and not in use by another application
   - Try changing the camera index in the code (0, 1, 2, etc.)

2. **"Model not found"**
   - Run the training script first: `python src/train.py`
   - Make sure the dataset is properly extracted to the `data/` folder

3. **"No hand detected"**
   - Ensure good lighting conditions
   - Position your hand clearly in the camera view
   - Try adjusting the skin color thresholds in `utils.py`

4. **Low accuracy predictions**
   - Retrain the model with more epochs
   - Improve lighting conditions
   - Ensure hand is clearly visible and well-positioned

### Performance Tips

1. **For Training:**
   - Use GPU if available (much faster)
   - Reduce batch size if you run out of memory
   - Increase epochs for better accuracy

2. **For Detection:**
   - Ensure good lighting
   - Keep hand steady and clearly visible
   - Use a plain background for better hand detection

## Customization

### Adding New Signs
1. Create a new folder in `data/asl_alphabet_train/` with your sign name
2. Add training images to the folder
3. Retrain the model

### Adjusting Detection Sensitivity
- Modify `CONFIDENCE_THRESHOLD` in the detection scripts
- Adjust skin color ranges in `extract_hand_region()` function

### Changing Model Architecture
- Modify the `create_model()` function in `utils.py`
- Try different base models (ResNet, EfficientNet, etc.)

## Support

If you encounter issues:
1. Check the troubleshooting section above
2. Ensure all dependencies are installed correctly
3. Verify the dataset structure matches the expected format
4. Check that your webcam is working properly

## License

This project is for educational purposes. Feel free to modify and extend it for your needs. 