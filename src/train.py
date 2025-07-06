import os
import sys
import numpy as np
import tensorflow as tf
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt

# Add the src directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils import (
    load_and_preprocess_data, 
    create_model, 
    plot_training_history, 
    save_model,
    plot_confusion_matrix
)

def main():
    # Configuration
    DATA_DIR = "data/asl_alphabet_train"
    MODEL_SAVE_PATH = "models/asl_detection_model.h5"
    IMG_SIZE = (64, 64)
    BATCH_SIZE = 32
    EPOCHS = 20
    
    print("Starting ASL Sign Language Detection Model Training")
    print("=" * 50)
    
    # Check if data directory exists
    if not os.path.exists(DATA_DIR):
        print(f"Error: Data directory '{DATA_DIR}' not found!")
        print("Please make sure you have downloaded and extracted the ASL Alphabet dataset.")
        print("Expected structure: data/asl_alphabet_train/ with subdirectories A/, B/, C/, etc.")
        return
    
    # Load and preprocess data
    print("Loading and preprocessing data...")
    try:
        train_generator, val_generator = load_and_preprocess_data(
            DATA_DIR, 
            img_size=IMG_SIZE, 
            batch_size=BATCH_SIZE
        )
        
        num_classes = len(train_generator.class_indices)
        print(f"Number of classes: {num_classes}")
        print(f"Class indices: {train_generator.class_indices}")
        
        # Save class indices for later use
        import json
        with open("models/class_indices.json", "w") as f: 
            json.dump(train_generator.class_indices, f)
        
    except Exception as e:
        print(f"Error loading data: {e}")
        return
    
    # Create model
    print("Creating model...")
    model = create_model(num_classes, IMG_SIZE)
    
    # Compile model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Print model summary
    model.summary()
    
    # Callbacks
    callbacks = [
        ModelCheckpoint(
            MODEL_SAVE_PATH,
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        ),
        EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=3,
            min_lr=1e-7,  # type: ignore
            verbose=1
        )
    ]
    
    # Train model
    print("Starting training...")
    history = model.fit(
        train_generator,
        epochs=EPOCHS,
        validation_data=val_generator,
        callbacks=callbacks,
        verbose="1"
    )
    
    # Plot training history
    print("Plotting training history...")
    plot_training_history(history)
    
    # Evaluate model
    print("Evaluating model...")
    val_loss, val_accuracy = model.evaluate(val_generator, verbose="1")
    print(f"Validation Accuracy: {val_accuracy:.4f}")
    print(f"Validation Loss: {val_loss:.4f}")
    
    # Save the final model
    save_model(model, MODEL_SAVE_PATH)
    
    print("Training completed successfully!")
    print(f"Model saved to: {MODEL_SAVE_PATH}")
    print(f"Class indices saved to: models/class_indices.json")

if __name__ == "__main__":
    main() 