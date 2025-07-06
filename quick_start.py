#!/usr/bin/env python3
"""
Quick Start Script for ASL Sign Language Detection
"""

import os
import sys
import subprocess
import tkinter as tk
from tkinter import messagebox

def check_model_exists():
    """Check if the trained model exists"""
    model_path = "models/asl_detection_model.h5"
    class_indices_path = "models/class_indices.json"
    
    if not os.path.exists(model_path):
        return False, "Model file not found"
    if not os.path.exists(class_indices_path):
        return False, "Class indices file not found"
    
    return True, "Model ready"

def run_training():
    """Run the training script"""
    print("Starting model training...")
    print("This may take a while depending on your hardware.")
    print("Make sure you have the dataset in the data/ folder.")
    
    try:
        subprocess.run([sys.executable, "src/train.py"], check=True)
        print("Training completed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Training failed: {e}")
        return False

def run_gui():
    """Run the GUI application"""
    print("Starting GUI application...")
    try:
        subprocess.run([sys.executable, "src/gui.py"], check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"GUI failed to start: {e}")
        return False

def run_detection():
    """Run the command-line detection"""
    print("Starting command-line detection...")
    print("Press 'q' to quit, 's' to save frame")
    try:
        subprocess.run([sys.executable, "src/detect.py"], check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Detection failed: {e}")
        return False

def show_menu():
    """Show the main menu"""
    print("\n" + "="*50)
    print("ASL Sign Language Detection - Quick Start")
    print("="*50)
    
    # Check model status
    model_ready, status_msg = check_model_exists()
    
    print(f"\nModel Status: {status_msg}")
    
    if not model_ready:
        print("\nOptions:")
        print("1. Train the model (requires dataset)")
        print("2. Exit")
        
        choice = input("\nEnter your choice (1-2): ").strip()
        
        if choice == "1":
            if run_training():
                print("Model trained successfully! You can now run the detection.")
            else:
                print("Training failed. Please check your dataset and try again.")
        else:
            print("Goodbye!")
            return
    else:
        print("\nOptions:")
        print("1. Start GUI Application")
        print("2. Start Command-line Detection")
        print("3. Retrain the model")
        print("4. Exit")
        
        choice = input("\nEnter your choice (1-4): ").strip()
        
        if choice == "1":
            run_gui()
        elif choice == "2":
            run_detection()
        elif choice == "3":
            if run_training():
                print("Model retrained successfully!")
            else:
                print("Training failed. Please check your dataset and try again.")
        else:
            print("Goodbye!")

def main():
    """Main function"""
    print("Welcome to ASL Sign Language Detection!")
    
    # Check if we're in the right directory
    if not os.path.exists("src"):
        print("Error: Please run this script from the project root directory.")
        print("Make sure you're in the Sign_Language_detection folder.")
        return
    
    # Check if requirements are installed
    try:
        import tensorflow
        import cv2
        import numpy
    except ImportError as e:
        print(f"Error: Missing required package - {e}")
        print("Please run: pip install -r requirements.txt")
        return
    
    show_menu()

if __name__ == "__main__":
    main() 