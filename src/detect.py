import os
import sys
import cv2
import numpy as np
import tensorflow as tf
import json
import time

# Add the src directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils import preprocess_frame, extract_hand_region, load_model

def main():
    # Configuration
    MODEL_PATH = "models/asl_detection_model.h5"
    CLASS_INDICES_PATH = "models/class_indices.json"
    IMG_SIZE = (64, 64)
    CONFIDENCE_THRESHOLD = 0.7
    
    print("Starting ASL Sign Language Detection")
    print("=" * 40)
    print("Press 'q' to quit")
    print("Press 's' to save current frame")
    print("=" * 40)
    
    # Load model
    model = load_model(MODEL_PATH)
    if model is None:
        print("Error: Could not load model. Please train the model first.")
        return
    
    # Load class indices
    if os.path.exists(CLASS_INDICES_PATH):
        with open(CLASS_INDICES_PATH, 'r') as f:
            class_indices = json.load(f)
        # Create reverse mapping
        idx_to_class = {v: k for k, v in class_indices.items()}
    else:
        print("Error: Class indices file not found.")
        return
    
    # Initialize webcam
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return
    
    # Set webcam properties
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    # Variables for tracking
    prediction_history = []
    history_length = 5
    current_prediction = None
    confidence = 0.0
    
    print("Starting detection...")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break
        
        # Flip frame horizontally for mirror effect
        frame = cv2.flip(frame, 1)
        
        # Extract hand region
        hand_region, bbox = extract_hand_region(frame)
        
        if hand_region is not None and bbox is not None:
            # Draw bounding box around hand
            x, y, w, h = bbox
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            # Preprocess hand region for prediction
            processed_frame = preprocess_frame(hand_region, IMG_SIZE)
            
            # Make prediction
            predictions = model.predict(processed_frame, verbose=0)
            predicted_class_idx = np.argmax(predictions[0])
            confidence = predictions[0][predicted_class_idx]
            
            # Get predicted class name
            predicted_class = idx_to_class.get(predicted_class_idx, "Unknown")
            
            # Add to history for stability
            prediction_history.append(predicted_class)
            if len(prediction_history) > history_length:
                prediction_history.pop(0)
            
            # Use most common prediction in recent history
            from collections import Counter
            most_common = Counter(prediction_history).most_common(1)
            if most_common and len(most_common) > 0 and most_common[0][1] >= history_length // 2:
                current_prediction = most_common[0][0]
            else:
                current_prediction = predicted_class
            
            # Display prediction and confidence
            if confidence > CONFIDENCE_THRESHOLD:
                # Draw prediction text
                text = f"{current_prediction} ({confidence:.2f})"
                cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                           1, (0, 255, 0), 2)
                
                # Draw background rectangle for text
                text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
                cv2.rectangle(frame, (5, 5), (text_size[0] + 15, text_size[1] + 15), 
                             (0, 0, 0), -1)
                cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                           1, (0, 255, 0), 2)
            else:
                cv2.putText(frame, "No clear sign detected", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        else:
            cv2.putText(frame, "No hand detected", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        # Display instructions
        cv2.putText(frame, "Press 'q' to quit, 's' to save", (10, frame.shape[0] - 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # Show frame
        cv2.imshow('ASL Sign Language Detection', frame)
        
        # Handle key presses
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            # Save current frame
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"captured_frame_{timestamp}.jpg"
            cv2.imwrite(filename, frame)
            print(f"Frame saved as {filename}")
    
    # Clean up
    cap.release()
    cv2.destroyAllWindows()
    print("Detection stopped.")

if __name__ == "__main__":
    main() 