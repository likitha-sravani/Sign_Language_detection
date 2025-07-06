import os
import sys
import cv2
import numpy as np
import tensorflow as tf
import json
import threading
import time
from tkinter import Tk, Frame, Label, Button, BOTH, LEFT, X, RAISED, NORMAL, DISABLED, messagebox
from PIL import Image, ImageTk

# Add the src directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils import preprocess_frame, extract_hand_region, load_model

class ASLDetectionGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("ASL Sign Language Detection")
        self.root.geometry("800x700")  # Increased height for buttons
        self.root.configure(bg='#2c3e50')
        self.root.resizable(False, False)  # Prevent resizing to avoid hiding controls
        
        # Configuration
        self.MODEL_PATH = "models/asl_detection_model.h5"
        self.CLASS_INDICES_PATH = "models/class_indices.json"
        self.IMG_SIZE = (64, 64)
        self.CONFIDENCE_THRESHOLD = 0.5
        
        # Variables
        self.is_detecting = False
        self.cap = None
        self.model = None
        self.idx_to_class = {}
        self.prediction_history = []
        self.history_length = 5
        self.current_prediction = None
        self.confidence = 0.0
        
        # Initialize components
        self.setup_ui()
        self.load_model()
        
    def setup_ui(self):
        """Setup the user interface"""
        # Main frame
        main_frame = Frame(self.root, bg='#2c3e50')
        main_frame.pack(fill=BOTH, padx=10, pady=10)  # Removed expand=True
        
        # Title
        title_label = Label(main_frame, text="ASL Sign Language Detection", 
                           font=('Arial', 20, 'bold'), bg='#2c3e50', fg='white')
        title_label.pack(pady=(0, 20))
        
        # Control frame (now above video frame)
        control_frame = Frame(main_frame, bg='yellow')  # Bright color for debugging
        control_frame.pack(pady=20)
        
        # Buttons
        self.start_button = Button(control_frame, text="Start Detection", 
                                  command=self.start_detection, 
                                  bg='#27ae60', fg='white', font=('Arial', 12),
                                  width=15, height=2, bd=3, relief=RAISED)
        self.start_button.pack(side=LEFT, padx=20)
        
        self.stop_button = Button(control_frame, text="Stop Detection", 
                                 command=self.stop_detection, 
                                 bg='#e74c3c', fg='white', font=('Arial', 12),
                                 width=15, height=2, state=DISABLED, bd=3, relief=RAISED)
        self.stop_button.pack(side=LEFT, padx=20)
        
        print("Buttons created")
        
        # Video frame (fixed size, does not expand)
        self.video_frame = Label(main_frame, bg='black', width=640, height=480, bd=2, relief=RAISED)
        self.video_frame.pack(pady=10)

        # Prediction display frame (directly below video frame)
        pred_frame = Frame(main_frame, bg='red', relief=RAISED, bd=4)  # Highly visible background for debugging
        pred_frame.pack(fill=X, pady=10, padx=40)

        # Prediction label
        self.pred_label = Label(pred_frame, text="TEST OUTPUT", 
                               font=('Arial', 22, 'bold'), bg='red', fg='#00ffd0', pady=10)
        self.pred_label.pack(pady=(10, 0))

        # Confidence label
        self.conf_label = Label(pred_frame, text="CONFIDENCE TEST", 
                               font=('Arial', 16), bg='red', fg='#f6ff00', pady=5)
        self.conf_label.pack(pady=(0, 10))
        
        # Status frame
        status_frame = Frame(main_frame, bg='#2c3e50')
        status_frame.pack(fill=X, pady=10)
        
        # Status label
        self.status_label = Label(status_frame, text="Ready", 
                                 font=('Arial', 10), bg='#2c3e50', fg='#95a5a6')
        self.status_label.pack()
        
        # Instructions
        instructions = """
        Instructions:
        • Click 'Start Detection' to begin
        • Show your hand to the webcam
        • The app will detect ASL signs in real-time
        • Click 'Stop Detection' to end
        """
        inst_label = Label(main_frame, text=instructions, 
                          font=('Arial', 10), bg='#2c3e50', fg='#bdc3c7',
                          justify=LEFT)
        inst_label.pack(pady=10)
        
    def load_model(self):
        """Load the trained model and class indices"""
        try:
            # Load model
            self.model = load_model(self.MODEL_PATH)
            if self.model is None:
                messagebox.showerror("Error", "Could not load model. Please train the model first.")
                return False
            
            # Load class indices
            if os.path.exists(self.CLASS_INDICES_PATH):
                with open(self.CLASS_INDICES_PATH, 'r') as f:
                    class_indices = json.load(f)
                self.idx_to_class = {v: k for k, v in class_indices.items()}
                self.status_label.config(text=f"Model loaded successfully. {len(self.idx_to_class)} classes available.")
                return True
            else:
                messagebox.showerror("Error", "Class indices file not found.")
                return False
                
        except Exception as e:
            messagebox.showerror("Error", f"Error loading model: {str(e)}")
            return False
    
    def start_detection(self):
        """Start the detection process"""
        if self.model is None:
            messagebox.showerror("Error", "Model not loaded. Please check the model file.")
            return
        
        try:
            # Initialize webcam
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                messagebox.showerror("Error", "Could not open webcam.")
                return
            
            # Set webcam properties
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            
            self.is_detecting = True
            self.start_button.config(state=DISABLED)
            self.stop_button.config(state=NORMAL)
            self.status_label.config(text="Detection started...")
            
            # Start detection thread
            self.detection_thread = threading.Thread(target=self.detection_loop)
            self.detection_thread.daemon = True
            self.detection_thread.start()
            
        except Exception as e:
            messagebox.showerror("Error", f"Error starting detection: {str(e)}")
    
    def stop_detection(self):
        """Stop the detection process"""
        self.is_detecting = False
        self.start_button.config(state=NORMAL)
        self.stop_button.config(state=DISABLED)
        self.status_label.config(text="Detection stopped.")
        
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        
        # Clear video frame
        self.video_frame.config(image='')
        self.pred_label.config(text="No detection running")
        self.conf_label.config(text="")
    
    def detection_loop(self):
        """Main detection loop running in separate thread"""
        while self.is_detecting:
            if self.cap is None:
                break
                
            ret, frame = self.cap.read()
            if not ret:
                break
            
            # Flip frame horizontally for mirror effect
            frame = cv2.flip(frame, 1)
            
            # Extract hand region
            hand_region, bbox = extract_hand_region(frame)
            
            if hand_region is not None and bbox is not None and self.model is not None:
                # Draw bounding box around hand
                x, y, w, h = bbox
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                
                # Preprocess hand region for prediction
                processed_frame = preprocess_frame(hand_region, self.IMG_SIZE)
                
                # Make prediction
                predictions = self.model.predict(processed_frame, verbose=0)
                self.confidence = float(np.max(predictions[0]))
                predicted_class_idx = int(np.argmax(predictions[0]))
                self.current_prediction = self.idx_to_class.get(predicted_class_idx, "Unknown")
            else:
                self.current_prediction = "No hand detected"
                self.confidence = 0.0
            
            # Update UI in main thread, passing prediction and confidence
            self.root.after(0, self.update_ui, frame, self.current_prediction, self.confidence)
            
            # Small delay to prevent excessive CPU usage
            time.sleep(0.03)

    def update_ui(self, frame, prediction, confidence):
        # Convert frame to PIL Image
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(frame_rgb)
        
        # Resize for display
        display_size = (640, 480)
        pil_image = pil_image.resize(display_size, Image.LANCZOS)
        
        # Convert to PhotoImage
        photo = ImageTk.PhotoImage(pil_image)
        
        # Update video frame
        self.video_frame.config(image=photo)
        self.video_frame.image = photo  # type: ignore
        
        # Update prediction and confidence labels
        self.pred_label.config(text=f"Detected: {prediction}")
        self.conf_label.config(text=f"Confidence: {confidence:.2f}")
        print("Prediction:", prediction, "Confidence:", confidence)
    
    def on_closing(self):
        """Handle window closing"""
        self.stop_detection()
        self.root.destroy()

def main():
    root = Tk()
    app = ASLDetectionGUI(root)
    
    # Handle window closing
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    
    # Start the GUI
    root.mainloop()

if __name__ == "__main__":
    main() 