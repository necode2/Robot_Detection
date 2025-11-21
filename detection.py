""" final code aided wirth claude ai """

import cv2
import numpy as np
import os

class FaceRecognizer:
    def __init__(self, contacts_folder="contacts_data"):
        """
        Simple face recognizer using template matching.
        
        Args:
            contacts_folder: Path to folder containing images named as "PersonName.jpg"
        """
        self.contacts_folder = contacts_folder
        self.known_faces = []
        self.known_names = []
        
        # Initialize face detector (Haar Cascade)
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        
        # Load known faces from contacts folder
        self.load_contacts()
        
    def load_contacts(self):
        """Load all faces from the contacts folder."""
        if not os.path.exists(self.contacts_folder):
            print(f"Creating {self.contacts_folder} folder...")
            os.makedirs(self.contacts_folder)
            print(f"Please add contact images to the '{self.contacts_folder}' folder")
            print("Name them like: 'John.jpg', 'Sarah.png', etc.")
            return
        
        print("Loading contacts...")
        
        for filename in os.listdir(self.contacts_folder):
            if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                # Load image
                path = os.path.join(self.contacts_folder, filename)
                img = cv2.imread(path)
                
                if img is None:
                    print(f"  ✗ Could not load: {filename}")
                    continue
                    
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                
                # Detect faces in the image
                detected_faces = self.face_cascade.detectMultiScale(
                    gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
                )
                
                if len(detected_faces) > 0:
                    # Use the first face found
                    (x, y, w, h) = detected_faces[0]
                    face_roi = gray[y:y+h, x:x+w]
                    
                    # Resize to standard size for comparison
                    face_roi = cv2.resize(face_roi, (100, 100))
                    
                    self.known_faces.append(face_roi)
                    
                    # Use filename without extension as name
                    name = os.path.splitext(filename)[0]
                    self.known_names.append(name)
                    print(f"  ✓ Loaded: {name}")
                else:
                    print(f"  ✗ No face found in: {filename}")
        
        print(f"\nLoaded {len(self.known_names)} contacts\n")
    
    def compare_faces(self, face1, face2):
        """
        Compare two face images using histogram comparison.
        Returns a similarity score (higher = more similar).
        """
        # Calculate histograms
        hist1 = cv2.calcHist([face1], [0], None, [256], [0, 256])
        hist2 = cv2.calcHist([face2], [0], None, [256], [0, 256])
        
        # Normalize
        hist1 = cv2.normalize(hist1, hist1).flatten()
        hist2 = cv2.normalize(hist2, hist2).flatten()
        
        # Compare using correlation (returns value between -1 and 1)
        similarity = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
        
        return similarity
    
    def recognize_faces(self, frame):
        """
        Detect and recognize faces in a frame.
        
        Args:
            frame: BGR image from OpenCV
            
        Returns:
            frame with boxes and labels drawn
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = self.face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
        )
        
        # Process each detected face
        for (x, y, w, h) in faces:
            # Extract and resize face
            face_roi = gray[y:y+h, x:x+w]
            face_roi_resized = cv2.resize(face_roi, (100, 100))
            
            name = "Unknown"
            color = (0, 0, 255)  # Red for unknown
            best_score = 0
            
            # Compare with known faces
            if self.known_faces:
                for idx, known_face in enumerate(self.known_faces):
                    score = self.compare_faces(face_roi_resized, known_face)
                    
                    if score > best_score:
                        best_score = score
                        
                        # Threshold for recognition (0.65 = 65% similarity)
                        if score > 0.65:
                            name = self.known_names[idx]
                            color = (0, 255, 0)  # Green for known
            
            # Draw box around face
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            
            # Draw label below face
            conf_text = f" ({int(best_score*100)}%)" if best_score > 0 else ""
            label_text = f"{name}{conf_text}"
            
            cv2.rectangle(frame, (x, y+h), (x+w, y+h+35), color, cv2.FILLED)
            cv2.putText(frame, label_text, (x+6, y+h+25), 
                       cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255), 1)
        
        return frame
    
    def run(self):
        """Start the webcam and run face recognition."""
        print("Starting webcam... Press 'q' to quit")
        
        # Open webcam
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("Error: Could not open webcam")
            return
        
        # Set resolution for better performance
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        while True:
            # Capture frame
            ret, frame = cap.read()
            
            if not ret:
                print("Error: Could not read frame")
                break
            
            # Recognize faces in frame
            frame = self.recognize_faces(frame)
            
            # Display instructions
            cv2.putText(frame, "Press 'q' to quit", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Show frame
            cv2.imshow('Face Recognition', frame)

            if cv2.waitKey(1) == 27:
                break
        
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()


# Run the recognizer
if __name__ == "__main__":
    print("=" * 50)
    print("SIMPLE FACE RECOGNITION SYSTEM")
    print("=" * 50)
    
    recognizer = FaceRecognizer(contacts_folder="contacts_data")
    recognizer.run()