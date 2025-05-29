# Inspired by https://medium.com/@aiphile/iris-segmentation-mediapipe-python-a4deb711aae3
# geraadpleegd op 22/04/2025
from dataclasses import dataclass, field
from typing import Optional, List, Any
import cv2 as cv 
import mediapipe as mp 
import numpy as np

# Constans
# Standard eye image size for RT-GENE model
EYE_IMAGE_WIDTH = 60
EYE_IMAGE_HEIGHT = 36
# Inidices are based on the mediapipe topology
# left eye and pupil indices
LEFT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
LEFT_IRIS = [469, 470, 471, 472]
# right eyes indices

# https://research.google/blog/mediapipe-iris-real-time-iris-tracking-depth-estimation/
RIGHT_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
RIGHT_IRIS = [474, 475, 476, 477]

@dataclass
class EyeData:
    """Data class to hold eye detection and tracking results"""
    detected: bool = False
    is_blinking: bool = False
    x_center: Optional[int] = None
    y_center: Optional[int] = None
    radius: Optional[int] = None
    image: Optional[np.ndarray] = None
    vertical_ratio: float = None
    horizontal_ratio: float = None
    @property
    def center(self):
        return (self.x_center, self.y_center)

class EyeDetector:
    def __init__(self):
        self.frame = None
        # Create instances of the dataclass for each eye
        self.left_eye = EyeData()
        self.right_eye = EyeData()
    
    def process_frame(self, frame, face_landmarks):
        self.frame = frame
        self.detect_eyes(face_landmarks)
        return self.frame, self.left_eye, self.right_eye

    def detect_eyes(self, face_landmarks):
        """Detects the eyes in the frame and updates the EyeData instances."""
        img_h, img_w = self.frame.shape[:2]
        # Calculate the coordinates 
        mesh_points=np.array([np.multiply([p.x, p.y], [img_w, img_h]).astype(int) 
        for p in face_landmarks[0].landmark])
        (l_cx, l_cy), l_radius = cv.minEnclosingCircle(mesh_points[LEFT_IRIS])
        (r_cx, r_cy), r_radius = cv.minEnclosingCircle(mesh_points[RIGHT_IRIS])
        center_left = np.array([l_cx, l_cy], dtype=np.int32)
        center_right = np.array([r_cx, r_cy], dtype=np.int32)
        # Update left EyeData
        self.left_eye.detected = True
        self.left_eye.x_center = center_left[0]
        self.left_eye.y_center = center_left[1]
        self.left_eye.radius = l_radius
        # Update right EyeData
        self.right_eye.detected = True
        self.right_eye.x_center = center_right[0]
        self.right_eye.y_center = center_right[1]
        self.right_eye.radius = r_radius
        
        # Create eye image for later processing with rt_gene
        left_eye_image = self.create_eye_image(self.left_eye)
        self.left_eye.image = left_eye_image
        right_eye_image = self.create_eye_image(self.right_eye)
        self.right_eye.image = right_eye_image
        # Calculate horizontal and vertical ratios
        self.calculate_horizontal_vertical_ratios(face_landmarks[0].landmark)
        # Check if the eyes are blinking
        self.is_blinking()
        # Draw the eye information on the frame
        self.draw_eye_info()

    def create_eye_image(self, eye_data):
        if not eye_data.detected or None in (eye_data.x_center, eye_data.y_center, eye_data.radius):
            return None
        try:
            # Extract a square region around the eye
            radius = eye_data.radius
            # Make sure we have a reasonable radius
            if radius < 5:
                radius = 5
                
            # Calculate the bounding box with some margin
            frame = self.frame
            margin = int(radius * 1.5)
            x_min = max(0, eye_data.x_center - margin)
            y_min = max(0, eye_data.y_center - margin)
            x_max = min(frame.shape[1], eye_data.x_center + margin)
            y_max = min(frame.shape[0], eye_data.y_center + margin)
            
            # Extract the eye region
            eye_region = frame[y_min:y_max, x_min:x_max]
            
            # Resize to the required dimensions for the RT-GENE model
            eye_image = cv.resize(eye_region, (EYE_IMAGE_WIDTH, EYE_IMAGE_HEIGHT))
            
            # Preprocess the image for the model
            # RT-GENE expects BGR images with specific mean subtraction
            eye_image = eye_image.reshape(EYE_IMAGE_HEIGHT, EYE_IMAGE_WIDTH, 3, order='F').astype(float)
            
            # Initialize an array for the preprocessed image
            processed_image = np.zeros((EYE_IMAGE_HEIGHT, EYE_IMAGE_WIDTH, 3))
            
            # Subtract mean values for each channel (BGR)
            processed_image[:, :, 0] = eye_image[:, :, 0] - 103.939
            processed_image[:, :, 1] = eye_image[:, :, 1] - 116.779
            processed_image[:, :, 2] = eye_image[:, :, 2] - 123.68
            
            return processed_image
        except Exception as e:
            print(f"Error creating eye image: {e}")
            return None


    def draw_eye_info(self):
        """Draws the eye information on the frame."""
        cv.circle(self.frame, self.left_eye.center, int(self.left_eye.radius), (0,255,0), 2, cv.LINE_AA)
        cv.circle(self.frame, self.right_eye.center, int(self.right_eye.radius), (0,255,0), 2, cv.LINE_AA)
        cv.putText(self.frame, f"Left Eye: {self.left_eye.is_blinking}", (self.left_eye.x_center, self.left_eye.y_center - 10), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv.putText(self.frame, f"Right Eye: {self.right_eye.is_blinking}", (self.right_eye.x_center, self.right_eye.y_center - 10), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    def calculate_horizontal_vertical_ratios(self, face_landmarks):
        """Calculates the horizontal and vertical ratios for both eyes."""
        image = self.frame
        if not self.left_eye.detected or not self.right_eye.detected:
            return None, None, None, None
        # Extract the bounding boxes pints and Denormalise them on the input image dimensions    
        # FIXED: Swapped left and right eye landmarks to match correct eye
        # Right eye (from subject's view) is on left side of image
        right_eye_miny = int(face_landmarks[386].y * image.shape[0])
        right_eye_maxy = int(face_landmarks[374].y * image.shape[0])
        right_eye_minx = int(face_landmarks[362].x * image.shape[1])            
        right_eye_maxx = int(face_landmarks[263].x * image.shape[1])
        
        # Left eye (from subject's view) is on right side of image
        left_eye_miny = int(face_landmarks[159].y * image.shape[0])
        left_eye_maxy = int(face_landmarks[145].y * image.shape[0])
        left_eye_minx = int(face_landmarks[33].x * image.shape[1])            
        left_eye_maxx = int(face_landmarks[133].x * image.shape[1])
        
        left_eye  = [left_eye_miny, left_eye_maxy, left_eye_minx, left_eye_maxx]
        right_eye = [right_eye_miny, right_eye_maxy, right_eye_minx, right_eye_maxx]
        
        # Calculate the horizontal and vertical ratio for both eyes
        h_left_range  = np.abs(left_eye[3] - left_eye[2])
        h_right_range = np.abs(right_eye[3] - right_eye[2])
                
        h_pupil_left  = np.abs(self.left_eye.x_center  - left_eye[2])  / h_left_range
        h_pupil_right = np.abs(self.right_eye.x_center - right_eye[2]) / h_right_range

        v_left_range  = np.abs(left_eye[1] - left_eye[0])
        v_right_range = np.abs(right_eye[1] - right_eye[0])            
                           
        v_pupil_left  = np.abs(self.left_eye.y_center  - left_eye[0])  / v_left_range
        v_pupil_right = np.abs(self.right_eye.y_center - right_eye[0]) / v_right_range
        # Update the horizontal and vertical ratios in the EyeData instances
        self.left_eye.horizontal_ratio = h_pupil_left
        self.right_eye.horizontal_ratio = h_pupil_right
        self.left_eye.vertical_ratio = v_pupil_left
        self.right_eye.vertical_ratio = v_pupil_right

    def is_blinking(self):
        """Checks if the eyes are blinking based on the vertical ratio."""
        if not self.left_eye.detected or not self.right_eye.detected:
            return False
        # Check if the vertical ratio is below a certain threshold
        blink_threshold = 0.2
        self.left_eye.is_blinking = self.left_eye.vertical_ratio < blink_threshold
        self.right_eye.is_blinking = self.right_eye.vertical_ratio < blink_threshold
