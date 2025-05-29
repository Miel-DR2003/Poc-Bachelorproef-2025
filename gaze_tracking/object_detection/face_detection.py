"""
The face detection module uses MediaPipe Face Mesh to detect facial landmarks with high precision. It provides:
- Face landmark detection (478 points)
- Face position tracking
- Bounding box calculation
- Eye region identification
"""
import cv2
import mediapipe as mp
import numpy as np
from dataclasses import dataclass
from typing import Tuple

@dataclass
class FaceData:
    """Data class to hold face detection results"""
    detected: bool = False
    center: tuple = None
    bbox: tuple = None
    landmarks: list = None
    image: np.ndarray = None
    theta: float = 0.0
    phi: float = 0.0

class FaceDetector:
    """
    Face detection module using MediaPipe Face Mesh.
    https://github.com/google-ai-edge/mediapipe/blob/master/docs/solutions/face_mesh.md
    This class handles face detection and landmark extraction.
    """
    def __init__(self, static_image_mode=False, max_num_faces=1, 
                 min_detection_confidence=0.5, min_tracking_confidence=0.5):
        """
        Initialize the face detector with MediaPipe Face Mesh.
        
        Args:
            static_image_mode: Whether to treat the input images as a batch of static images
            max_num_faces: Maximum number of faces to detect
            min_detection_confidence: Minimum confidence for face detection
            min_tracking_confidence: Minimum confidence for face tracking
        """
        self.face_data = FaceData()

        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        # Initialize MediaPipe Face Mesh WITH IRIS model!!!
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=static_image_mode,
            max_num_faces=max_num_faces,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
            refine_landmarks=True  # This enables iris landmarks
        )
        
        # Face mesh indices for specific facial landmarks
        # These indices are based on MediaPipe Face Mesh topology
        # https://www.researchgate.net/publication/376909901_Improving_Detection_of_DeepFakes_through_Facial_Region_Analysis_in_Images/figures?lo=1
        self.FACE_OVAL = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 
                          397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 
                          172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]
                
        # Key points for head pose estimation
        self.NOSE_TIP = 1
        self.LEFT_EYE_LEFT_CORNER = 263
        self.LEFT_EYE_RIGHT_CORNER = 362
        self.RIGHT_EYE_LEFT_CORNER = 133
        self.RIGHT_EYE_RIGHT_CORNER = 33
        self.MOUTH_LEFT = 61
        self.MOUTH_RIGHT = 291
        self.CHIN = 199
        

    def find_faces(self, image):
        """
        Detect faces in the image and extract landmarks.
        
        Args:
            image: Input image (BGR format)
            
        Returns:
            processed_image: Image with face landmarks drawn
            face_landmarks: List of detected face landmarks
            face_data: FaceDetecting data
        """
        self.face_data.image = image
        # Convert the BGR image to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Process the image to find faces
        results = self.face_mesh.process(image_rgb)
        
        # Create a copy of the image to draw on
        processed_image = image.copy()
                
        # Check if any faces were detected
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                # Draw the face mesh
                self.mp_drawing.draw_landmarks(
                    image=processed_image,
                    landmark_list=face_landmarks,
                    connections=self.mp_face_mesh.FACEMESH_TESSELATION, #mesh.FACEMESH_IRISES
                    landmark_drawing_spec=None,
                    connection_drawing_spec=self.mp_drawing_styles.get_default_face_mesh_tesselation_style()
                )
                
                # Draw the face contours
                self.mp_drawing.draw_landmarks(
                    image=processed_image,
                    landmark_list=face_landmarks,
                    connections=self.mp_face_mesh.FACEMESH_CONTOURS,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=self.mp_drawing_styles.get_default_face_mesh_contours_style()
                )
                
                # Extract face position information
                h, w, c = image.shape
                face_3d = []
                face_2d = []
                
                # Get the bounding box of the face
                x_min, y_min = w, h
                x_max, y_max = 0, 0
                
                # Convert landmarks to numpy arrays for easier processing
                landmarks = []
                for idx, lm in enumerate(face_landmarks.landmark):
                    x, y = int(lm.x * w), int(lm.y * h)
                    landmarks.append((x, y))
                    
                    # Update bounding box
                    x_min = min(x_min, x)
                    y_min = min(y_min, y)
                    x_max = max(x_max, x)
                    y_max = max(y_max, y)
                    
                    # Get specific landmarks for pose estimation
                    if idx == self.NOSE_TIP:
                        face_3d.append([lm.x * w, lm.y * h, lm.z * w])
                        face_2d.append([x, y])
                
                # Calculate face center
                face_center = ((x_min + x_max) // 2, (y_min + y_max) // 2)
                
                # Update face position information
                self.face_data.detected = True
                self.face_data.center = face_center
                self.face_data.bbox = (x_min, y_min, x_max, y_max)
                self.face_data.landmarks = landmarks
                
                # Draw bounding box
                cv2.rectangle(processed_image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                
                # Draw face center
                cv2.circle(processed_image, face_center, 5, (255, 0, 0), -1)
                
                # Only process the first face if multiple faces are detected
                break

        self.estimate_head_pose(results, processed_image.shape)
        return processed_image, results.multi_face_landmarks, self.face_data
    
    def estimate_head_pose(self, face_landmarks, image_shape) -> Tuple[float, float]:
        """
        Estimate head pose from MediaPipe face landmarks.
        
        Args:
            face_landmarks: MediaPipe face landmarks
            image_shape: Shape of the input image
            
        Returns:
            Tuple of (theta, phi) head pose angles in radians
        """
        if not face_landmarks:
            return 0.0, 0.0
        
        h, w = image_shape[:2]
        
        # Extract 3D facial landmarks
        face_3d = []
        face_2d = []
        
        # Get key facial landmarks for pose estimation
        if face_landmarks.multi_face_landmarks:
            landmarks = face_landmarks.multi_face_landmarks[0].landmark
        else:
            print("No face detected")
            return 0.0, 0.0  # No face detected, return default values
        
        # Define key points for pose estimation
        key_points = [
            self.NOSE_TIP,
            self.LEFT_EYE_LEFT_CORNER,
            self.LEFT_EYE_RIGHT_CORNER,
            self.RIGHT_EYE_LEFT_CORNER,
            self.RIGHT_EYE_RIGHT_CORNER,
            self.MOUTH_LEFT,
            self.MOUTH_RIGHT,
            self.CHIN
        ]
        
        # Extract 3D and 2D coordinates for key points
        for idx in key_points:
            lm = landmarks[idx]
            x, y = int(lm.x * w), int(lm.y * h)
            
            # Add to 2D points
            face_2d.append([x, y])
            
            # Add to 3D points (z is approximated based on the landmark's z value)
            face_3d.append([x, y, lm.z * 1000])  # Scale z for better estimation
        
        # Convert to numpy arrays
        face_2d = np.array(face_2d, dtype=np.float64)
        face_3d = np.array(face_3d, dtype=np.float64)
        
        # Camera matrix estimation (approximate)
        focal_length = w
        center = (w / 2, h / 2)
        camera_matrix = np.array(
            [[focal_length, 0, center[0]],
             [0, focal_length, center[1]],
             [0, 0, 1]], dtype=np.float64
        )
        
        # Assuming no lens distortion
        dist_coeffs = np.zeros((4, 1), dtype=np.float64)
        
        # Solve PnP
        success, rotation_vector, translation_vector = cv2.solvePnP(
            face_3d, face_2d, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE
        )
        
        if not success:
            return 0.0, 0.0
        
        # Get rotation matrix
        rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
        
        # Convert rotation matrix to Euler angles
        euler_angles = self._rotation_matrix_to_euler_angles(rotation_matrix)
        
        # Convert to theta (vertical) and phi (horizontal) angles
        # Note: The conversion depends on the coordinate system used
        theta = -euler_angles[0]  # Pitch (up/down)
        phi = euler_angles[1]     # Yaw (left/right)
        # Update face data
        self.face_data.theta = theta
        self.face_data.phi = phi
    
    def _rotation_matrix_to_euler_angles(self, R):
        """
        Convert rotation matrix to Euler angles (in radians).
        
        Args:
            R: Rotation matrix
            
        Returns:
            Euler angles [pitch, yaw, roll]
        """
        # Check if the rotation matrix is valid
        sy = np.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
        
        if sy > 1e-6:
            x = np.arctan2(R[2, 1], R[2, 2])
            y = np.arctan2(-R[2, 0], sy)
            z = np.arctan2(R[1, 0], R[0, 0])
        else:
            x = np.arctan2(-R[1, 2], R[1, 1])
            y = np.arctan2(-R[2, 0], sy)
            z = 0
        
        return np.array([x, y, z])
