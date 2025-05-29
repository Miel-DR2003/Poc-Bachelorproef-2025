import cv2
from dataclasses import dataclass, field


from .object_detection.face_detection import FaceDetector
from .object_detection.eye_detection import EyeDetector
from .chinrest.virtual_chinrest import VirtualChinrest
from .mouse.mouse_movement import MouseControl
from .lighting.lighting_handler import LightingHandler
from .object_detection.gaze_detection import GazeEstimation

@dataclass
class TrackingData:
    gaze_history: list = field(default_factory=list)
    current_gaze: str = "unknown"
    max_history: int = 30  # Store last 30 gaze points
    gaze_dir: str = "center"

class EyeTrackingSystem:
    def __init__(self):
        """
        Initialize the eye tracking system.
        """
        # Initialize components
        self.lighting_handler = LightingHandler()
        self.face_detector = FaceDetector()
        self.chinrest = VirtualChinrest(self.face_detector)
        self.eye_detector = EyeDetector()
        self.gaze_detector = GazeEstimation()  # Placeholder for gaze detection
        self.mouse = MouseControl()
        
        # System state
        self.is_calibrated = False
        self.is_tracking = False
        self.head_position_valid = False
        
        # Tracking data
        self.tracking_data = TrackingData()
        
        # UI settings
        self.show_mesh = True
        self.show_eyes = True
        
        # Colors
        self.COLOR_RED = (0, 0, 255)
        self.COLOR_GREEN = (0, 255, 0)
        self.COLOR_BLUE = (255, 0, 0)
        self.COLOR_YELLOW = (0, 255, 255)

    def toggle_mesh_visibility(self):
        """Toggle face mesh visibility."""
        self.show_mesh = not self.show_mesh
        
    def toggle_eyes_visibility(self):
        """Toggle eye detection visibility."""
        self.show_eyes = not self.show_eyes

    def toggle_mouse_control(self):
        """Toggle mouse control."""
        self.mouse.toggleMouseControl()
                
    def calibrate_chinrest(self):
        """Start chinrest calibration."""
        self.chinrest.reset_calibration()
        return True
        
    def is_chinrest_calibrated(self):
        """Check if chinrest is calibrated."""
        return self.chinrest.is_calibrated
            
    def draw_gaze_visualization(self, image):
        """
        Draw gaze visualization on the image.
        
        Args:
            image: Input image
            
        Returns:
            image: Image with gaze visualization
        """
        # Draw gaze coordinates in the top right corner
        gaze_text = f"Gaze: ({self.tracking_data.gaze_dir}"
        text_size = cv2.getTextSize(gaze_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
        image_width = image.shape[1]
        cv2.putText(image, gaze_text, (image_width - text_size[0] - 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.COLOR_GREEN, 2)
        return image
    
    def draw_system_info(self, image):
        """
        Draw system information on the image.
        
        Args:
            image: Input image
            
        Returns:
            image: Image with system information
        """
        h, w = image.shape[:2]        
        # Draw system status
        status_text = "TRACKING" if self.is_tracking else "NOT TRACKING"
        status_color = self.COLOR_GREEN if self.is_tracking else self.COLOR_RED
        cv2.putText(image, status_text, (w - 120, h - 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, status_color, 1)
        
        # Draw mouse control status
        control_status = "Mouse Control: ON" if self.mouse.mouse_control else "Mouse Control: OFF (press 'm' to toggle)"
        cv2.putText(image, control_status, (10, h - 60), cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 255, 0), 2)

        # Draw help text
        help_text = "Press: C-Calibrate, M-Toggle Mesh, E-Toggle Eyes, Q-Quit"
        cv2.putText(image, help_text, (10, h - 20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.COLOR_YELLOW, 1)
        
        return image
    
    def process_frame(self, frame, calibrate_chinrest=False):
        """
        Process a frame through the eye tracking pipeline.
        
        Args:
            frame: Input frame from camera
            calibrate_chinrest: Whether to calibrate the chinrest
            
        Returns:
            processed_frame: Processed frame with visualizations
            tracking_data: Dictionary containing tracking data
        """        
        # Make a copy of the frame
        processed_frame = frame.copy()

        # Step 1: Enhance image for better eye detection in different lighting conditions
        processed_frame = self.lighting_handler.apply_clahe(processed_frame)
        
        # Step 2: Face detection
        if self.show_mesh:
            processed_frame, face_landmarks, face_data = self.face_detector.find_faces(processed_frame)
        else:
            # Skip drawing the mesh but still get the data
            _, face_landmarks, face_data = self.face_detector.find_faces(frame)
        
        # Step 3: Virtual chinrest processing
        processed_frame, head_position_valid = self.chinrest.process_frame(
            processed_frame, face_data, calibrate_chinrest
        )
        self.head_position_valid = head_position_valid
        
        # Step 4: Eye detection and tracking (only if head position is valid)
        if face_data.detected:
            if self.show_eyes:
                # face_landmarks -> results.multi_face_landmarks
                processed_frame, left_eye, right_eye = self.eye_detector.process_frame(processed_frame, face_landmarks)
            else:
                # Skip drawing eye visualization but still get the data
                _, left_eye, right_eye = self.eye_detector.process_frame(frame, face_landmarks)
                
        # Draw system information
        processed_frame = self.draw_system_info(processed_frame)
        self.left_eye = left_eye
        self.right_eye = right_eye
        # Update tracking state
        self.is_tracking = face_data.detected and self.head_position_valid

        # Step 5: Estimate gaze
        if self.is_tracking:
            frame, gaze_dir = self.gaze_detector.process_frame(frame, face_data, left_eye, right_eye)
            self.tracking_data.gaze_dir = gaze_dir
            # Draw gaze visualization
            processed_frame = self.draw_gaze_visualization(processed_frame)
            # Step 7: Move mouse (if enabled)
            if self.mouse.mouse_control:
                self.mouse.move_mouse_2(gaze_dir)
                # Step 8: Perform click (if is blinking)
                if left_eye.is_blinking or right_eye.is_blinking:
                    self.mouse.performClick()

        return processed_frame, self.tracking_data

        
    




    

