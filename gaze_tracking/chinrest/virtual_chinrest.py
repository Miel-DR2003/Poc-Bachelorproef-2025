import cv2

class VirtualChinrest:
    """
    Virtual chinrest module for guiding users to maintain proper head positioning.
    This class handles reference position definition, visual guidance, and position validation.
    """
    
    def __init__(self, face_detector):
        """
        Initialize the virtual chinrest.
        
        Args:
            face_detector: Instance of FaceDetector class
        """
        self.face_detector = face_detector
        
        # Reference position parameters
        self.reference_position = None
        self.reference_distance = None
        self.reference_size = None
        
        # Tolerance thresholds
        self.position_tolerance = 50  # pixels
        self.size_tolerance = 0.2     # percentage
        
        # Calibration parameters
        self.is_calibrated = False
        self.calibration_frames = 30
        self.calibration_counter = 0
        self.calibration_positions = []
        
        # Status parameters
        self.position_status = "Not calibrated"
        self.guidance_text = "Please calibrate the chinrest"
        
        # Colors
        self.COLOR_RED = (0, 0, 255)
        self.COLOR_GREEN = (0, 255, 0)
        self.COLOR_BLUE = (255, 0, 0)
        self.COLOR_YELLOW = (0, 255, 255)
        
    def calibrate(self, face_data):
        """
        Calibrate the virtual chinrest by collecting reference face position.
        
        Args:
            face_data: Face data
            
        Returns:
            is_calibrated: Whether calibration is complete
            progress: Calibration progress (0-100%)
        """
        if not face_data.detected:
            return False, 0
        
        # Reset calibration if starting new
        if self.calibration_counter == 0:
            self.calibration_positions = []
            self.is_calibrated = False
        
        # Collect face position data
        if self.calibration_counter < self.calibration_frames:
            # Extract face center and size
            center = face_data.center
            x_min, y_min, x_max, y_max = face_data.bbox
            face_width = x_max - x_min
            face_height = y_max - y_min
            
            # Store position data
            self.calibration_positions.append({
                "center": center,
                "width": face_width,
                "height": face_height
            })
            
            self.calibration_counter += 1
            progress = (self.calibration_counter / self.calibration_frames) * 100
            
            return False, progress
        
        # Calculate average position from collected data
        avg_x = sum(pos["center"][0] for pos in self.calibration_positions) / len(self.calibration_positions)
        avg_y = sum(pos["center"][1] for pos in self.calibration_positions) / len(self.calibration_positions)
        avg_width = sum(pos["width"] for pos in self.calibration_positions) / len(self.calibration_positions)
        avg_height = sum(pos["height"] for pos in self.calibration_positions) / len(self.calibration_positions)
        
        # Set reference position
        self.reference_position = (int(avg_x), int(avg_y))
        self.reference_size = (int(avg_width), int(avg_height))
        
        # Calculate reference distance (can be used for depth estimation)
        # This is a relative measure based on face size
        self.reference_distance = (avg_width + avg_height) / 2
        
        self.is_calibrated = True
        self.position_status = "Calibrated"
        self.guidance_text = "Position calibrated successfully"
        
        return True, 100
    
    def reset_calibration(self):
        """
        Reset the calibration.
        """
        self.reference_position = None
        self.reference_distance = None
        self.reference_size = None
        self.is_calibrated = False
        self.calibration_counter = 0
        self.calibration_positions = []
        self.position_status = "Not calibrated"
        self.guidance_text = "Please calibrate the chinrest"
    
    def check_position(self, face_data):
        """
        Check if the current face position is within tolerance of the reference position.
        
        Args:
            face_data: Face data
            
        Returns:
            position_valid: Whether the position is valid
            deviation: Dictionary with deviation information
        """
        if not self.is_calibrated or not face_data.detected:
            return False, {"x": 0, "y": 0, "size": 0}
        
        # Extract current face data
        current_center = face_data.center
        x_min, y_min, x_max, y_max = face_data.bbox
        current_width = x_max - x_min
        current_height = y_max - y_min
        
        # Calculate position deviation
        x_deviation = current_center[0] - self.reference_position[0]
        y_deviation = current_center[1] - self.reference_position[1]
        
        # Calculate size deviation (as percentage)
        width_ratio = current_width / self.reference_size[0]
        height_ratio = current_height / self.reference_size[1]
        size_deviation = ((width_ratio + height_ratio) / 2) - 1.0
        
        # Check if position is within tolerance
        x_valid = abs(x_deviation) <= self.position_tolerance
        y_valid = abs(y_deviation) <= self.position_tolerance
        size_valid = abs(size_deviation) <= self.size_tolerance
        
        position_valid = x_valid and y_valid and size_valid
        
        # Update status and guidance
        if position_valid:
            self.position_status = "Valid"
            self.guidance_text = "Head position is good"
        else:
            self.position_status = "Invalid"
            
            # Provide specific guidance
            guidance = []
            
            if not x_valid:
                direction = "right" if x_deviation < 0 else "left"
                guidance.append(f"Move head {direction}")
            
            if not y_valid:
                direction = "down" if y_deviation < 0 else "up"
                guidance.append(f"Move head {direction}")
            
            if not size_valid:
                direction = "closer" if size_deviation < 0 else "farther"
                guidance.append(f"Move {direction} to camera")
            
            self.guidance_text = ", ".join(guidance)
        
        # Return validation result and deviation data
        deviation = {
            "x": x_deviation,
            "y": y_deviation,
            "size": size_deviation
        }
        
        return position_valid, deviation
    
    def draw_guidance(self, image, face_data):
        """
        Draw visual guidance for head positioning on the image.
        
        Args:
            image: Input image
            face_data: Face data
            
        Returns:
            image: Image with guidance visualization
        """
        h, w = image.shape[:2]
        
        # Draw calibration status
        status_color = self.COLOR_GREEN if self.is_calibrated else self.COLOR_YELLOW
        cv2.putText(image, f"Status: {self.position_status}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
        
        # Draw guidance text
        guidance_color = self.COLOR_GREEN if self.position_status == "Valid" else self.COLOR_RED
        cv2.putText(image, f"Guidance: {self.guidance_text}", (10, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, guidance_color, 2)
        
        # If not calibrated, show calibration progress
        if not self.is_calibrated and self.calibration_counter > 0:
            progress = (self.calibration_counter / self.calibration_frames) * 100
            progress_text = f"Calibration: {progress:.1f}%"
            cv2.putText(image, progress_text, (10, 90), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.COLOR_YELLOW, 2)
        
        # If calibrated, draw reference position
        if self.is_calibrated:
            # Draw reference position marker
            cv2.circle(image, self.reference_position, 10, self.COLOR_BLUE, 2)
            
            # Draw reference box
            ref_width, ref_height = self.reference_size
            ref_x, ref_y = self.reference_position
            ref_x_min = ref_x - ref_width // 2
            ref_y_min = ref_y - ref_height // 2
            ref_x_max = ref_x + ref_width // 2
            ref_y_max = ref_y + ref_height // 2
            
            # Draw reference box with dashed lines
            dash_length = 10
            dash_gap = 5
            
            # Draw horizontal dashed lines
            for x in range(ref_x_min, ref_x_max, dash_length + dash_gap):
                x2 = min(x + dash_length, ref_x_max)
                cv2.line(image, (x, ref_y_min), (x2, ref_y_min), self.COLOR_BLUE, 2)
                cv2.line(image, (x, ref_y_max), (x2, ref_y_max), self.COLOR_BLUE, 2)
            
            # Draw vertical dashed lines
            for y in range(ref_y_min, ref_y_max, dash_length + dash_gap):
                y2 = min(y + dash_length, ref_y_max)
                cv2.line(image, (ref_x_min, y), (ref_x_min, y2), self.COLOR_BLUE, 2)
                cv2.line(image, (ref_x_max, y), (ref_x_max, y2), self.COLOR_BLUE, 2)
            
            # If face is detected, check position and draw guidance
            if face_data.detected:
                position_valid, deviation = self.check_position(face_data)
                
                # Draw current position
                current_center = face_data.center
                
                # Choose color based on position validity
                position_color = self.COLOR_GREEN if position_valid else self.COLOR_RED
                
                # Draw current position marker
                cv2.circle(image, current_center, 10, position_color, -1)
                
                # Draw line from reference to current position
                cv2.line(image, self.reference_position, current_center, position_color, 2)
                
                # Draw deviation values
                deviation_text = f"X: {deviation['x']:.1f}, Y: {deviation['y']:.1f}, Size: {deviation['size']*100:.1f}%"
                cv2.putText(image, deviation_text, (10, 120), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, position_color, 2)
        
        return image
    
    def process_frame(self, image, face_data, calibrate=False):
        """
        Process a frame to provide virtual chinrest guidance.
        
        Args:
            image: Input image
            face_data: Face data
            calibrate: Whether to calibrate the chinrest
            
        Returns:
            processed_image: Image with guidance visualization
            position_valid: Whether the position is valid
        """
        position_valid = False
        
        # If calibration is requested and face is detected
        if calibrate and face_data.detected:
            self.calibrate(face_data)
        
        # If already calibrated, check position
        if self.is_calibrated and face_data.detected:
            position_valid, _ = self.check_position(face_data)
        
        # Draw guidance visualization
        processed_image = self.draw_guidance(image, face_data)
        
        return processed_image, position_valid

