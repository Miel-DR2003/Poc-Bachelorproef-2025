# def look at https://medium.com/@amit.aflalo2/eye-gaze-estimation-using-a-webcam-in-100-lines-of-code-570d4683fe23
# other inspo: https://github.com/LaithAlShimaysawee/GazeOrientation
import cv2 
import numpy as np

class GazeEstimation():
    def __init__(self, calibration=None):
        self.gaze_dir = None
        self.calibration = calibration
        # Default thresholds (will be overridden by calibration)
        self.hori_gaze_range = [0.45, 0.55]
        self.vert_gaze_range = [0.35, 0.65]


    def process_frame(self, frame, face_data, left_eye, right_eye):
        gaze_dir = self.estimate_gaze_dir(frame, face_data, left_eye, right_eye)
            
        processed_frame = self.plot_gaze_direction(frame)
        return processed_frame, gaze_dir

    def estimate_gaze_dir(self, frame, face_data, left_eye, right_eye):
        # Rest of the method remains the same with the dynamic thresholds
        left_horizontal_ratio = left_eye.horizontal_ratio
        left_vertical_ratio = left_eye.vertical_ratio
        right_horizontal_ratio = right_eye.horizontal_ratio
        right_vertical_ratio = right_eye.vertical_ratio
        
        horizontal_ratio = np.mean([left_horizontal_ratio, right_horizontal_ratio])
        vertical_ratio = np.mean([left_vertical_ratio, right_vertical_ratio])
        
        # For debugging
        # if not self.calibration.is_calibrated:
        #     self.calibration.calibrate(frame, face_data, left_eye, right_eye)
        #     print(f"H-ratio: {horizontal_ratio:.2f}, V-ratio: {vertical_ratio:.2f}")
        # if self.calibration.calibration_data:
        #     self.hori_gaze_range = [np.min(self.calibration.calibration_data["left"]), np.max(self.calibration.calibration_data["right"])]
        #     self.vert_gaze_range = [np.min(self.calibration.calibration_data["top"]), np.max(self.calibration.calibration_data["bottom"])]

        
        if horizontal_ratio <= self.hori_gaze_range[0]:
            gaze_dir = "left"
        elif horizontal_ratio >= self.hori_gaze_range[1]:
            gaze_dir = "right"
        else:
            gaze_dir = "center"
            
        if vertical_ratio <= self.vert_gaze_range[0]:
            gaze_dir += "_top"
        elif vertical_ratio >= self.vert_gaze_range[1]:
            gaze_dir += "_bottom"
        else:
            gaze_dir += "_center"
            
        self.gaze_dir = gaze_dir
        return gaze_dir
    
    def plot_gaze_direction(self, frame):
        # Draw the gaze direction on the frame
        cv2.putText(frame, f"Gaze Direction: {self.gaze_dir}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        return frame
    


