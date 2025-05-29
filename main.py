import cv2
import time

from gaze_tracking.util.camera import Camera
from gaze_tracking.eye_tracking_system import EyeTrackingSystem
from gaze_tracking.object_detection.gaze_detection import GazeEstimation



calibrate_chinrest = False
# Initialize camera
camera = Camera()
# Initialize the eye tracking system
system = EyeTrackingSystem()
while True:
    # Capture frame
    frame = camera.capture_frame()

    # Process frame ; calibrate_chinrest flag gets reset after each frame
    processed_frame, tracking_data = system.process_frame(frame, calibrate_chinrest)

    # Reset calibration flag after one frame
    calibrate_chinrest = False

    # Display the processed frame
    cv2.imshow("Eye Tracking System", processed_frame)

    # Control panel
    key = cv2.waitKey(1)
    if key == ord('q'):  # Quit
        break
    elif key == ord('m'):  # Toggle mouse control with 'm' key
        system.toggle_mouse_control()
        time.sleep(0.2)  # Small delay to avoid multiple toggles
    elif key == ord('c'):  # Calibrate chinrest
        print("Calibrating chinrest...")
        calibrate_chinrest = True
    elif key == ord('e'):  # Toggle eye detection visibility
        system.toggle_eyes_visibility()
    elif key == ord('t'):  # Toggle face mesh visibility
        system.toggle_mesh_visibility()


# Release resources
camera.close()
cv2.destroyAllWindows()





