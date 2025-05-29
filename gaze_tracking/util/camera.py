import cv2

class Camera:
    def __init__(self, camera_id=0):
        self.camera_id = camera_id
        self.camera = cv2.VideoCapture(self.camera_id)
        if not self.camera.isOpened():
            raise ValueError("Unable to open camera with ID: {}".format(self.camera_id))

    def capture_frame(self):
        # Capture frame
        _, frame = self.camera.read()
        if not _:
            print("Error: Failed to capture frame.")
        frame = cv2.flip(frame, 1)
        return frame
    
    def close(self):
        self.camera.release()
