import cv2

class LightingHandler:
    """
    Lighting condition handling module for eye tracking system.
    This class provides methods to enhance eye detection in various lighting conditions.
    """
    
    def __init__(self):
        """
        Initialize the lighting handler.
        """
        # Default parameters
        self.clahe_clip_limit = 2.0
        self.clahe_grid_size = (8, 8)
        
        # https://en.wikipedia.org/wiki/Histogram_equalization
        # Initialize CLAHE (Contrast Limited Adaptive Histogram Equalization)
        self.clahe = cv2.createCLAHE(
            clipLimit=self.clahe_clip_limit,
            tileGridSize=self.clahe_grid_size
        )
    
    # https://docs.opencv.org/4.x/d5/daf/tutorial_py_histogram_equalization.html
    # https://www.geeksforgeeks.org/clahe-histogram-eqalization-opencv/
    # Geraadpleegd op 22/04/2025
    def apply_clahe(self, image):
        """
        Enhance image for better eye detection in different lighting conditions.
        
        Args:
            image: Input image (BGR format)
            
        Returns:
            enhanced_image: Enhanced image by applying clache in BGR format
        """
        # Convert to grayscale if not already
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Apply CLAHE for contrast enhancement
        clahe_image = self.clahe.apply(gray)

        # Convert back to BGR format to be consistent with opencv default format
        clahe_image = cv2.cvtColor(clahe_image, cv2.COLOR_GRAY2BGR)
        
        return clahe_image
