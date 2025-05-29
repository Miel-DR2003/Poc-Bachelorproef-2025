import pyautogui
from collections import deque
import numpy as np
import time
import math

class MouseControl:
    def __init__(self):
        # Get screen size
        self.screen_width, self.screen_height = pyautogui.size()
        # Flag to enable/disable mouse control
        self.mouse_control = False
        # Smoothing
        self.sensitivity = 0.5
        self.smoothing = 0.5
        # Position history for smoothing
        self.position_history = deque(maxlen=5)


    def moveMouse(self, gaze_x, gaze_y):
        """Move the mouse using calibrated mapping.
        
        Args:
            x: Horizontal gaze coordinate (0-1)
            y: Vertical gaze coordinate (0-1)
        """
        # Apply calibration
        x, y = self.apply_callibration(gaze_x, gaze_y)
        
        # Convert to screen coordinates
        screen_x = int(x * self.screen_width)
        screen_y = int(y * self.screen_height)

        # Add to position history for smoothing
        self.position_history.append((screen_x, screen_y))

        # Apply smoothing by averaging recent positions
        if len(self.position_history) > 1:
            smoothed_x = sum(pos[0] for pos in self.position_history) / len(self.position_history)
            smoothed_y = sum(pos[1] for pos in self.position_history) / len(self.position_history)
            screen_x, screen_y = int(smoothed_x), int(smoothed_y)
        
        print(f"Moving mouse to: x={screen_x} y={screen_y}")
        
        pyautogui.FAILSAFE = False
        # Relative positioning (original behavior) - gradually move toward target
        current_x, current_y = pyautogui.position()
        target_x = current_x + int((screen_x - current_x) * self.sensitivity)
        target_y = current_y + int((screen_y - current_y) * self.sensitivity)
        pyautogui.moveTo(target_x, target_y)


    def performClick(self):
        print("Click performed on the coördinates: ", pyautogui.position())
        pyautogui.click()

    def toggleMouseControl(self):
        self.mouse_control = not self.mouse_control
    
    def move_mouse_2(self, direction):
        #print(f"Moving mouse in direction: {direction}")
        match direction:
            case "left_top":
                pyautogui.moveRel(-10, -10)
            case "left_bottom":
                pyautogui.moveRel(-10, 10)
            case "left_center":
                pyautogui.moveRel(-10, 0)
            case "right_top":
                pyautogui.moveRel(10, -10)
            case "right_bottom":
                pyautogui.moveRel(10, 10)
            case "right_center":
                pyautogui.moveRel(10, 0)
            case "centre_top":
                pyautogui.moveRel(0, -10)
            case "centre_bottom":
                pyautogui.moveRel(0, 10)
            case "centre_centre":
                pyautogui.moveRel(0, 0)

