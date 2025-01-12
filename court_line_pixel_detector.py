import cv2
import numpy as np

class CourtLinePixelDetector:
    class Parameters:
        def __init__(self):
            self.threshold = 0
            self.diffThreshold = 0
            self.t = 0
            self.gradientKernelSize = 0
            self.kernelSize = 0

    debug = False
    windowName = "Court Line Pixel Detector"

    def __init__(self, parameters=None):
        if parameters is None:
            parameters = self.Parameters()
        self.parameters = parameters

    def run(self, frame):
        luminance_channel = self.get_luminance_channel(frame)
        line_pixels = self.detect_line_pixels(luminance_channel)
        filtered_pixels = self.filter_line_pixels(line_pixels, luminance_channel)
        return filtered_pixels

    def get_luminance_channel(self, frame):
        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    def detect_line_pixels(self, image):
        _, binary_image = cv2.threshold(image, self.parameters.threshold, 255, cv2.THRESH_BINARY)
        return binary_image

    def filter_line_pixels(self, binary_image, luminance_image):
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (self.parameters.kernelSize, self.parameters.kernelSize))
        filtered_image = cv2.morphologyEx(binary_image, cv2.MORPH_CLOSE, kernel)
        return filtered_image

    def is_line_pixel(self, image, x, y):
        return image[y, x] > self.parameters.threshold

    def compute_structure_tensor_elements(self, image):
        dx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=self.parameters.gradientKernelSize)
        dy = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=self.parameters.gradientKernelSize)
        dx2 = dx * dx
        dxy = dx * dy
        dy2 = dy * dy
        return dx2, dxy, dy2
