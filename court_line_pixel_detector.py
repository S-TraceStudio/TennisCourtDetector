import cv2
import numpy as np
from utils import displayDebugImage
from global_paramaters import global_params

class CourtLinePixelDetector:
    class Parameters:
        def __init__(self):
            self.threshold = 80
            self.diffThreshold = 20
            self.t = 4
            self.gradientKernelSize = 3
            self.kernelSize = 41

    debug = False
    windowName = "Court Line Pixel Detector"

    def __init__(self, parameters=None):
        if parameters is None:
            parameters = self.Parameters()
        self.parameters = parameters

    def run(self, frame, debug=False):
        luminance_channel = self.get_luminance_channel(frame,debug)
        if debug:
            displayDebugImage(luminance_channel)

        line_pixels = self.detect_line_pixels(luminance_channel)
        if debug:
            displayDebugImage(line_pixels)

        filtered_pixels = self.filter_line_pixels(line_pixels, luminance_channel)
        if debug:
            displayDebugImage(filtered_pixels)

        return filtered_pixels

    def get_luminance_channel(self, frame, debug=False):
        # Convert RGB to YCrCb
        imgYCbCr = cv2.cvtColor(frame, cv2.COLOR_RGB2YCrCb)

        # Extract the luminance channel
        luminanceChannel = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)
        from_to = [0, 0]
        cv2.mixChannels([frame], [luminanceChannel], from_to)

        if debug:
            displayDebugImage(frame)
            displayDebugImage(imgYCbCr)
            displayDebugImage(luminanceChannel)

        return luminanceChannel

    def detect_line_pixels(self, image, debug=False):
        # Assuming this class is defined elsewhere
        bg_value = global_params.bg_value
        pixel_image = np.full((image.shape[0], image.shape[1]), bg_value, dtype=np.uint8)

        # Fill the pixel image based on the isLinePixel function
        for x in range(image.shape[1]):
            for y in range(image.shape[0]):
                pixel_image[y, x] = self.is_line_pixel(image, x, y)

        if debug:
            displayDebugImage(pixel_image)

        return pixel_image

    def filter_line_pixels(self, binary_image, luminance_image):
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (self.parameters.kernelSize, self.parameters.kernelSize))
        filtered_image = cv2.morphologyEx(binary_image, cv2.MORPH_CLOSE, kernel)
        return filtered_image

    def is_line_pixel(self, image, x, y):
        if x < self.parameters.t or image.shape[1] - x <= self.parameters.t:
            return global_params.bg_value
        if y < self.parameters.t or image.shape[0] - y <= self.parameters.t:
            return global_params.bg_value

        value = image[y, x]

        if value < self.parameters.threshold:
            return global_params.bg_value

        # Use np.clip to constrain the values
        max_value = np.iinfo(value.dtype).max
        min_value = np.iinfo(value.dtype).min

        top_value = image[y - self.parameters.t, x]
        lower_value = image[y + self.parameters.t, x]
        left_value = image[y, x - self.parameters.t]
        right_value = image[y, x + self.parameters.t]

        value = np.clip(value, min_value, max_value)
        left_value = np.clip(left_value, min_value, max_value)
        right_value = np.clip(right_value, min_value, max_value)
        top_value = np.clip(top_value, min_value, max_value)
        lower_value = np.clip(lower_value, min_value, max_value)

        if (value - left_value > self.parameters.diffThreshold) and (value - right_value > self.parameters.diffThreshold):
            return global_params.fg_value

        if (value - top_value > self.parameters.diffThreshold) and (value - lower_value > self.parameters.diffThreshold):
            return global_params.fg_value
        return global_params.bg_value;



    def compute_structure_tensor_elements(self, image):
        dx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=self.parameters.gradientKernelSize)
        dy = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=self.parameters.gradientKernelSize)
        dx2 = dx * dx
        dxy = dx * dy
        dy2 = dy * dy
        return dx2, dxy, dy2
