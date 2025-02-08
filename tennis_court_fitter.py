import cv2
import numpy as np
from line import Line
from tennis_court_model import TennisCourtModel
from utils import displayDebugImage
from debug_helper import draw_lines, draw_line
from geometry import sort_lines_by_distance_to_point

class TennisCourtFitter:
    debug = True
    windowName = "TennisCourtFitter"

    def __init__(self):
        debug = True

    def run(self, lines, binaryImage, rgbImage):
        hLines, vLines = self.get_horizontal_and_vertical_lines(lines, rgbImage)

        self.sort_horizontal_lines(hLines, rgbImage)
        self.sort_vertical_lines(vLines, rgbImage)




    def get_horizontal_and_vertical_lines(self, lines, rgbImage):
        hLines = []
        vLines = []
        for line in lines:
            if line.is_vertical():
                vLines.append(line)
            else:
                hLines.append(line)

        if self.debug:
            print(f"Horizontal lines = {len(hLines)}")
            print(f"Vertical lines = {len(vLines)}")
            image = rgbImage.copy()
            draw_lines(hLines, image, color=(255, 0, 0))
            draw_lines(vLines, image, color=(0, 255, 0))
            displayDebugImage(image, widow_name=self.windowName )

        return hLines, vLines

    def sort_horizontal_lines(self, hLines, rgb_image):
        x = rgb_image.shape[1] / 2.0
        sort_lines_by_distance_to_point(hLines, (x, 0))

        if self.debug:
            for line in hLines:
                image = rgb_image.copy()
                draw_line(line, image, color=(255, 0, 0))
                displayDebugImage(image, widow_name=self.windowName)

    def sort_vertical_lines(self, vLines, rgb_image):
        y = rgb_image.shape[2] / 2.0
        sort_lines_by_distance_to_point(vLines, (0, y))

        if self.debug:
            for line in vLines:
                image = rgb_image.copy()
                draw_line(line, image, color=(0, 255, 0))
                displayDebugImage(image, widow_name=self.windowName)








