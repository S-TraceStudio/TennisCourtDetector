import cv2
import numpy as np
from line import Line
from tennis_court_model import TennisCourtModel
from utils import displayDebugImage
from debug_helper import draw_lines

class TennisCourtFitter:
    debug = True
    windowName = "TennisCourtFitter"

    def __init__(self):
        debug = True

    def run(self, lines, binaryImage, rgbImage):
        hLines, vLines = self.get_horizontal_and_vertical_lines(lines, rgbImage)

        #self.sortHorizontalLines(hLines, rgbImage)
        #self.sortVerticalLines(vLines, rgbImage)




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











