import cv2
import numpy as np
from line import Line
from tennis_court_model import TennisCourtModel
from utils import displayDebugImage
from debug_helper import draw_lines, draw_line
from geometry import sort_lines_by_distance_to_point
from global_paramaters import global_params

class TennisCourtFitter:
    debug = True
    windowName = "TennisCourtFitter"
    hLinePairs = []
    vLinePairs = []

    def __init__(self):
        debug = True

    def run(self, lines, binary_image, rgb_image):
        hLines, vLines = self.get_horizontal_and_vertical_lines(lines, rgb_image)

        self.sort_horizontal_lines(hLines, rgb_image)
        self.sort_vertical_lines(vLines, rgb_image)

        self.hLinePairs = TennisCourtModel.get_possible_line_pairs(hLines);
        self.vLinePairs = TennisCourtModel.get_possible_line_pairs(vLines);

        if self.debug:
            print(f"Horizontal line pairs = {len(self.hLinePairs)}")
            print(f"Vertical line pairs = {len(self.vLinePairs)}")

        self.find_best_model_fit(binary_image, rgb_image)




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

    def find_best_model_fit(self, binary_image, rgb_image):
        count = 0
        total_count = len(self.hLinePairs)+len(self.vLinePairs)

        bestScore = global_params.initial_fit_score

        for h_line_pair in self.hLinePairs:
            for v_line_pair in self.vLinePairs:
                count += 1

                model = TennisCourtModel()
                score = model.fit(h_line_pair, v_line_pair, binary_image, rgb_image)
                if score > bestScore:
                    bestScore = score
                    bestModel = model

                    print(f"Best score: {bestScore}")

            percentage = float(count) / total_count
            print(f"percentage: {percentage} %")

        if self.debug:
            print(f"Best model score = {bestScore}")
            image = rgb_image.copy()
            bestModel.drawModel(image)
            displayDebugImage(image, widow_name=self.windowName )


                









