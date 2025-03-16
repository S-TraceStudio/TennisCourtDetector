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
    debug_sort_line = False
    windowName = "TennisCourtFitter"
    hLinePairs = []
    vLinePairs = []
    best_model = TennisCourtModel();

    def __init__(self):
        debug = True

    def run(self, lines, binary_image, rgb_image):
        hLines, vLines = self.get_horizontal_and_vertical_lines(lines, rgb_image)

        self.sort_horizontal_lines(hLines, rgb_image)
        self.sort_vertical_lines(vLines, rgb_image)
        print("Horizontal lines")
        print(hLines)
        print("Vertical lines")
        print(vLines)

        self.hLinePairs.clear()
        self.vLinePairs.clear()
        self.hLinePairs = TennisCourtModel.get_possible_line_pairs(hLines);
        self.vLinePairs = TennisCourtModel.get_possible_line_pairs(vLines);

        if self.debug:
            print(f"Horizontal line pairs = {len(self.hLinePairs)}")
            print(f"Vertical line pairs = {len(self.vLinePairs)}")
            print()
            print("Horizontal line pairs")
            print(self.hLinePairs)
            print("Vertical line pairs")
            print(self.vLinePairs)

        self.find_best_model_fit(binary_image, rgb_image)

        return self.best_model


    def get_horizontal_and_vertical_lines(self, lines, rgbImage):
        hLines = []
        vLines = []
        for line in lines:
            if line.is_vertical():
                vLines.append(line)
            else:
                hLines.append(line)

        if self.debug:
            print()
            print(f"Total lines = {len(lines)}")
            print(f"Horizontal lines = {len(hLines)}")
            print(f"Vertical lines = {len(vLines)}")
            print()

            image = rgbImage.copy()
            draw_lines(hLines, image, color=(255, 0, 0))
            draw_lines(vLines, image, color=(0, 255, 0))
            displayDebugImage(image, widow_name=self.windowName )

        return hLines, vLines

    def sort_horizontal_lines(self, hLines, rgb_image):
        x = rgb_image.shape[1] / 2.0
        #print(f"Before: {hLines}")
        sort_lines_by_distance_to_point(hLines, (x, 0))
        # print(f"After: {hLines}")

        if self.debug_sort_line:
            for line in hLines:
                image = rgb_image.copy()
                draw_line(line, image, color=(255, 0, 0))
                displayDebugImage(image, widow_name=self.windowName)

    def sort_vertical_lines(self, vLines, rgb_image):
        y = rgb_image.shape[0] / 2.0
        sort_lines_by_distance_to_point(vLines, (0, y))

        if self.debug_sort_line:
            for line in vLines:
                image = rgb_image.copy()
                draw_line(line, image, color=(0, 255, 0))
                displayDebugImage(image, widow_name=self.windowName)

    def find_best_model_fit(self, binary_image, rgb_image):
        count = 0
        total_count = len(self.hLinePairs)*len(self.vLinePairs)

        best_score = global_params.initial_fit_score

        for h_line_pair in self.hLinePairs:
            for v_line_pair in self.vLinePairs:
                count += 1

                model = TennisCourtModel()
                score = model.fit(h_line_pair, v_line_pair, binary_image, rgb_image)
                if score > best_score:
                    best_score = score
                    self.best_model = model

                    if self.debug:
                        print(f"Best score: {best_score}")
                        print(model.transformation_matrix)

            if self.debug:
                percentage = float(count) / total_count
                print(f"percentage: {percentage} %")
                print("-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------")

        if self.debug:
            print(f"Best model score = {best_score}")
            image_copy = rgb_image.copy()
            self.best_model.draw_model(image = image_copy, color=(0, 255, 255))
            displayDebugImage(debug_image = image_copy, widow_name=self.windowName )


                









