import cv2
import numpy as np
from line import Line

LinePair = tuple[cv2.line, cv2.line]


class TennisCourtModel:
    def __init__(self):
        h_vector = np.array([1.0, 0.0], dtype=np.float32)
        upper_base_line = Line(np.array([0.0, 0.0], dtype=np.float32), h_vector)
        upper_service_line = Line(np.array([0.0, 5.49], dtype=np.float32), h_vector)
        net_line = Line(np.array([0.0, 11.89], dtype=np.float32), h_vector)
        lower_service_line = Line(np.array([0.0, 18.29], dtype=np.float32), h_vector)
        lower_base_line = Line(np.array([0.0, 23.78], dtype=np.float32), h_vector)
        h_lines = [upper_base_line, upper_service_line, net_line, lower_service_line, lower_base_line]

        v_vector = np.array([0.0, 1.0], dtype=np.float32)
        left_side_line = Line(np.array([0.0, 0.0], dtype=np.float32), v_vector)
        left_singles_line = Line(np.array([1.37, 0.0], dtype=np.float32), v_vector)
        centre_service_line = Line(np.array([5.485, 0.0], dtype=np.float32), v_vector)
        right_singles_line = Line(np.array([9.6, 0.0], dtype=np.float32), v_vector)
        right_side_line = Line(np.array([10.97, 0.0], dtype=np.float32), v_vector)
        v_lines = [left_side_line, left_singles_line, centre_service_line, right_singles_line, right_side_line]







    @staticmethod
    def get_possible_line_pairs(lines):
        line_pairs = []
        for first in range(len(lines)):
            for second in range(first + 1, len(lines)):
                line_pairs.append((lines[first], lines[second]))
        return line_pairs
