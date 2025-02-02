import cv2
import numpy as np
from line import Line

LinePair = tuple[cv2.line, cv2.line]

class TennisCourtModel:
    def __init__(self):
        hVector = np.array([1.0, 0.0], dtype=np.float32)






    @staticmethod
    def get_possible_line_pairs(lines):
        line_pairs = []
        for first in range(len(lines)):
            for second in range(first + 1, len(lines)):
                line_pairs.append((lines[first], lines[second]))
        return line_pairs