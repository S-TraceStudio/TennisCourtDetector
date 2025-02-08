import cv2
import numpy as np
from line import Line

LinePair = tuple[cv2.line, cv2.line]


class TennisCourtModel:
    h_lines = []
    v_lines = []
    h_line_pairs = []
    v_line_pairs = []
    court_points = []

    def __init__(self):
        h_vector = np.array([1.0, 0.0], dtype=np.float32)
        upper_base_line = Line(np.array([0.0, 0.0], dtype=np.float32), h_vector)
        upper_service_line = Line(np.array([0.0, 5.49], dtype=np.float32), h_vector)
        net_line = Line(np.array([0.0, 11.89], dtype=np.float32), h_vector)
        lower_service_line = Line(np.array([0.0, 18.29], dtype=np.float32), h_vector)
        lower_base_line = Line(np.array([0.0, 23.78], dtype=np.float32), h_vector)
        self.h_lines = [upper_base_line, upper_service_line, net_line, lower_service_line, lower_base_line]

        v_vector = np.array([0.0, 1.0], dtype=np.float32)
        left_side_line = Line(np.array([0.0, 0.0], dtype=np.float32), v_vector)
        left_singles_line = Line(np.array([1.37, 0.0], dtype=np.float32), v_vector)
        centre_service_line = Line(np.array([5.485, 0.0], dtype=np.float32), v_vector)
        right_singles_line = Line(np.array([9.6, 0.0], dtype=np.float32), v_vector)
        right_side_line = Line(np.array([10.97, 0.0], dtype=np.float32), v_vector)
        self.v_lines = [left_side_line, left_singles_line, centre_service_line, right_singles_line, right_side_line]

        # Add pairs to h_line_pairs
        self.h_line_pairs.append((self.h_lines[0], self.h_lines[4]))
        self.h_line_pairs.append((self.h_lines[0], self.h_lines[3]))
        self.h_line_pairs.append((self.h_lines[1], self.h_lines[3]))
        self.h_line_pairs.append((self.h_lines[1], self.h_lines[4]))

        # Add pairs to v_line_pairs
        self.v_line_pairs.append((self.v_lines[0], self.v_lines[4]))
        self.v_line_pairs.append((self.v_lines[0], self.v_lines[3]))
        self.v_line_pairs.append((self.v_lines[1], self.v_lines[4]))
        self.v_line_pairs.append((self.v_lines[1], self.v_lines[3]))

        point = np.array([0.0, 0.0], dtype=np.float32)

        if upper_base_line.compute_intersection_point(left_side_line, point):
            self.court_points.append(point.copy())  # P1

        if lower_base_line.compute_intersection_point(left_side_line, point):
            self.court_points.append(point.copy())  # P2

        if lower_base_line.compute_intersection_point(right_side_line, point):
            self.court_points.append(point.copy())  # P3

        if upper_base_line.compute_intersection_point(right_side_line, point):
            self.court_points.append(point.copy())  # P4

        if upper_base_line.compute_intersection_point(left_singles_line, point):
            self.court_points.append(point.copy())  # P5

        if lower_base_line.compute_intersection_point(left_singles_line, point):
            self.court_points.append(point.copy())  # P6

        if lower_base_line.compute_intersection_point(right_singles_line, point):
            self.court_points.append(point.copy())  # P7

        if upper_base_line.compute_intersection_point(right_singles_line, point):
            self.court_points.append(point.copy())  # P8

        if left_singles_line.compute_intersection_point(upper_service_line, point):
            self.court_points.append(point.copy())  # P9

        if right_singles_line.compute_intersection_point(upper_service_line, point):
            self.court_points.append(point.copy())  # P10

        if left_singles_line.compute_intersection_point(lower_service_line, point):
            self.court_points.append(point.copy())  # P11

        if right_singles_line.compute_intersection_point(lower_service_line, point):
            self.court_points.append(point.copy())  # P12

        if upper_service_line.compute_intersection_point(centre_service_line, point):
            self.court_points.append(point.copy())  # P13

        if lower_service_line.compute_intersection_point(centre_service_line, point):
            self.court_points.append(point.copy())  # P14

        if left_side_line.compute_intersection_point(net_line, point):
            self.court_points.append(point.copy())  # P15

        if right_side_line.compute_intersection_point(net_line, point):
            self.court_points.append(point.copy())  # P16

        assert len(self.court_points) == 16

    @staticmethod
    def get_possible_line_pairs(lines):
        line_pairs = []
        for first in range(len(lines)):
            for second in range(first + 1, len(lines)):
                line_pairs.append((lines[first], lines[second]))
        return line_pairs
