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

        compute = upper_base_line.compute_intersection_point(left_side_line)
        if compute[0]:
            self.court_points.append(compute[1].copy())  # P1

        compute = lower_base_line.compute_intersection_point(left_side_line)
        if compute[0]:
            self.court_points.append(compute[1].copy())  # P2

        compute = lower_base_line.compute_intersection_point(right_side_line)
        if compute[0]:
            self.court_points.append(compute[1].copy())  # P3

        compute = upper_base_line.compute_intersection_point(right_side_line)
        if compute[0]:
            self.court_points.append(compute[1].copy())  # P4

        compute = upper_base_line.compute_intersection_point(left_singles_line)
        if compute[0]:
            self.court_points.append(compute[1].copy())  # P5

        compute = lower_base_line.compute_intersection_point(left_singles_line)
        if compute[0]:
            self.court_points.append(compute[1].copy())  # P6

        compute = lower_base_line.compute_intersection_point(right_singles_line)
        if compute[0]:
            self.court_points.append(compute[1].copy())  # P7

        compute = upper_base_line.compute_intersection_point(right_singles_line)
        if compute[0]:
            self.court_points.append(compute[1].copy())  # P8

        compute = left_singles_line.compute_intersection_point(upper_service_line)
        if compute[0]:
            self.court_points.append(compute[1].copy())  # P9

        compute = right_singles_line.compute_intersection_point(upper_service_line)
        if compute[0]:
            self.court_points.append(compute[1].copy())  # P10

        compute = left_singles_line.compute_intersection_point(lower_service_line)
        if compute[0]:
            self.court_points.append(compute[1].copy())  # P11

        compute = right_singles_line.compute_intersection_point(lower_service_line)
        if compute[0]:
            self.court_points.append(compute[1].copy())  # P12

        compute = upper_service_line.compute_intersection_point(centre_service_line)
        if compute[0]:
            self.court_points.append(compute[1].copy())  # P13

        compute = lower_service_line.compute_intersection_point(centre_service_line)
        if compute[0]:
            self.court_points.append(compute[1].copy())  # P14

        compute = left_side_line.compute_intersection_point(net_line)
        if compute[0]:
            self.court_points.append(compute[1].copy())  # P15

        compute = right_side_line.compute_intersection_point(net_line)
        if compute[0]:
            self.court_points.append(compute[1].copy())  # P16

        assert len(self.court_points) == 16

    @staticmethod
    def get_possible_line_pairs(lines):
        line_pairs = []
        for first in range(len(lines)):
            for second in range(first + 1, len(lines)):
                line_pairs.append((lines[first], lines[second]))
        return line_pairs

    def get_intersection_points(self, hLinePair, vLinePair):
        v = []
        point = np.array([0.0, 0.0], dtype=np.float32)

        if hLinePair.first.compute_intersection_point(vLinePair.first, point):
            v.append(point)

        if hLinePair.first.compute_intersection_point(vLinePair.second, point):
            v.append(point)

        if hLinePair.second.compute_intersection_point(vLinePair.first, point):
            v.append(point)

        if hLinePair.second.compute_intersection_point(vLinePair.second, point):
            v.append(point)

        assert len(v) == 4

        return v
