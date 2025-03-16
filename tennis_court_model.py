import cv2
import numpy as np
from line import Line
from global_paramaters import global_params
from geometry import distance, normalize
from debug_helper import draw_lines, draw_line, draw_line_segment

LinePair = tuple[cv2.line, cv2.line]

class TennisCourtModel:
    h_lines = []
    v_lines = []
    h_line_pairs = []
    v_line_pairs = []
    court_points = []
    transformation_matrix = np.zeros((3, 3), dtype=np.float32)

    def __init__(self):
        self.court_points.clear()
        self.h_line_pairs.clear()
        self.v_line_pairs.clear()
        self.h_lines.clear()
        self.v_lines.clear()
        self.transformation_matrix = np.zeros((3, 3), dtype=np.float32)
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

        # print(f"Total points = {len(self.court_points)}")
        # print (self.court_points)
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

        compute = hLinePair[0].compute_intersection_point(vLinePair[0])
        if compute[0]:
            v.append(compute[1].copy())

        compute = hLinePair[0].compute_intersection_point(vLinePair[1])
        if compute[0]:
            v.append(compute[1].copy())

        compute = hLinePair[1].compute_intersection_point(vLinePair[0])
        if compute[0]:
            v.append(compute[1].copy())

        compute = hLinePair[1].compute_intersection_point(vLinePair[1])
        if compute[0]:
            v.append(compute[1].copy())

        assert len(v) == 4
        return v

    def fit(self, h_line_pair, v_line_pair, binary_image, rgb_image):
        best_score = global_params.initial_fit_score
        points = self.get_intersection_points(h_line_pair, v_line_pair)

        print()
        print("Horizontal Line Pair")
        print(h_line_pair)
        print("Vertical Line Pair")
        print(v_line_pair)
        print()
        print("Points")
        print(points)

        # print(f"Fit horizontal line pairs = {len(self.h_line_pairs)}")
        # print(f"Fit vertical line pairs = {len(self.v_line_pairs)}")

        for model_h_line_pair in self.h_line_pairs:
            for model_v_line_pair in self.v_line_pairs:
                model_points = self.get_intersection_points(model_h_line_pair, model_v_line_pair)
                matrix = cv2.getPerspectiveTransform(np.array(model_points, dtype=np.float32), np.array(points, dtype=np.float32))
                transformed_model_points = np.zeros((16, 2), dtype=np.float32)
                transformed_model_points = cv2.perspectiveTransform(np.array([self.court_points], dtype=np.float32), matrix)[0]

                #print(transformed_model_points)

                score = self.evaluate_model(transformed_model_points, binary_image)
                if score > best_score:
                    best_score = score
                    self.transformation_matrix = matrix

        return best_score

    def compute_score_for_line_segment(self, start, end, binary_image):
        score = 0
        fg_score = 1
        bg_score = -0.5
        length = int(round(distance(start, end)))

        vec = normalize(np.array(end) - np.array(start))

        for i in range(length):
            p = np.array(start) + np.multiply(vec, i)
            x = int(round(p[0]))
            y = int(round(p[1]))
            if self.is_inside_the_image(x, y, binary_image):
                image_value = binary_image[y, x]
                if image_value == global_params.fg_value:
                    score += fg_score
                else:
                    score += bg_score
        return score

    def is_inside_the_image(self, x, y, image):
        return (0 <= x < image.shape[1]) and (0 <= y < image.shape[0])

    def evaluate_model(self, court_points, binary_image):
        score = 0

        # TODO: heuristic to see whether the model makes sense
        d1 = distance(court_points[0], court_points[1])
        d2 = distance(court_points[1], court_points[2])
        d3 = distance(court_points[2], court_points[3])
        d4 = distance(court_points[3], court_points[0])
        t = 30
        if d1 < t or d2 < t or d3 < t or d4 < t:
            return global_params.initial_fit_score  # Replace with your initial fit score value

        score += self.compute_score_for_line_segment(court_points[0], court_points[1], binary_image)
        score += self.compute_score_for_line_segment(court_points[1], court_points[2], binary_image)
        score += self.compute_score_for_line_segment(court_points[2], court_points[3], binary_image)
        score += self.compute_score_for_line_segment(court_points[3], court_points[0], binary_image)
        score += self.compute_score_for_line_segment(court_points[4], court_points[5], binary_image)
        score += self.compute_score_for_line_segment(court_points[6], court_points[7], binary_image)
        score += self.compute_score_for_line_segment(court_points[8], court_points[9], binary_image)
        score += self.compute_score_for_line_segment(court_points[10], court_points[11], binary_image)
        score += self.compute_score_for_line_segment(court_points[12], court_points[13], binary_image)
        #  score += self.compute_score_for_line_segment(court_points[14], court_points[14], binary_image)

        #  print("Score =", score)

        return score

    def draw_model(self, image, color = (0, 255, 255)):
        print("draw model 1")
        # print(self.transformation_matrix)
        transformed_model_points = cv2.perspectiveTransform(np.array([self.court_points], dtype='float32'), self.transformation_matrix)[0]
        print(self.transformation_matrix)
        self.draw_model_points(transformed_model_points, image, color)

    def draw_model_points(self, transformed_points, image, color= (0, 255, 255)):
        print("draw model 2")

        print(transformed_points)

        thickness = 2
        start = transformed_points[0].astype(np.int32)
        end = transformed_points[1].astype(np.int32)
        draw_line_segment(start, end, image, color,thickness)

        start = transformed_points[1].astype(np.int32)
        end = transformed_points[2].astype(np.int32)
        draw_line_segment(start, end, image, color,thickness)

        start = transformed_points[2].astype(np.int32)
        end = transformed_points[3].astype(np.int32)
        draw_line_segment(start, end, image, color,thickness)

        start = transformed_points[3].astype(np.int32)
        end = transformed_points[0].astype(np.int32)
        draw_line_segment(start, end, image, color,thickness)


        start = transformed_points[4].astype(np.int32)
        end = transformed_points[5].astype(np.int32)
        draw_line_segment(start, end, image, color,thickness)

        start = transformed_points[6].astype(np.int32)
        end = transformed_points[7].astype(np.int32)
        draw_line_segment(start, end, image, color,thickness)


        start = transformed_points[8].astype(np.int32)
        end = transformed_points[9].astype(np.int32)
        draw_line_segment(start, end, image, color,thickness)

        start = transformed_points[10].astype(np.int32)
        end = transformed_points[11].astype(np.int32)
        draw_line_segment(start, end, image, color,thickness)


        start = transformed_points[12].astype(np.int32)
        end = transformed_points[13].astype(np.int32)
        draw_line_segment(start, end, image, color,thickness)

        start = transformed_points[14].astype(np.int32)
        end = transformed_points[15].astype(np.int32)
        draw_line_segment(start, end, image, color,thickness)









#        draw_line_segment(transformed_points[4], transformed_points[5], image, color2, thickness)
#        draw_line_segment(transformed_points[6], transformed_points[7], image, color2, thickness)

#        draw_line_segment(transformed_points[8], transformed_points[9], image, color2, thickness)
#        draw_line_segment(transformed_points[10], transformed_points[11], image, color2, thickness)

#        draw_line_segment(transformed_points[12], transformed_points[13], image, color2, thickness)
#        draw_line_segment(transformed_points[14], transformed_points[15], image, color2, thickness)


        #draw_line(Line(transformed_points[0], transformed_points[1]), image, color)
        #draw_line(Line(transformed_points[1], transformed_points[2]), image, color)
        #draw_line(Line(transformed_points[2], transformed_points[3]), image, color)
        #draw_line(Line(transformed_points[3], transformed_points[0]), image, color)

        #draw_line(Line(transformed_points[4], transformed_points[5]), image, color)
        #draw_line(Line(transformed_points[6], transformed_points[7]), image, color)

        #draw_line(Line(transformed_points[8], transformed_points[9]), image, color)
        #draw_line(Line(transformed_points[10], transformed_points[11]), image, color)

        #draw_line(Line(transformed_points[12], transformed_points[13]), image, color)
        #draw_line(Line(transformed_points[14], transformed_points[15]), image, color)
