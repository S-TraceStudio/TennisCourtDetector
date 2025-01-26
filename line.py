import numpy as np
from geometry import perpendicular

class Line:
    def __init__(self, point, vector):
        self.u = point
        self.v = self.normalize(vector)

    @staticmethod
    def from_rho_theta(rho, theta):
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        p1 = np.array([x0 + 2000 * (-b), y0 + 2000 * a], dtype=np.float32)
        p2 = np.array([x0 - 2000 * (-b), y0 - 2000 * a], dtype=np.float32)
        return Line.from_two_points(p1, p2)

    @staticmethod
    def from_two_points(p1, p2):
        vec = p2 - p1
        return Line(p1, vec)

    @staticmethod
    def normalize(vector):
        return vector / np.linalg.norm(vector)

    def get_point(self):
        return self.u

    def get_vector(self):
        return self.v

    def get_distance(self, point):
        point_on_line = self.get_point_on_line_closest_to(point)
        return np.linalg.norm(point - point_on_line)

    def get_point_on_line_closest_to(self, point):
        n, c = self.to_implicit()
        q = float(c - np.dot(n, point))
        return point - q * np.asarray(n)

    def is_duplicate(self, other_line):
        n1, c1 = self.to_implicit()
        n2, c2 = other_line.to_implicit()
        dot = np.abs(np.dot(n1, n2))
        dot_threshold = np.cos(np.deg2rad(1))
        return (dot > dot_threshold) and (np.abs(np.abs(c1) - np.abs(c2)) < 10)

    def to_implicit(self):
        n = perpendicular(self.v)
        c = np.dot(n, self.u)
        return n, c

    def is_vertical(self):
        n, _ = self.to_implicit()
        value = ((np.abs(np.arctan2(n[1], n[0])) < np.deg2rad(65)) or (np.abs(np.arctan2(-n[1], -n[0])) < np.deg2rad(65)))
        return value

    def compute_intersection_point(self, other_line):
        x = other_line.get_point() - self.u
        d1 = self.v
        d2 = other_line.get_vector()
        cross = d1[0] * d2[1] - d1[1] * d2[0]
        if np.abs(cross) < 1e-8:
            return False, None
        t1 = (x[0] * d2[1] - x[1] * d2[0]) / cross
        intersection_point = self.u + d1 * t1
        return True, intersection_point
