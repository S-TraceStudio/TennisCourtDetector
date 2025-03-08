from line import Line
import cv2
import numpy as np
from global_paramaters import global_params
from debug_helper import draw_lines
from utils import displayDebugImage

class CourtLineCandidateDetector:
    class Parameters:
        def __init__(self):
            self.houghThreshold = 240
            self.distanceThreshold = 8  # in pixels
            self.refinementIterations = 5

    debug = True
    windowName = "Court Line Candidate Detector"

    def __init__(self, parameters=None):
        if parameters is None:
            self.parameters = self.Parameters()
        else:
            self.parameters = parameters

    def run(self, binaryImage, rgbImage):
        lines = self.extractLines(binaryImage, rgbImage)
        for i in range(self.parameters.refinementIterations):
            print(f"Iteration: {i}")
            self.refineLineParameters(lines, binaryImage, rgbImage)
            self.removeDuplicateLines(lines, rgbImage)
        return lines

    def extractLines(self, binaryImage, rgbImage):
        lines = []
        tmpLines = cv2.HoughLines(binaryImage, 1, np.pi / 180, self.parameters.houghThreshold)
        if tmpLines is not None:
            for rho, theta in tmpLines[:, 0]:
                line = Line.from_rho_theta(rho, theta)
                print(line)
                lines.append(Line.from_rho_theta(rho, theta))
        if self.debug:
            print(f"CourtLineCandidateDetector::extractLines line count = {len(lines)}")
            image = rgbImage.copy()
            draw_lines(lines, image)
            displayDebugImage(image)
        return lines

    def refineLineParameters(self, lines, binaryImage, rgbImage):
        for i, line in enumerate(lines):
            lines[i] = self.getRefinedParameters(line, binaryImage, rgbImage)
        if self.debug:
            image = rgbImage.copy()
            draw_lines(lines, image)
            displayDebugImage(image)

    def getRefinedParameters(self, line, binaryImage, rgbImage):
        A = self.getClosePointsMatrix(line, binaryImage, rgbImage)
        [vx, vy, x, y] = cv2.fitLine(A, cv2.DIST_L2, 0, 0.01, 0.01)
        line = Line([x.item(), y.item()], [vx.item(), vy.item()])
        return line

    def getClosePointsMatrix(self, line, binaryImage, rgbImage):
        points = []
        for x in range(binaryImage.shape[1]):
            for y in range(binaryImage.shape[0]):
                if binaryImage[y, x] == global_params.fg_value:
                    point = (float(x), float(y))
                    distance = line.get_distance(point)
                    if distance < self.parameters.distanceThreshold:
                        points.append((x, y))
        return np.array(points, dtype=np.float32)

    def line_equal(a, b):
        return a.is_duplicate(b)

    def removeDuplicateLines(self, lines, rgbImage):
        self.image = rgbImage.copy()
        # Remove duplicates from the 'lines' list
        unique_lines = []
        for line in lines:
            if not any(line.is_duplicate(unique_line) for unique_line in unique_lines):
                if self.debug:
                    print(line)
                unique_lines.append(line)
        # Update the 'lines' list to only contain unique lines
        lines = unique_lines
        if self.debug:
            print(f"CourtLineCandidateDetector::removeDuplicateLines line count = {len(lines)}")
            image = rgbImage.copy()
            draw_lines(lines, image)
            displayDebugImage(image,widow_name=self.windowName)



