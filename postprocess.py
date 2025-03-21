import cv2
import numpy as np
from scipy.spatial import distance
from utils import line_intersection, displayDebugImage


def postprocess(heatmap, scaleX=1, scaleY=1, low_thresh=155, min_radius=10, max_radius=30):
    x_pred, y_pred = None, None
    ret, heatmap = cv2.threshold(heatmap, low_thresh, 255, cv2.THRESH_BINARY)
    circles = cv2.HoughCircles(heatmap, cv2.HOUGH_GRADIENT, dp=1, minDist=20, param1=50, param2=2, minRadius=min_radius,
                               maxRadius=max_radius)
    if circles is not None:
        x_pred = circles[0][0][0] * scaleX
        y_pred = circles[0][0][1] * scaleY
    return x_pred, y_pred

def refine_kps(img, x_ct, y_ct, crop_size=40, debug=False):
    refined_x_ct, refined_y_ct = x_ct, y_ct
    
    img_height, img_width = img.shape[:2]
    x_min = max(x_ct-crop_size, 0)
    x_max = min(img_height, x_ct+crop_size)
    y_min = max(y_ct-crop_size, 0)
    y_max = min(img_width, y_ct+crop_size)

    img_crop = img[x_min:x_max, y_min:y_max]
    lines = detect_lines(img_crop,debug)

    # if (debug):
    #     newIntersectionMethod(img_crop)

    if (debug):
        displayDebugImage(img_crop,scale=5)
        print("Lines count",len(lines))
        lineCount = len(lines)
        # Afficher les lignes
        linesImage = img_crop.copy()
        if lines is not None:
            i = 0
            for line in lines:
                x1, y1, x2, y2 = line
                colorMax = 200
                color = (i*colorMax/lineCount,i*colorMax/lineCount,i*colorMax/lineCount)
                cv2.line(linesImage, (x1, y1), (x2, y2), color, 1)
                i += 1
            displayDebugImage(linesImage,scale=5)

    if len(lines) > 1:
        lines = merge_lines(lines)
        print("Lines merged count",len(lines))

        if len(lines) != 2:
            lines = merge_lines(lines)

        if (debug):
            if lines is not None:
                for line in lines:
                    x1, y1, x2, y2 = line
                    cv2.line(linesImage, (x1, y1), (x2, y2), (200, 0, 0), 1)
                displayDebugImage(linesImage,scale=5)

        if len(lines) == 2:
            inters = line_intersection(lines[0], lines[1])
            if inters:
                new_x_ct = int(inters[1])
                new_y_ct = int(inters[0])
                if new_x_ct > 0 and new_x_ct < img_crop.shape[0] and new_y_ct > 0 and new_y_ct < img_crop.shape[1]:
                    refined_x_ct = x_min + new_x_ct
                    refined_y_ct = y_min + new_y_ct                    
    return refined_y_ct, refined_x_ct


def detect_lines(image, debug=False):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if (debug):
        displayDebugImage(gray, scale=5)
    gray = cv2.threshold(gray, 155, 255, cv2.THRESH_BINARY)[1]
    if (debug):
        displayDebugImage(gray, scale=5)
    lines = cv2.HoughLinesP(gray, 1, np.pi / 180, 30, minLineLength=10, maxLineGap=30)
    lines = np.squeeze(lines) 
    if len(lines.shape) > 0:
        if len(lines) == 4 and not isinstance(lines[0], np.ndarray):
            lines = [lines]
    else:
        lines = []
    return lines

def merge_lines(lines):
    lines = sorted(lines, key=lambda item: item[0])
    mask = [True] * len(lines)
    new_lines = []

    for i, line in enumerate(lines):
        if mask[i]:
            for j, s_line in enumerate(lines[i + 1:]):
                if mask[i + j + 1]:
                    x1, y1, x2, y2 = line
                    x3, y3, x4, y4 = s_line
                    dist1 = distance.euclidean((x1, y1), (x3, y3))
                    dist2 = distance.euclidean((x2, y2), (x4, y4))
                    if dist1 < 20 and dist2 < 20:
                        line = np.array([int((x1+x3)/2), int((y1+y3)/2), int((x2+x4)/2), int((y2+y4)/2)],
                                        dtype=np.int32)
                        mask[i + j + 1] = False
            new_lines.append(line)  
    return new_lines

def find_intersection(line1, line2):
    x1, y1, x2, y2 = line1
    x3, y3, x4, y4 = line2
    denominator = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if denominator == 0:
        return None # Lines are parallel
    px = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / denominator
    py = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / denominator

    return px, py

def newIntersectionMethod(image):
    newIntersectionImage = image.copy()
    gray = cv2.cvtColor(newIntersectionImage, cv2.COLOR_BGR2GRAY)
    # Detect edges
    edges = cv2.Canny(gray, 10, 200, apertureSize=3)
    displayDebugImage(edges, scale=5)
    # Detect lines using Hough Line Transform
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=100, minLineLength=10, maxLineGap=1)
    # Find intersections if lines is not None:
    for i in range(len(lines)):
        for j in range(i + 1, len(lines)):
            line1 = lines[i][0]
            line2 = lines[j][0]
            intersection_point = find_intersection(line1, line2)
            if intersection_point:
                px, py = intersection_point
                cv2.circle(newIntersectionImage, (int(px), int(py)), 5, (0, 0, 255), -1) #
    # Display the result
    displayDebugImage(newIntersectionImage, scale=5)

