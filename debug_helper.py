import cv2
import numpy as np
from line import Line

def print_video_info(vc):
    frame_count = int(vc.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(vc.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vc.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(vc.get(cv2.CAP_PROP_FPS))
    print(f"Video properties (frames: {frame_count}, w: {width}, h: {height}, fps: {fps})")

def display_image(window_name, image, delay):
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.imshow(window_name, image)
    cv2.waitKey(delay)


def draw_lines(lines, image, color = (0, 0, 255), thickness = 2):
    for line in lines:
        draw_line(line, image, color, thickness)

def draw_line(line, image, color = (0, 0, 255), thickness = 2):
    top_border = Line([0, 0], [1, 0])
    left_border = Line([0, 0], [0, 1])
    bottom_border = Line([image.shape[1], image.shape[0]], [1, 0])
    right_border = Line([image.shape[1], image.shape[0]], [0, 1])

    p1, p2 = None, None
    compute1 = line.compute_intersection_point(top_border)
    compute2 = line.compute_intersection_point(bottom_border)
    compute3 = line.compute_intersection_point(left_border)
    compute4 = line.compute_intersection_point(right_border)

    if compute1[0] and compute2[0]:
        p1 = compute1[1].astype(np.int32)
        p2 = compute2[1].astype(np.int32)
        cv2.line(image, p1, p2, color, thickness, 8)
    elif compute3[0] and compute4[0]:
        p1 = compute3[1].astype(np.int32)
        p2 = compute4[1].astype(np.int32)
        cv2.line(image, p1, p2, color, thickness, 8)
    else:
        raise RuntimeError("No intersections found!")

def draw_line_segment(start, end, image, color, thickness):
    cv2.line(image, start, end, color, thickness, 8)

def draw_points(points, image, color):
    for point in points:
        draw_point(point, image, color)

def draw_point(point, image, color):
    cv2.circle(image, point, 3, color, -1)

def print_info(matrix, name):
    print(f"{name} rows {matrix.shape[0]}, cols {matrix.shape[1]}, type {matrix.dtype}")

def print_info_point(point, name):
    print(f"{name}({point[0]}, {point[1]})")

def print_info_line(line, name):
    print(f"{name}: ")
    print_info_point(line[0], "Point")
    print_info_point(line[1], "Vector")
