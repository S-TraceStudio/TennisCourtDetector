import math
from typing import List, Tuple
import functools

def length(v: Tuple[float, float]) -> float:
    return math.sqrt(v[0]**2 + v[1]**2)

def distance(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
    dx = p1[0] - p2[0]
    dy = p1[1] - p2[1]
    return math.sqrt(dx**2 + dy**2)

def perpendicular(v: Tuple[float, float]) -> Tuple[float, float]:
    return (-v[1], v[0])

def normalize(v: Tuple[float, float]) -> Tuple[float, float]:
    l = length(v)
    return (v[0] / l, v[1] / l)

def bubbleSort(arr,point):
    n = len(arr)
    # Traverse through all array elements
    for i in range(n):
        swapped = False
        # Last i elements are already in place
        for j in range(0, n - i - 1):
            # Traverse the array from 0 to n-i-1
            # Swap if the element found is greater
            # than the next element
            if LineComparator(point)(arr[j], arr[j + 1]):
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
                swapped = True
        if (swapped == False):
            break


class LineComparator:
    def __init__(self, point: Tuple[float, float]):
        self.p = point

    def __call__(self, lineA, lineB):
        #print(f"p = {self.p}")
        #print(f"lineA = {lineA}")
        #print(f"lineA.getDistance(p) = {lineA.get_distance(self.p)}")
        #print(f"lineB = {lineB}")
        #print(f"lineB.getDistance(p) = {lineB.get_distance(self.p)}")
        return lineA.get_distance(self.p) > lineB.get_distance(self.p)

def sort_lines_by_distance_to_point(lines: List, point: Tuple[float, float]) -> None:
    bubbleSort(lines,point)
    #lines = sorted(lines,key=functools.cmp_to_key(LineComparator(point)))

