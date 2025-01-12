import math
from typing import List, Tuple

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

class LineComparator:
    def __init__(self, point: Tuple[float, float]):
        self.p = point

    def __call__(self, lineA, lineB):
        return lineA.getDistance(self.p) < lineB.getDistance(self.p)

def sortLinesByDistanceToPoint(lines: List, point: Tuple[float, float]) -> None:
    lines.sort(key=LineComparator(point))
