"""
Utility functions for warehouse simulation and optimization.

This module contains helper functions for distance computation,
data loading/saving, and visualization utilities.
"""

from typing import Tuple, List
import numpy as np
import matplotlib.pyplot as plt


def euclidean_distance(point1: Tuple[float, float], point2: Tuple[float, float]) -> float:
    """
    Calculate Euclidean distance between two 2D points.
    
    Args:
        point1: Tuple (x, y) coordinates of first point
        point2: Tuple (x, y) coordinates of second point
    
    Returns:
        Euclidean distance between the two points
    """
    return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)


def manhattan_distance(point1: Tuple[float, float], point2: Tuple[float, float]) -> float:
    """
    Calculate Manhattan distance between two 2D points.
    
    Args:
        point1: Tuple (x, y) coordinates of first point
        point2: Tuple (x, y) coordinates of second point
    
    Returns:
        Manhattan distance between the two points
    """
    return abs(point1[0] - point2[0]) + abs(point1[1] - point2[1])


def compute_distance_matrix(
    points: List[Tuple[float, float]], 
    distance_type: str = "euclidean"
) -> np.ndarray:
    """
    Compute pairwise distance matrix for a list of points.
    
    Args:
        points: List of (x, y) coordinate tuples
        distance_type: Type of distance metric ("euclidean" or "manhattan")
    
    Returns:
        n×n numpy array where entry (i, j) is the distance from point i to point j
    """
    n = len(points)
    distance_func = euclidean_distance if distance_type == "euclidean" else manhattan_distance
    
    matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                matrix[i, j] = distance_func(points[i], points[j])
            # Diagonal is already 0
    
    return matrix

