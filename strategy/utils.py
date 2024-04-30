import numpy as np

def normalize(x: np.ndarray, ord=2):
    norm = np.linalg.norm(x, ord)
    if norm <= 0:
        return x
    return x/norm

def clamp(x, a, b):
    return min(max(x, a), b)