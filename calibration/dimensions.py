import numpy as np

dx = 1.493/2
dy = 0.995/2

calibration_ids = [
    (20, np.array([dx, -dy, 0])),
    (21, np.array([-dx, -dy, 0])),
    (22, np.array([dx,  dy, 0])),
    (23, np.array([-dx,  dy, 0]))
]