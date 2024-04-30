import cv2
import numpy as np
import time

image_points_a = np.array([[100, 100], [150, 100], [150, 150], [100, 150]], np.float32)
image_points_b = np.array([[300, 300], [350, 300], [500, 450], [300, 450]], np.float32)

mat = cv2.getPerspectiveTransform(image_points_a, image_points_b)

stop = False
while not stop:
    t = time.time()
    y = (np.sin(t)*0.5+0.5)*50+100
    x = (np.sin(4*t)*0.5+0.5)*50+100
    point = np.array([[[x, y]]], np.float32)
    point_mapped = cv2.perspectiveTransform(point, mat)
    # point_mapped: np.ndarray = mat@np.array([point[0,0,0], point[0,0,1], 1])
    # point_mapped = point_mapped[:2]/point_mapped[2]
    point_mapped: np.ndarray = point_mapped[0,0]

    point: np.ndarray = point[0,0]
    # point += image_points_a[0]
    # point_mapped += image_points_b[0]

    f = np.zeros((600, 600, 3))
    cv2.polylines(f, [image_points_a.astype(np.int32).reshape((-1, 1, 2))], True, (255, 0, 0), 2)
    cv2.polylines(f, [image_points_b.astype(np.int32).reshape((-1, 1, 2))], True, (0, 0, 255), 2)

    cv2.circle(f, point.astype(np.int32), 8, (255, 0, 0))
    cv2.circle(f, np.round(point_mapped).astype(np.int32), 8, (0, 0, 255))

    cv2.imshow("before", f)
    k = cv2.waitKey(1)
    
    if k == ord("q"):
        stop = True