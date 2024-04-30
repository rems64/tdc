#!/usr/bin/env python3

import json
import threading
import cv2
import depthai as dai
import numpy as np

from http.server import BaseHTTPRequestHandler, HTTPServer, SimpleHTTPRequestHandler
from scipy.spatial.transform import Rotation   
import time

should_stop = False

# Create pipeline
pipeline = dai.Pipeline()

# Define source and output
camRgb = pipeline.create(dai.node.ColorCamera)
camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_12_MP)
camRgb.setInterleaved(False)
camRgb.setIspScale(1, 5)  # 4056x3040 -> 812x608
camRgb.setPreviewSize(608, 608)
camRgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.RGB)
# Slightly lower FPS to avoid lag, as ISP takes more resources at 12MP
camRgb.setFps(10)

# xoutIsp = pipeline.create(dai.node.XLinkOut)
# xoutIsp.setStreamName("isp")
# camRgb.isp.link(xoutIsp.input)

# Use ImageManip to resize to 300x300 with letterboxing
manip = pipeline.createImageManip()
# manip.initialConfig.setResize(300,300)
# manip.initialConfig.setResizeThumbnail(300, 300)
# manip.setResize(300,300)
manip.setMaxOutputFrameSize(812*608*3)  # 300x300x3
# manip.setMaxOutputFrameSize(270000)  # 300x300x3
# manip.setMaxOutputFrameSize(1108992) # 300x300x3
camRgb.isp.link(manip.inputImage)

xoutRgb = pipeline.create(dai.node.XLinkOut)
xoutRgb.setStreamName("rgb")
manip.out.link(xoutRgb.input)

# Aruco detection
dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250)
parameters = cv2.aruco.DetectorParameters()
detector = cv2.aruco.ArucoDetector(dictionary, parameters)

dx = 1.493
dy = 0.995

calibration_ids = [
    (20, np.array([dx, -dy, 0])),
    (21, np.array([-dx, -dy, 0])),
    (22, np.array([dx,  dy, 0])),
    (23, np.array([-dx,  dy, 0]))
]

camera_position = np.array([0, 0, 0])
camera_rotation = np.array([0, 0, 0])

class HTTPRequestHandler(SimpleHTTPRequestHandler):
    def end_headers (self):
        self.send_header('Access-Control-Allow-Origin', '*')
        SimpleHTTPRequestHandler.end_headers(self)
    
    def log_message(self, format: str, *args) -> None:
        return
        
    def do_GET(self):
        req = self.path[1:].split('/')
        # print(f"Received {req}")
        if req[0] == "get_position":
            self.send_response(200, json.dumps(camera_position.tolist()))
        elif req[0] == "get_rotation":
            self.send_response(200, json.dumps(camera_rotation.tolist()))
        else:
            self.send_response(404, "Command not found")
        self.end_headers()

def server_thread():
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    server.server_close()

server = HTTPServer(('', 8765), HTTPRequestHandler)
server.timeout = 0.1
threading.Thread(target=server_thread).start()

try:
    with dai.Device(pipeline) as device:
        qRgb = device.getOutputQueue(name='rgb')
        # qIsp = device.getOutputQueue(name='isp')

        def frameNorm(frame, bbox):
            normVals = np.full(len(bbox), frame.shape[0])
            normVals[::2] = frame.shape[1]
            return (np.clip(np.array(bbox), 0, 1) * normVals).astype(int)

        # Get the camera intrinsics
        # calibData = device.readFactoryCalibration()
        calibData = device.readCalibration()
        # 812x608 => 300x225 => 300x300
        intrinsics = np.array(calibData.getCameraIntrinsics(
            # dai.CameraBoardSocket.CAM_A, 812, 608))
            # dai.CameraBoardSocket.CAM_A, 300, 300))
            # dai.CameraBoardSocket.CAM_A, 300, 225))
            dai.CameraBoardSocket.CAM_A, 812, 608))
        distCoeffs = np.array(calibData.getDistortionCoefficients(
            dai.CameraBoardSocket.CAM_A))

        print(intrinsics)

        calibrated = False

        while True:
            if qRgb.has():
                frame = qRgb.get()
                f = frame.getCvFrame()
                gray = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)

                corners, ids, rejected = detector.detectMarkers(gray)
                if ids is not None:
                    cv2.aruco.drawDetectedMarkers(f, corners, ids)

                    if not calibrated:
                        all_ids_present = True
                        for (calibration_id, _) in calibration_ids:
                            if calibration_id not in ids:
                                all_ids_present = False
                                break
                        if all_ids_present:
                            # print("[INFO] All calibration markers found")
                            # Use the four markers to get the camera extrinsics

                            imagePoints = np.zeros((4, 2))
                            objectPoints = np.zeros((4, 3))
                            for i, (calibration_id, position) in enumerate(calibration_ids):
                                corner = corners[list(ids).index(
                                    calibration_id)][0]
                                midpoint = np.mean(corner, axis=0)
                                imagePoints[i] = midpoint
                                objectPoints[i] = position

                            # print(imagePoints)
                            # print(objectPoints)

                            success, rvec, tvec = cv2.solvePnP(
                                objectPoints, imagePoints, intrinsics, distCoeffs, flags=cv2.SOLVEPNP_AP3P)
                            x_l, y_l, z_l = tvec
                            # Construct projection matrix based on rvec and tvec
                            rmat, _ = cv2.Rodrigues(rvec)
                            matrix = np.vstack(
                                (np.hstack((rmat, tvec)), np.array([0, 0, 0, 1])))

                            # print(np.cos(rvec[0])*z_l)
                            # print(f"X: {x_l}, Y: {y_l}, Z: {z_l}")

                            # print(rvec)
                            # print()
                            # print(tvec)
                            # print()
                            # print(matrix)
                            # print()

                            matrix_inv = np.linalg.inv(matrix)
                            # print(matrix_inv)
                            translation = matrix_inv[0:3, 3]
                            
                            # camera_position = tvec
                            # camera_rotation = rvec
                            camera_position = translation
                            tmp_rotation = rvec
                            tmp_rotation[0] += np.pi
                            camera_rotation = -tmp_rotation
                            # camera_rotation = Rotation.from_matrix(matrix[:3,:3]).as_euler("zyx")

                            # exit()

                            # calibrated = True

                # cv2.putText(f, str(f.shape), (20, 20), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (255,255,255))
                cv2.imshow("RGB", f)
                # cv2.imshow("gray", gray)
            # if qIsp.has():
            #     frame = qIsp.get()
            #     f = frame.getCvFrame()
            #     cv2.putText(f, str(f.shape), (20, 20), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (255,255,255))
            #     cv2.imshow("ISP", f)

            if cv2.waitKey(1) == ord('q'):
                break
except Exception as e:
    print(e)
    # cv2.destroyAllWindows()
    server.shutdown()
    pass
print("Shutting down server...")
server.shutdown()