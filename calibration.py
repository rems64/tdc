#!/usr/bin/env python3

import cv2
import depthai as dai
import numpy as np

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
manip.setMaxOutputFrameSize(812*608*3)
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

dx = 1.493/2
dy = 0.995/2

calibration_ids = [
    (20, np.array([dx, -dy, 0])),
    (21, np.array([-dx, -dy, 0])),
    (22, np.array([dx,  dy, 0])),
    (23, np.array([-dx,  dy, 0]))
]

with dai.Device(pipeline) as device:
    qRgb = device.getOutputQueue(name='rgb')
    # qIsp = device.getOutputQueue(name='isp')

    def frameNorm(frame, bbox):
        normVals = np.full(len(bbox), frame.shape[0])
        normVals[::2] = frame.shape[1]
        return (np.clip(np.array(bbox), 0, 1) * normVals).astype(int)

    # Get the camera intrinsics
    calibData = device.readFactoryCalibration()
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

    decompte = 10

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
                        if decompte > 0:
                            decompte -= 1
                            print(f"Calibration in {decompte}")
                        else:
                            print("shape", gray.shape)
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
                                objectPoints, imagePoints, intrinsics, distCoeffs, flags=cv2.SOLVEPNP_P3P)
                            x_l, y_l, z_l = tvec
                            # Construct projection matrix based on rvec and tvec
                            rmat, _ = cv2.Rodrigues(rvec)
                            # extrinsic_matrix = np.hstack(
                            #     (np.linalg.inv(rmat), -tvec))
                            extrinsic_matrix_inv = np.vstack(
                                (np.hstack((rmat, tvec)), np.array([0, 0, 0, 1])))
                            # print(matrix)

                            # print(np.cos(rvec[0])*z_l)
                            # print(f"X: {x_l}, Y: {y_l}, Z: {z_l}")

                            # print(rvec)
                            # print()
                            # print(tvec)
                            # print()
                            # print(matrix)
                            # print()

                            extrinsic_matrix = np.linalg.inv(extrinsic_matrix_inv)
                            print(extrinsic_matrix)
                            print(extrinsic_matrix_inv)
                            translation = extrinsic_matrix[0:3, 3]
                            print("camera_position", translation)
                            save_path = "camera_extrinsic"
                            np.save(save_path, extrinsic_matrix)
                            # exit()
                            calibrated = True

                            print(f"Calibrated and saved to {save_path}")
                else:
                    break
            else:
                decompte = 10

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
