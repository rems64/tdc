#!/usr/bin/env python3

import cv2
import depthai as dai
import numpy as np
from dimensions import calibration_ids

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

resize_node = pipeline.create(dai.node.ImageManip)
# resize_node.setMaxOutputFrameSize(812*608*3)
# resize_node.setMaxOutputFrameSize(1228800)
resize_node.initialConfig.setResizeThumbnail(640, 640)
# resize_node.initialConfig.setKeepAspectRatio(False)
# resize_node.initialConfig.setResize(640, 640)
resize_node.initialConfig.setFrameType(dai.RawImgFrame.Type.RGB888p)
resize_node.setMaxOutputFrameSize(640*640*3)
# resize_node.initialConfig.setKeepAspectRatio(False)



camRgb.isp.link(resize_node.inputImage)

xoutRgb = pipeline.create(dai.node.XLinkOut)
xoutRgb.setStreamName("rgb")
resize_node.out.link(xoutRgb.input)

# Aruco detection
dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250)
parameters = cv2.aruco.DetectorParameters()
detector = cv2.aruco.ArucoDetector(dictionary, parameters)

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
        dai.CameraBoardSocket.CAM_A, 812, 608))
    dist_coeffs = np.array(calibData.getDistortionCoefficients(
        dai.CameraBoardSocket.CAM_A))

    print(intrinsics)

    calibrated = False
    should_calibrate = False

    decompte = 10
    print("[INFO] Press c to calibrate")

    while True:
        if qRgb.has():
            frame = qRgb.get()
            f = frame.getCvFrame()
            f = cv2.undistort(f, intrinsics, dist_coeffs)
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
                    if all_ids_present and should_calibrate:
                        if decompte > 0:
                            decompte -= 1
                            print(f"Calibration in {decompte}")
                        else:
                            imagePoints = np.zeros((4, 2))
                            objectPoints = np.zeros((4, 2)) # Object points are at z=0
                            for i, (calibration_id, position) in enumerate(calibration_ids):
                                corner = corners[list(ids).index(
                                    calibration_id)][0]
                                midpoint = np.mean(corner, axis=0)
                                imagePoints[i] = midpoint
                                objectPoints[i] = position[:2]
                            
                            mat = cv2.getPerspectiveTransform(imagePoints.astype(np.float32), objectPoints.astype(np.float32))

                            # success, rvec, tvec = cv2.solvePnP(
                            #     objectPoints, imagePoints, intrinsics, dist_coeffs, flags=cv2.SOLVEPNP_P3P)
                            # x_l, y_l, z_l = tvec
                            # Construct projection matrix based on rvec and tvec
                            # rmat, _ = cv2.Rodrigues(rvec)
                            np.save("image_points", imagePoints)
                            np.save("homography_mat", mat)
                            print("[INFO] Calibration success")
                            exit()
                else:
                    break
            else:
                decompte = 10

            cv2.imshow("RGB", f)
        
        k = cv2.waitKey(1)
        if k == ord('q'):
            break
        elif k == ord('c'):
            should_calibrate = True
