from pathlib import Path
import sys
import cv2
import depthai as dai
import numpy as np
import time

PROD = False
GET_RGB = True

path = "./custom_trained.blob"
nnPath = str((Path(__file__).parent /
             Path('./custom_trained.blob')).resolve().absolute())

if not Path(nnPath).exists():
    import sys
    raise FileNotFoundError(f"Couldn't find {path}")

labelMap = ["Plant", "Pot"]

syncNN = True

pipeline = dai.Pipeline()

# Define sources and outputs
camRgb = pipeline.create(dai.node.ColorCamera)

camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_12_MP)
camRgb.setInterleaved(False)
camRgb.setIspScale(1, 5)  # 4056x3040 -> 812x608
# camRgb.setPreviewSize(640, 480)
camRgb.setPreviewSize(812, 608)
camRgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
# Slightly lower FPS to avoid lag, as ISP takes more resources at 12MP
camRgb.setFps(20 if PROD else 10)


nnOut = pipeline.create(dai.node.XLinkOut)
nnOut.setStreamName("nn")

isp_width = camRgb.getIspWidth()
isp_height = camRgb.getIspHeight()
print(f"width ({isp_width}) height ({isp_height})")
x_margin = max(0, (640-640/isp_height*isp_width)/2)
y_margin = max(0, (640-640/isp_width*isp_height)/2)
print(f"x_margin is {x_margin}")
print(f"y_margin is {y_margin}")

if GET_RGB:
    xoutRgb = pipeline.create(dai.node.XLinkOut)
    xoutRgb.setStreamName("rgb")

resize_node = pipeline.create(dai.node.ImageManip)
# resize_node.setMaxOutputFrameSize(812*608*3)
# resize_node.setMaxOutputFrameSize(1228800)
resize_node.initialConfig.setResizeThumbnail(640, 640)
# resize_node.initialConfig.setKeepAspectRatio(False)
# resize_node.initialConfig.setResize(640, 640)
resize_node.initialConfig.setFrameType(dai.RawImgFrame.Type.RGB888p)
resize_node.setMaxOutputFrameSize(640*640*3)
# resize_node.initialConfig.setKeepAspectRatio(False)

detectionNetwork = pipeline.create(dai.node.YoloDetectionNetwork)
detectionNetwork.setConfidenceThreshold(0.5)
detectionNetwork.setNumClasses(2)
detectionNetwork.setCoordinateSize(4)
detectionNetwork.setAnchors([10.0, 13.0, 16.0, 30.0, 33.0, 23.0, 30.0,
                            61.0, 62.0, 45.0, 59.0, 119.0, 116.0, 90.0, 156.0, 198.0, 373.0, 326.0])
detectionNetwork.setAnchorMasks(
    {"side80": [0, 1, 2], "side40": [3, 4, 5], "side20": [6, 7, 8]})
# detectionNetwork.setAnchorMasks({"side26": [1, 2, 3], "side13": [3, 4, 5]})
# detectionNetwork.setAnchorMasks({"side26": [1, 2, 3], "side13": [3, 4, 5]})
detectionNetwork.setIouThreshold(0.5)
detectionNetwork.setBlobPath(nnPath)
detectionNetwork.setNumInferenceThreads(2)
detectionNetwork.input.setBlocking(False)

# Linking
# camRgb.preview.link(detectionNetwork.input)
camRgb.isp.link(resize_node.inputImage)
resize_node.out.link(detectionNetwork.input)
if GET_RGB:
    if syncNN:
        resize_node.out.link(xoutRgb.input)
        # detectionNetwork.passthrough.link(xoutRgb.input)
    else:
        resize_node.out.link(xoutRgb.input)

detectionNetwork.out.link(nnOut.input)

intrinsic_matrix = None
# extrinsic_matrix = np.load("camera_extrinsic.npy")

image_points = np.load("image_points.npy")
homography_mat = np.load("homography_mat.npy")
unitary_vectors = np.zeros((4,2))

def get_position_on_board(detection: dai.ImgDetection, rgb):
    detection_xmin = detection[0,0]
    detection_ymin = detection[0,1]
    detection_xmax = detection[1,0]
    detection_ymax = detection[1,1]
    x_d = (detection_xmin+detection_xmax)/2 * 640
    y_d = (detection_ymin+detection_ymax)/2 * 640
    # print(x_d, y_d)
    cam_point = np.array([[[x_d, y_d]]], np.float32)
    point_mapped = cv2.perspectiveTransform(cam_point, homography_mat)
    point_mapped: np.ndarray = 1000.*point_mapped[0,0]
    print(point_mapped)
    return point_mapped

# Aruco detection
dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250)
parameters = cv2.aruco.DetectorParameters()
detector = cv2.aruco.ArucoDetector(dictionary, parameters)

# Connect to device and start pipeline
with dai.Device(pipeline) as device:
    calib_data = device.readFactoryCalibration()
    # 812x608 => 300x225 => 300x300
    intrinsic_matrix = np.array(calib_data.getCameraIntrinsics(
        dai.CameraBoardSocket.CAM_A, 812, 608))
    print("intrinsic matrix\n", intrinsic_matrix)
    dist_coeffs = np.array(calib_data.getDistortionCoefficients(dai.CameraBoardSocket.CAM_A))

    # Output queues will be used to get the rgb frames and nn data from the outputs defined above
    if GET_RGB:
        qRgb = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)
    qDet = device.getOutputQueue(name="nn", maxSize=4, blocking=False)

    frame = None
    detections = []
    startTime = time.monotonic()
    counter = 0
    color2 = (255, 255, 255)

    # nn data, being the bounding box locations, are in <0..1> range - they need to be normalized with frame width/height
    def frame_norm(frame, bbox):
        normVals = np.full(len(bbox), frame.shape[0])
        normVals[::2] = frame.shape[1]
        return (np.clip(np.array(bbox), 0, 1) * normVals).astype(int)

    def display_frame(name, frame):
        color = (255, 0, 0)
        for detection in detections:
            bbox = frame_norm(
                frame, (detection.xmin, detection.ymin, detection.xmax, detection.ymax))
            cv2.putText(frame, labelMap[detection.label], (bbox[0] +
                        10, bbox[1] + 20), cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)
            cv2.putText(frame, f"{int(detection.confidence * 100)}%",
                        (bbox[0] + 10, bbox[1] + 40), cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)
            cv2.rectangle(frame, (bbox[0], bbox[1]),
                          (bbox[2], bbox[3]), color, 2)
        # Show the frame
        cv2.imshow(name, frame)

    fps = 0
    while True:
        if syncNN:
            inDet = qDet.get()
        else:
            inDet = qDet.tryGet()

        if inDet is not None:
            detections = inDet.detections
            counter += 1
            fps = counter / (time.monotonic() - startTime)

        if not GET_RGB:
            # print(f"We detected {len(detections)} objects ({fps} fps)")
            if len(detections) > 0:
                plant_id = 0
                for (i, detection) in enumerate(detections):
                    if detection.label==0:
                        plant_id = i
                get_position_on_board(detections[plant_id])
        else:
            if syncNN:
                inRgb = qRgb.get()
            else:
                inRgb = qRgb.tryGet()
            if inRgb is not None:
                frame = inRgb.getCvFrame()
                frame = cv2.undistort(frame, intrinsic_matrix, dist_coeffs)
                # detec = dai.ImgDetection()
                # w, h = 0.05, 0.05
                # x, y = 0.52, 0.67
                # # x, y = 1.0, 0.15
                # detec.xmin = x-w
                # detec.xmax = x+w
                # detec.ymin = y-h
                # detec.ymax = y+h
                for (i, detection) in enumerate(detections):
                    points = np.array([[detection.xmin*frame.shape[0], detection.ymin*frame.shape[1]], [detection.xmax*frame.shape[0], detection.ymax*frame.shape[1]]])
                    # print("shape1", points.shape)
                    undistorded_points = cv2.undistortImagePoints(points, intrinsic_matrix, dist_coeffs)[:,0,:]/frame.shape[:2]
                    # print("shape2", undistorded_points.shape)
                    # print(undistorded_points)
                    position = get_position_on_board(undistorded_points, frame)
                    bbox = frame_norm(frame, (undistorded_points[0,0], undistorded_points[0,1], undistorded_points[1,0], undistorded_points[1,1]))
                    cv2.rectangle(frame, (bbox[0], bbox[1]),
                                (bbox[2], bbox[3]), (0, 255, 0), 2)
                    cv2.putText(frame, str([int(x) for x in position]), (bbox[0], bbox[1]), cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)
                cv2.imshow("name", frame)
                # cv2.imwrite("capture.png", frame)
            # break
            if cv2.waitKey(1) == ord('q'):
                break
