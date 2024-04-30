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
extrinsic_matrix = np.load("camera_extrinsic.npy")
print("extrinsic matrix is\n", extrinsic_matrix)
print("inverse extrinsic matrix is\n", np.linalg.inv(extrinsic_matrix))

# TODO: Distortion
def print_position_on_board(detection: dai.ImgDetection, rgb):
    print(rgb.shape)
    x_d = (detection.xmin+detection.xmax)/2
    y_d = (detection.ymin+detection.ymax)/2
    x = isp_width*(640*x_d-x_margin)/(640-2*x_margin)
    y = isp_height*(640*y_d-y_margin)/(640-2*y_margin)
    # cv2.circle(rgb, (int(y), int(x)), 10, (255, 0, 0))
    cv2.circle(rgb, (int(x/isp_width*(640-2*x_margin)+x_margin), int(y/isp_height*(640-2*y_margin)+y_margin)), 4, (0, 255, 0), 4)
    print("x", x, "y", y)
    focal = intrinsic_matrix[0,0]
    print("focal", focal)
    intrinsic_inv = np.linalg.inv(intrinsic_matrix)
    extrinsic_inv = np.linalg.inv(extrinsic_matrix)
    camera_x = extrinsic_inv[0,3]
    camera_y = extrinsic_inv[1,3]
    camera_z = extrinsic_inv[2,3]
    print("camera_x", camera_x, "camera_y", camera_y, "camera_z", camera_z)
    # print("intrinsic_inv\n", intrinsic_inv)
    pt_camspace = intrinsic_inv@np.array([x, y, focal])
    print("pt_cameraspace\t", pt_camspace)
    proj_res = intrinsic_matrix@extrinsic_inv[:3,:]@np.array([0, 0, 0, 1])
    print("projection", proj_res[:2]/proj_res[2])
    pt = extrinsic_matrix@np.hstack((pt_camspace, 1))
    pt_x = pt[0]
    pt_y = pt[1]
    pt_z = pt[2]
    alpha = -camera_z/(pt_z-camera_z)
    obj = np.array([(pt_x-camera_x)*alpha+camera_x, (pt_y-camera_y)*alpha+camera_y, (pt_z-camera_z)*alpha+camera_z])
    # pt[2]/
    # print(x, y)
    print(obj)

# Connect to device and start pipeline
with dai.Device(pipeline) as device:
    calib_data = device.readFactoryCalibration()
    # 812x608 => 300x225 => 300x300
    intrinsic_matrix = np.array(calib_data.getCameraIntrinsics(
        dai.CameraBoardSocket.CAM_A, 812, 608))
    print("intrinsic matrix\n", intrinsic_matrix)

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
                print_position_on_board(detections[plant_id])
        else:
            if syncNN:
                inRgb = qRgb.get()
            else:
                inRgb = qRgb.tryGet()
            if inRgb is not None:
                frame = inRgb.getCvFrame()
                detec = dai.ImgDetection()
                w, h = 0.05, 0.05
                x, y = 0.52, 0.67
                # x, y = 1.0, 0.15
                detec.xmin = x-w
                detec.xmax = x+w
                detec.ymin = y-h
                detec.ymax = y+h
                print_position_on_board(detec, frame)
                bbox = frame_norm(frame, (detec.xmin, detec.ymin, detec.xmax, detec.ymax))
                cv2.rectangle(frame, (bbox[0], bbox[1]),
                            (bbox[2], bbox[3]), (0, 255, 0), 2)
                cv2.imshow("name", frame)
                cv2.imwrite("capture.png", frame)
                # if len(detections) > 0:
                #     for (i, detection) in enumerate(detections):
                #         print_position_on_board(detection, frame)
                # cv2.putText(frame, "NN fps: {:.2f}".format(counter / (time.monotonic() - startTime)),
                #             (2, frame.shape[0] - 4), cv2.FONT_HERSHEY_TRIPLEX, 0.4, color2)
            # if frame is not None:
            #     display_frame("rgb", frame)
            break
            if cv2.waitKey(1) == ord('q'):
                break
