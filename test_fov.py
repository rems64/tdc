import depthai as dai

with dai.Device() as device:
    calibData = device.readCalibration()
    calibData.getCameraExtrinsics()
    print(f"FOVs")
    print(f"\tCAM_A {calibData.getFov(dai.CameraBoardSocket.CAM_A)}")
    print(f"\tCAM_B {calibData.getFov(dai.CameraBoardSocket.CAM_B)}")
    print(f"\tCAM_C {calibData.getFov(dai.CameraBoardSocket.CAM_C)}")