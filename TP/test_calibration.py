from tp import *

class TestCalibration1(Test):
    def __init__(self):
        Test.__init__(self, "test_calibration", "3.1")
    
    def execute_test(self):
        self.success("calibration successfull")


if __name__=="__main__":
    run_test(TestCalibration1)