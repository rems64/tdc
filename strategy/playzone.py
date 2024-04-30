import numpy as np


class Playzone:
    def __init__(self):
        self.size = np.array([200., 100.])
        self.POIs = np.zeros((100, 3))

        self.POIs[:, 0] = -1


class Robot:
    def __init__(self):
        self.position = np.array([0., 0.])
        self.orientation = 0
        self.slots = -np.ones((6,), dtype=int)
    
    def get_available_slot(self)->int:
        for i in range(6):
            if self.slots[i]<0:
                return i
        return -1
    
    def pick(self, position: np.ndarray):
        pass
    
    def goto(self, target: np.ndarray):
        pass
    
    def is_full(self):
        return len(self.slots[self.slots<0])<=0
    
    def drop(self, slot: int):
        pass