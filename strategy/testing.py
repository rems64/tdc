import random
from matplotlib.collections import EllipseCollection
from matplotlib.patches import Polygon
import numpy as np
import matplotlib.pyplot as plt

from playzone import Playzone, Robot
from utils import clamp, normalize

# 0: pot
# 1: plante

# 0: blue
# 1: yellow

MAP_SIZE = (3000, 2000)
RADIUS = np.array([35, 25])
COLORS = np.array([(1, 0, 0), (0, 1, 0)])

ROBOT_RADIUS = 170
CIRCLES = [np.array([[r*np.cos(a), r*np.sin(a)]
                    for a in np.linspace(0, 2*np.pi, 7)]) for r in RADIUS]

# [side, x1, y1, x2, y2]
ZONES = np.array([
    [0, -1500, 1000, -1055, 560],
    [1, -1500, 225, -1055, -225],
    [0, -1500, -1000, -1055, -560],
    [1, 1500, 1000, 1055, 560],
    [0, 1500, 225, 1055, -225],
    [1, 1500, -1000, 1055, -560]
])


class TestingPlayzone(Playzone):
    def __init__(self):
        Playzone.__init__(self)

        self._draw_figure = plt.figure()
        self._paillettes = np.zeros((300,2))
        self._velocities = np.zeros((300,2))
        self._index = 0

    def show(self, robot: 'TestingRobot'):
        # POIs
        relevant_pois = self.POIs[self.POIs[:, 0] >= 0]
        indices = relevant_pois[:, 0].astype(int)
        plt.clf()
        ax = plt.subplot()
        ax.set_xlim(-MAP_SIZE[0]//2, MAP_SIZE[0]//2)
        ax.set_ylim(-MAP_SIZE[1]//2, MAP_SIZE[1]//2)
        for zone in ZONES:
            x1, y1, x2, y2 = zone[1:]
            rect = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]])
            draw_polygon(rect, np.array(
                [0., 0.]), 0, ax, (0, 0, 1) if zone[0] == 0 else (1, 1, 0))
        d = 2*RADIUS[indices]
        c = COLORS[indices]
        ax.add_collection(EllipseCollection(d, d, 0, units='xy', facecolors=c,
                          offsets=relevant_pois[:, 1:3], transOffset=ax.transData))
        for _ in range(10):
            self._paillettes[self._index,0] = robot.position[0] + (2*random.random()-1)*30
            self._paillettes[self._index,1] = robot.position[1] + (2*random.random()-1)*30
            self._velocities[self._index,0] = (2*random.random()-1)*5
            self._velocities[self._index,1] = (2*random.random()-1)*5
            self._index = (self._index+1)%300
        self._paillettes += self._velocities
        ax.scatter(self._paillettes[:,0], self._paillettes[:,1], s=0.1, c=(1, 0, 0))
        robot.draw(ax)

    def resolve_collisions(self, robot: 'TestingRobot'):
        # relevant_pois = self.POIs[self.POIs[:, 0] >= 0]
        for i in range(self.POIs.shape[0]):
            if self.POIs[i,0]<0:
                continue
            poi_pos = self.POIs[i,1:]
            dist = np.linalg.norm(poi_pos-robot.position)
            ri = RADIUS[int(self.POIs[i,0])]
            if 0 < dist < ROBOT_RADIUS:
                self.POIs[i,1:] += (poi_pos-robot.position)/dist*ROBOT_RADIUS
            for j in range(i+1, self.POIs.shape[0]):
                poj_pos = self.POIs[j,1:] 
                if self.POIs[j,0]<0:
                    continue
                dist = np.linalg.norm(poi_pos-poj_pos)
                rj = RADIUS[int(self.POIs[j,0])]
                if dist < (ri+rj):
                    delta = (poj_pos-poi_pos)/dist*((ri+rj)-dist)/2
                    self.POIs[i,1:] -= delta
                    self.POIs[j,1:] += delta
            
            self.POIs[i,1] = clamp(self.POIs[i,1], -1500+ri, 1500-ri)
            self.POIs[i,2] = clamp(self.POIs[i,2], -1000+ri, 1000-ri)


class TestingRobot(Robot):
    def __init__(self, side):
        Robot.__init__(self)
        self._target = np.array([0., 0.])
        self._max_speed = 1000.
        self._state = "idle"
        self._action = "none"

        self._side = side

    def position_to_start(self):
        available_start_positions = ZONES[ZONES[:, 0] == self._side]
        zone = ZONES[random.randrange(0, len(available_start_positions))]
        self.position[0] = (zone[1]+zone[3])/2
        self.position[1] = (zone[2]+zone[4])/2

    def goto(self, target: np.ndarray):
        Robot.goto(self, target)
        self._target = target

    def move(self, dt: float):
        velocity = np.zeros((2,))
        if self._action == "move_to_target":
            velocity = self._target-self.position

        velocity = normalize(velocity)*self._max_speed
        self.position += velocity*dt

    def pick(self, position: np.ndarray, playzone: Playzone)->int:
        Robot.pick(self, position)
        if np.linalg.norm(position-self.position) > 300:
            print("[WARN] Attempting to grab too far from the robot")
            return -1

        for i in range(100):
            if playzone.POIs[i, 0] < 0:
                continue
            norm = np.linalg.norm(position-playzone.POIs[i, 1:])
            if norm < 100:
                slot = self.get_available_slot()
                if slot >= 0:
                    self.slots[slot] = playzone.POIs[i, 0]
                    playzone.POIs[i, 0] = -1
                    return 0
                print("[WARN] Attempted to grab with no empty slot left")
                return -2
        print("[WARN] Attempted to grab but no object found")
        return -3

    def drop(self, slot: int, playzone: Playzone):
        Robot.drop(self, slot)
        if self.slots[slot]<0:
            return -1
        drop_point = self.position+np.array([0., ROBOT_RADIUS + 50.])
        poi_index = -1
        for i in range(100):
            if playzone.POIs[i,0]<0:
                poi_index = i
                break
        playzone.POIs[poi_index][0] = self.slots[slot]
        playzone.POIs[poi_index][1] = drop_point[0]+random.random()*10.
        playzone.POIs[poi_index][2] = drop_point[1]+random.random()*10.
        self.slots[slot] = -1
        return 0
    
    def drop_all(self, playzone: Playzone):
        for i in range(6):
            if self.slots[i]>=0:
                print(f"[INFO] Dropping slot {i}")
                self.drop(i, playzone)        

    def draw(self, ax):
        # Robot
        body = np.array([[ROBOT_RADIUS*np.cos(a+np.pi/2), ROBOT_RADIUS*np.sin(a+np.pi/2)]
                        for a in np.linspace(0, 2*np.pi, 4)])
        arrow = np.array([[-40, -60], [0, 60], [40, -60]])
        draw_polygon(body, self.position,
                     self.orientation, ax, (0.49, 0.30, 0.27))
        draw_polygon(arrow, self.position,
                     self.orientation, ax, (1, 1, 1))
        for slot in range(6):
            if self.slots[slot] >= 0:
                id = self.slots[slot]
                delta_angle = slot*np.pi/3
                position = np.array(
                    [ROBOT_RADIUS*np.cos(delta_angle), ROBOT_RADIUS*np.sin(delta_angle)])
                draw_polygon(CIRCLES[id], position +
                             self.position, 0, ax, COLORS[id])

def draw_polygon(points, position, orientation, ax: plt.Axes, color: tuple[3]):
    robot_points_oriented = np.zeros_like(points)
    cos_o = np.cos(orientation)
    sin_o = np.sin(orientation)
    robot_points_oriented[:, 0] = cos_o*points[:, 0] + sin_o*points[:, 1]
    robot_points_oriented[:, 1] = -sin_o*points[:, 0] + cos_o*points[:, 1]
    ax.add_patch(Polygon(position+robot_points_oriented, color=color))


def initial_setup(playzone: Playzone):
    playzone.POIs[0] = np.array([0, -500, -1000+35])
    playzone.POIs[1] = np.array([0, -570, -1000+35])
    playzone.POIs[2] = np.array([0, -430, -1000+35])
    playzone.POIs[3] = np.array([0, -465, -1000+2*1.4*35])
    playzone.POIs[4] = np.array([0, -535, -1000+2*1.4*35])

    playzone.POIs[5] = np.array([0, 500, -1000+35])
    playzone.POIs[6] = np.array([0, 570, -1000+35])
    playzone.POIs[7] = np.array([0, 430, -1000+35])
    playzone.POIs[8] = np.array([0, 465, -1000+2*1.4*35])
    playzone.POIs[9] = np.array([0, 535, -1000+2*1.4*35])

    playzone.POIs[10] = np.array([0, -1500+35, -320])
    playzone.POIs[11] = np.array([0, -1500+35, -390])
    playzone.POIs[12] = np.array([0, -1500+35, -460])
    playzone.POIs[13] = np.array([0, -1500+2*1.4*35, -355])
    playzone.POIs[14] = np.array([0, -1500+2*1.4*35, -425])

    playzone.POIs[15] = np.array([0, -1500+35, 320])
    playzone.POIs[16] = np.array([0, -1500+35, 390])
    playzone.POIs[17] = np.array([0, -1500+35, 460])
    playzone.POIs[18] = np.array([0, -1500+2*1.4*35, 355])
    playzone.POIs[19] = np.array([0, -1500+2*1.4*35, 425])

    playzone.POIs[20] = np.array([0, 1500-35, -320])
    playzone.POIs[21] = np.array([0, 1500-35, -390])
    playzone.POIs[22] = np.array([0, 1500-35, -460])
    playzone.POIs[23] = np.array([0, 1500-2*1.4*35, -355])
    playzone.POIs[24] = np.array([0, 1500-2*1.4*35, -425])

    playzone.POIs[25] = np.array([0, 1500-35, 320])
    playzone.POIs[26] = np.array([0, 1500-35, 390])
    playzone.POIs[27] = np.array([0, 1500-35, 460])
    playzone.POIs[28] = np.array([0, 1500-2*1.4*35, 355])
    playzone.POIs[29] = np.array([0, 1500-2*1.4*35, 425])


def end_loop():
    plt.pause(1/60)
    # Exit if all windows are closed, TO REMOVE IN PROD
    if len(plt.get_figlabels()) <= 0:
        return True
    return False
