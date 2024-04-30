import random
from testing import TestingPlayzone, end_loop, initial_setup, TestingRobot, ZONES
from utils import normalize
import numpy as np
import matplotlib.pyplot as plt
import time

# Ordres de priorités
# 
# Actions
# 1) Orientation de panneaux solaires
# 2) Dépôt de pots si présents
# 3) Dépôt de plantes si présentes
# 4) Récupération de plante si pot présent
# 5) Récupération de pot si plus de pot
#
# Zones privilégiées
# 1) Zone opposée à celle du robot adverse
# 2) Jardinières adverses

def get_state(t: float) -> str:
    if t > 100:
        return "end"
    if t > 90:
        return "pami"
    if t > 0:
        return "match"

SIDE = 0

if __name__ == "__main__":
    print("Starting strategy...")
    running = True
    plt.ion()
    playzone = TestingPlayzone()
    robot = TestingRobot(SIDE)
    initial_setup(playzone)
    robot.position_to_start()
    start_time = time.time()
    t = start_time
    pick_target = playzone.POIs[random.randrange(0, 30), 1:]
    j = 0
    while running:
        dt = min(time.time()-t, 1/20)
        t = time.time()
        if robot._state == "idle":
            robot._state = "take_pot"
        elif robot._state=="take_pot":
            distance = np.linalg.norm(robot._target-robot.position)
            if distance > 100:
                robot._action = "move_to_target"
            else:
                robot._action = "take"
        elif robot._state == "drop_planter":
            distance = np.linalg.norm(robot._target-robot.position)
            if distance > 100:
                robot._action = "move_to_target"
            else:
                robot._action = "drop"
        if robot._action == "take":
            pick_status = robot.pick(pick_target, playzone)
            if pick_status==-2 or robot.is_full(): # Robot slots are full
                robot._state = "drop_planter"
                zones = ZONES[ZONES[:,0]==SIDE]
                zone = zones[random.randrange(0, len(zones))]
                robot.goto(np.array([(zone[1]+zone[3])/2, (zone[2]+zone[4])/2]))
                print("[INFO] Robot is full and will drop pots and plants")
            else:
                robot._state = "take_pot"
                pick_target = playzone.POIs[random.randrange(0, 30), 1:]
                u = normalize(pick_target-robot.position)
                robot.goto(pick_target-170*u)
        elif robot._action == "drop":
            drop_status = robot.drop_all(playzone)
            robot._state = "take_pot"


        robot.move(dt)
        playzone.resolve_collisions(robot)
        playzone.show(robot)

        plt.scatter(robot._target[0], robot._target[1], marker="x")

        if end_loop():
            break
