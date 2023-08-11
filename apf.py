"""
Potential Field based path planner
author: Atsushi Sakai (@Atsushi_twi)
Ref:
https://www.cs.cmu.edu/~motionplanning/lecture/Chap4-Potential-Field_howie.pdf
"""

from collections import deque
import numpy as np
import matplotlib.pyplot as plt
import time
import pandas as pd

# Parameters
KP = 5.0  # attractive potential gain
ETA = 85.0  # repulsive potential gain
AREA_WIDTH = 20.0  # potential area width [m]
# the number of previous positions used to check oscillations
OSCILLATIONS_DETECTION_LENGTH = 3

show_animation = True


def calc_potential_field(gx, gy, ox, oy, size, reso, rr, sx, sy):
    minx = -10 # min(min(ox), sx, gx) - AREA_WIDTH / 2.0
    # print(minx)
    miny = -10 # min(min(oy), sy, gy) - AREA_WIDTH / 2.0
    # print(miny)
    maxx = 10 # max(max(ox), sx, gx) + AREA_WIDTH / 2.0
    maxy = 10 # max(max(oy), sy, gy) + AREA_WIDTH / 2.0
    xw = int(round((maxx - minx) / reso))
    yw = int(round((maxy - miny) / reso))

    # calc each potential
    pmap = [[0.0 for i in range(yw)] for i in range(xw)]
    # print(pmap)

    for ix in range(xw):
        x = ix * reso + minx

        for iy in range(yw):
            y = iy * reso + miny
            ug = calc_attractive_potential(x, y, gx, gy)
            uo = calc_repulsive_potential(x, y, ox, oy, size, rr)
            uf = ug + uo
            pmap[ix][iy] = uf

    return pmap, minx, miny


def calc_attractive_potential(x, y, gx, gy):
    return 0.5 * KP * (np.hypot(x - gx, y - gy))**2


def calc_repulsive_potential(x, y, ox, oy, size, rr):
    # search nearest obstacle
    minid = -1
    dmin = float("inf")
    for i, _ in enumerate(ox):
        d = np.hypot(x - ox[i], y - oy[i])
        if dmin >= d:
            dmin = d
            minid = i

    # calc repulsive potential
    dq = np.hypot(x - ox[minid], y - oy[minid])

    if dq <= (2*rr): #np.sqrt(rr**2+size**2):
        if dq <= 0.1:
            dq = 0.1

        return 0.5 * ETA * (1.0 / dq - 1.0 / (2*rr)) ** 2
    else:
        return 0.0


def get_motion_model():
    # dx, dy
    motion = [[1, 0],
              [0, 1],
              [-1, 0],
              [0, -1],
              [-1, -1],
              [-1, 1],
              [1, -1],
              [1, 1]]

    return motion


def oscillations_detection(previous_ids, ix, iy):
    previous_ids.append((ix, iy))

    if (len(previous_ids) > OSCILLATIONS_DETECTION_LENGTH):
        previous_ids.popleft()

    # check if contains any duplicates by copying into a set
    previous_ids_set = set()
    for index in previous_ids:
        if index in previous_ids_set:
            return True
        else:
            previous_ids_set.add(index)
    return False


def potential_field_planning(sx, sy, gx, gy, ox, oy, size, reso, rr, ax):

    # calc potential field
    pmap, minx, miny = calc_potential_field(gx, gy, ox, oy, size, reso, rr, sx, sy)
    # print(pmap)

    # search path
    d = np.hypot(sx - gx, sy - gy)
    ix = round((sx - minx) / reso)
    iy = round((sy - miny) / reso)
    gix = round((gx - minx) / reso)
    giy = round((gy - miny) / reso)

    if show_animation:
        # draw_heatmap(pmap, ax, reso, miny)
        # for stopping simulation with the esc key.
        plt.gcf().canvas.mpl_connect('key_release_event',
                lambda event: [exit(0) if event.key == 'escape' else None])
        # ax.plot(ix, iy, 'orangered', marker='o', markersize=10, label='AGV Initial Position')
        # ax.plot(gix, giy, 'gold', marker='^', markersize=10, label='Goal Position')

    rx, ry = [sx], [sy]
    motion = get_motion_model()
    previous_ids = deque()

    while d >= reso:
        minp = float("inf")
        minix, miniy = -1, -1
        for i, _ in enumerate(motion):
            inx = int(ix + motion[i][0])
            iny = int(iy + motion[i][1])
            if inx >= len(pmap) or iny >= len(pmap[0]) or inx < -10 or iny < -10:
                p = float("inf")  # outside area
                print("outside potential!")
            else:
                p = pmap[inx][iny]
            if minp > p:
                minp = p
                minix = inx
                miniy = iny
        ix = minix
        iy = miniy
        xp = ix * reso + minx
        yp = iy * reso + miny
        d = np.hypot(gx - xp, gy - yp)
        rx.append(xp)
        ry.append(yp)

        if (oscillations_detection(previous_ids, ix, iy)):
            print("Oscillation detected at ({},{})!".format(ix, iy))
            break

        # if show_animation:
        #     ax.plot(ix, iy, ".r")
        #     plt.pause(0.01)

    print("Goal!!")

    return rx, ry


def draw_heatmap(data, ax, reso, miny):
    data = np.array(data).T
    for i in range(len(data)):
        ih = data[i]
        for j in range(len(ih)):
            jv = ih[j]
            jv = jv*reso + miny
            #ih[j] = jv
            data[i][j] = jv
    # print(data)
    ax.pcolor(data, vmax=20.0, cmap=plt.cm.Blues)


def main(sx,sy,gx,gy,obstacle_list,robot_radius, i):
    print("potential_field_planning start")

    grid_size = 0.1  # potential grid size [m]

    traj_df = pd.DataFrame({'X':[], 'Y':[]})

    px = 1/plt.rcParams['figure.dpi']  # pixel in inches
    fig = plt.figure(figsize=(1080*px, 1150*px))
    ax = fig.add_subplot(1, 1, 1)
    # ax.set_position([0, 0, 1080*px, 1080*px])
    im = plt.imread('C:/D/ABHYAAS/Coding/imagesx10/env1.png')
    implot = plt.imshow(im, extent=(-11, 11, -11, 11))
    ax.set_xlim([-11, 11])
    ax.set_ylim([-11, 11])

    ox, oy, size  = [], [], []
    for j in range(0,len(obstacle_list)):
        x = obstacle_list[j][0]
        y = obstacle_list[j][1]
        R = obstacle_list[j][2]
        size.append(obstacle_list[j][2])
        for theta in range(0,361):
            ox.append(x + R*np.cos(theta*np.pi/180))
            oy.append(y + R*np.sin(theta*np.pi/180))
        

    # path generation
    start_time = time.localtime(time.time())
    rx, ry = potential_field_planning(sx, sy, gx, gy, ox, oy, size, grid_size, robot_radius, ax)
    end_time = time.localtime(time.time())

    traj_len = 0
    if len(rx) >= 1:
        for k in range(0,len(rx)-1):
            x0 = rx[k]
            y0 = ry[k]
            x1 = rx[k+1]
            y1 = ry[k+1]
            # temp = math.sqrt((x1-x0)**2 + (y1-y0)**2)
            temp = np.hypot(x1-x0, y1-y0)
            traj_len = traj_len + temp
            x0 = x1
            y0 = y1

        for k in range(0,len(rx)):
            traj_temp = pd.DataFrame({'X':[rx[k]], 'Y':[ry[k]]})
            traj_df = pd.concat([traj_df, traj_temp])
    traj_df.to_csv(f'APF/Outdoor_0_05/apf_trajectory{i+1}.csv', index=False)
    print('APF csv saved')

    if show_animation:
        ax.plot(sx, sy, 'orangered', marker='o', markersize=10, label='AGV Initial Position')
        ax.plot(gx, gy, 'gold', marker='^', markersize=10, label='Goal Position')
        ax.plot(ox, oy, ".k", markersize=0.5, alpha=0.75)
        ax.plot(rx, ry, 'yellow', linestyle = '-')
        ax.grid(color = 'black', linestyle = ':', which='both')
        # # Set the font size for x tick labels
        plt.rc('xtick', labelsize=25)
        # # Set the font size for y tick labels
        plt.rc('ytick', labelsize=25)
        plt.legend(loc='lower center', bbox_to_anchor=(0.5,-0.15), fontsize=15, ncol=2)
        # plt.axis("equal")
        # plt.pause(0.01)
        plt.savefig(f'APF/Outdoor_0_05/apf_g{i+1}.png')
        plt.show()

    return rx[0], ry[0], start_time, end_time, traj_len


if __name__ == '__main__':
    print(__file__ + " start!!")
    main()
    print(__file__ + " Done!!")
