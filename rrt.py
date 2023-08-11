"""
Path planning Sample Code with Randomized Rapidly-Exploring Random Trees (RRT)
author: AtsushiSakai(@Atsushi_twi)
"""

import math
import random

import matplotlib.pyplot as plt
import numpy as np
import time
import pandas as pd

show_animation = True


class RRT:
    """
    Class for RRT planning
    """

    class Node:
        """
        RRT Node
        """

        def __init__(self, x, y):
            self.x = x
            self.y = y
            self.path_x = []
            self.path_y = []
            self.parent = None

    class AreaBounds:

        def __init__(self, area):
            self.xmin = float(area[0])
            self.xmax = float(area[1])
            self.ymin = float(area[2])
            self.ymax = float(area[3])


    def __init__(self,
                 start,
                 goal,
                 obstacle_list,
                 rand_area,
                 expand_dis=3.0,
                 path_resolution=0.1,
                 goal_sample_rate=5,
                 max_iter=1000,
                 play_area=None,
                 robot_radius=0.0,
                 ):
        """
        Setting Parameter
        start:Start Position [x,y]
        goal:Goal Position [x,y]
        obstacleList:obstacle Positions [[x,y,size],...]
        randArea:Random Sampling Area [min,max]
        play_area:stay inside this area [xmin,xmax,ymin,ymax]
        robot_radius: robot body modeled as circle with given radius
        """
        self.start = self.Node(start[0], start[1])
        self.end = self.Node(goal[0], goal[1])
        self.min_rand = rand_area[0]
        self.max_rand = rand_area[1]
        if play_area is not None:
            self.play_area = self.AreaBounds(play_area)
        else:
            self.play_area = None
        self.expand_dis = expand_dis
        self.path_resolution = path_resolution
        self.goal_sample_rate = goal_sample_rate
        self.max_iter = max_iter
        self.obstacle_list = obstacle_list
        self.node_list = []
        self.robot_radius = robot_radius

    def planning(self, animation, ax):
        """
        rrt path planning
        animation: flag for animation on or off
        """

        self.node_list = [self.start]
        for i in range(self.max_iter):
            rnd_node = self.get_random_node()
            nearest_ind = self.get_nearest_node_index(self.node_list, rnd_node)
            nearest_node = self.node_list[nearest_ind]

            new_node = self.steer(nearest_node, rnd_node, self.expand_dis)

            if self.check_if_outside_play_area(new_node, self.play_area) and \
               self.check_collision(
                   new_node, self.obstacle_list, self.robot_radius):
                self.node_list.append(new_node)

            if animation and i % 5 == 0:
                self.draw_graph(ax, rnd_node)

            if self.calc_dist_to_goal(self.node_list[-1].x,
                                      self.node_list[-1].y) <= self.expand_dis:
                final_node = self.steer(self.node_list[-1], self.end,
                                        self.expand_dis)
                if self.check_collision(
                        final_node, self.obstacle_list, self.robot_radius):
                    return self.generate_final_course(len(self.node_list) - 1)

            if animation and i % 5:
                self.draw_graph(ax, rnd_node)

        return None  # cannot find path

    def steer(self, from_node, to_node, extend_length=float("inf")):

        new_node = self.Node(from_node.x, from_node.y)
        d, theta = self.calc_distance_and_angle(new_node, to_node)

        new_node.path_x = [new_node.x]
        new_node.path_y = [new_node.y]

        if extend_length > d:
            extend_length = d

        n_expand = math.floor(extend_length / self.path_resolution)

        for _ in range(n_expand):
            new_node.x += self.path_resolution * math.cos(theta)
            new_node.y += self.path_resolution * math.sin(theta)
            new_node.path_x.append(new_node.x)
            new_node.path_y.append(new_node.y)

        d, _ = self.calc_distance_and_angle(new_node, to_node)
        if d <= self.path_resolution:
            new_node.path_x.append(to_node.x)
            new_node.path_y.append(to_node.y)
            new_node.x = to_node.x
            new_node.y = to_node.y

        new_node.parent = from_node

        return new_node

    def generate_final_course(self, goal_ind):
        path = [[self.end.x, self.end.y]]
        node = self.node_list[goal_ind]
        while node.parent is not None:
            path.append([node.x, node.y])
            node = node.parent
        path.append([node.x, node.y])

        return path

    def calc_dist_to_goal(self, x, y):
        dx = x - self.end.x
        dy = y - self.end.y
        return math.hypot(dx, dy)

    def get_random_node(self):
        if random.randint(0, 100) > self.goal_sample_rate:
            rnd = self.Node(
                random.uniform(self.min_rand, self.max_rand),
                random.uniform(self.min_rand, self.max_rand))
        else:  # goal point sampling
            rnd = self.Node(self.end.x, self.end.y)
        return rnd

    def draw_graph(self, ax, rnd=None):
        ax.cla()
        # for stopping simulation with the esc key.
        plt.gcf().canvas.mpl_connect(
            'key_release_event',
            lambda event: [exit(0) if event.key == 'escape' else None])
        if rnd is not None:
            ax.plot(rnd.x, rnd.y, "^b")
            if self.robot_radius > 0.0:
                self.plot_circle(rnd.x, rnd.y, self.robot_radius, ax, '-w')
        for node in self.node_list:
            if node.parent:
                ax.plot(node.path_x, node.path_y, "-c")

        for (ox, oy, size) in self.obstacle_list:
            self.plot_circle(ox, oy, size, ax, '-k')

        if self.play_area is not None:
            ax.plot([self.play_area.xmin, self.play_area.xmax,
                      self.play_area.xmax, self.play_area.xmin,
                      self.play_area.xmin],
                     [self.play_area.ymin, self.play_area.ymin,
                      self.play_area.ymax, self.play_area.ymax,
                      self.play_area.ymin],
                     "-k")

        ax.plot(self.start.x, self.start.y, 'orangered', marker='o', markersize=10, label='AGV Initial Position')
        ax.plot(self.end.x, self.end.y, 'gold', marker='^', markersize=10, label='Goal Position')
        # plt.axis("equal")
        # plt.axis([-11, 11, -11, 11])
        ax.grid(color = 'black', linestyle = ':', which='both')
        plt.pause(0.01)

    @staticmethod
    def plot_circle(x, y, size, ax, color="-b"):  # pragma: no cover
        deg = list(range(0, 360, 5))
        deg.append(0)
        xl = [x + size * math.cos(np.deg2rad(d)) for d in deg]
        yl = [y + size * math.sin(np.deg2rad(d)) for d in deg]
        ax.plot(xl, yl, color)
    
    @staticmethod
    def plot_rectangle(x, y, length, width, ax, color="-b"):
        ox, oy = [], []
        for k in np.arange(x-length/2,x+length/2+0.01, 0.01):
            ox.append(k)
            ox.append(k)
            oy.append(y + width/2)
            oy.append(y - width/2)
        for k in np.arange(y-width/2,y+width/2+0.01, 0.01):
            ox.append(x + length/2)
            ox.append(x - length/2)
            oy.append(k)
            oy.append(k)
        ax.plot(ox, oy, ".k", markersize=0.5, alpha=0.75)


    @staticmethod
    def get_nearest_node_index(node_list, rnd_node):
        dlist = [(node.x - rnd_node.x)**2 + (node.y - rnd_node.y)**2
                 for node in node_list]
        minind = dlist.index(min(dlist))

        return minind

    @staticmethod
    def check_if_outside_play_area(node, play_area):

        if play_area is None:
            return True  # no play_area was defined, every pos should be ok

        if node.x < play_area.xmin or node.x > play_area.xmax or \
           node.y < play_area.ymin or node.y > play_area.ymax:
            return False  # outside - bad
        else:
            return True  # inside - ok

    @staticmethod
    def check_collision(node, obstacleList, robot_radius):

        if node is None:
            return False

        for (ox, oy, size) in obstacleList:
            dx_list = [ox - x for x in node.path_x]
            dy_list = [oy - y for y in node.path_y]
            d_list = [dx * dx + dy * dy for (dx, dy) in zip(dx_list, dy_list)]

            if min(d_list) <= (size+robot_radius)**2:
                return False  # collision

        # for (ox, oy, length, width) in obstacleList:
        #     dx_list = [ox - x for x in node.path_x]
        #     dy_list = [oy - y for y in node.path_y]
        #     d_list = [dx * dx + dy * dy for (dx, dy) in zip(dx_list, dy_list)]

        #     if (min(d_list) <= (length/2)**2 and min(d_list) <= (width/2)**2):
        #         return False  # collision

        return True  # safe

    @staticmethod
    def calc_distance_and_angle(from_node, to_node):
        dx = to_node.x - from_node.x
        dy = to_node.y - from_node.y
        d = math.hypot(dx, dy)
        theta = math.atan2(dy, dx)
        return d, theta

    @staticmethod
    def get_path_length(path):
        le = 0
        for i in range(len(path) - 1):
            dx = path[i + 1][0] - path[i][0]
            dy = path[i + 1][1] - path[i][1]
            d = math.hypot(dx, dy)
            le += d

        return le
    
    @staticmethod
    def get_target_point(path, targetL):
        le = 0
        ti = 0
        lastPairLen = 0
        for i in range(len(path) - 1):
            dx = path[i + 1][0] - path[i][0]
            dy = path[i + 1][1] - path[i][1]
            d = math.hypot(dx, dy)
            le += d
            if le >= targetL:
                ti = i - 1
                lastPairLen = d
                break

        partRatio = (le - targetL) / lastPairLen

        x = path[ti][0] + (path[ti + 1][0] - path[ti][0]) * partRatio
        y = path[ti][1] + (path[ti + 1][1] - path[ti][1]) * partRatio

        return [x, y, ti]
    
    @staticmethod
    def line_collision_check(first, second, obstacleList):
        # Line Equation

        x1 = first[0]
        y1 = first[1]
        x2 = second[0]
        y2 = second[1]

        try:
            a = y2 - y1
            b = -(x2 - x1)
            c = y2 * (x2 - x1) - x2 * (y2 - y1)
        except ZeroDivisionError:
            return False

        for (ox, oy, size) in obstacleList:
            d = abs(a * ox + b * oy + c) / (math.hypot(a, b))
            if d <= size:
                return False

        return True  # OK
    
    def path_smoothing(self, path, max_iter, obstacle_list):
        le = self.get_path_length(path)

        for i in range(max_iter):
            # Sample two points
            pickPoints = [random.uniform(0, le), random.uniform(0, le)]
            pickPoints.sort()
            first = self.get_target_point(path, pickPoints[0])
            second = self.get_target_point(path, pickPoints[1])

            if first[2] <= 0 or second[2] <= 0:
                continue

            if (second[2] + 1) > len(path):
                continue

            if second[2] == first[2]:
                continue

            # collision check
            if not self.line_collision_check(first, second, obstacle_list):
                continue

            # Create New path
            newPath = []
            newPath.extend(path[:first[2] + 1])
            newPath.append([first[0], first[1]])
            newPath.append([second[0], second[1]])
            newPath.extend(path[second[2] + 1:])
            path = newPath
            le = self.get_path_length(path)

        return path
    

def main(sx,sy,gx,gy,obstacle_list,robot_radius,i):
    traj_df = pd.DataFrame({'X':[], 'Y':[]})

    px = 1/plt.rcParams['figure.dpi']  # pixel in inches
    fig = plt.figure(figsize=(1080*px, 1150*px))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlim([-11, 11])
    ax.set_ylim([-11, 11])
       
    # Set Initial parameters
    rrt = RRT(
        start=[sx, sy],
        goal=[gx, gy],
        rand_area=[-10, 10],
        obstacle_list=obstacle_list,
        play_area=[-10,10,-10,10],
        robot_radius=robot_radius
        )
    start_time = time.localtime(time.time())
    path = rrt.planning(show_animation, ax)

    end_time = time.localtime(time.time())

    traj_len = 0

    if path is None:
        print("Cannot find path")
        # path[0] = None
        if show_animation:
            ax.grid(color = 'black', linestyle = ':', which='both')
            plt.pause(0.01)  # Need for Mac
            plt.savefig(f'RRT/Outdoor_0_05/rrt_g{i+1}.png')
            plt.show()
        return [0,0], start_time, end_time, traj_len
    
    else:
        print("found path!!")

        for k in range(0,len(path)-1):
            x0 = path[k][0]
            y0 = path[k][1]
            x1 = path[k+1][0]
            y1 = path[k+1][1]
            # temp = math.sqrt((x1-x0)**2 + (y1-y0)**2)
            temp = math.hypot(x1-x0, y1-y0)
            traj_len = traj_len + temp
            x0 = x1
            y0 = y1

        for k in range(0,len(path)):
            traj_temp = pd.DataFrame({'X':[path[k][0]], 'Y':[path[k][1]]})
            traj_df = pd.concat([traj_df, traj_temp])
        traj_df.to_csv(f'RRT/Outdoor_0_05/rrt_trajectory{i+1}.csv', index=False)
        print('RRT csv saved')

        # Draw final path
        if show_animation:
            rrt.draw_graph(ax)
            im = plt.imread('C:/D/ABHYAAS/Coding/imagesx10/env1.png')
            implot = plt.imshow(im, extent=(-11, 11, -11, 11))
            # ax.plot([x for (x, y) in path], [y for (x, y) in path], 'yellow', linestyle = '-')
            ax.plot([x for (x, y) in path], [y for (x, y) in path], 'yellow', linestyle = '-')
            ax.grid(color = 'black', linestyle = ':', which='both')
            # # Set the font size for x tick labels
            plt.rc('xtick', labelsize=25)
            # # Set the font size for y tick labels
            plt.rc('ytick', labelsize=25)
            plt.legend(loc='lower center', bbox_to_anchor=(0.5,-0.15), fontsize=15, ncol=2)
            plt.pause(0.01)  # Need for Mac
            plt.savefig(f'RRT/Outdoor_0_05/rrt_g{i+1}.png')
            plt.show()
        
        return path[0], start_time, end_time, traj_len


if __name__ == '__main__':
    main()
