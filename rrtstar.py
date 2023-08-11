"""
Path planning Sample Code with RRT*
author: Atsushi Sakai(@Atsushi_twi)
"""
"""
Additional changes have been made to the original code for the purpose of this project
author: Apoorva Khairnar
"""

import math
import sys
import matplotlib.pyplot as plt
import pathlib
import pandas as pd
import time
sys.path.append(str(pathlib.Path(__file__).parent.parent))

from rrt import RRT

show_animation = True


class RRTStar(RRT):
    """
    Class for RRT Star planning
    """

    class Node(RRT.Node):
        def __init__(self, x, y):
            super().__init__(x, y)
            self.cost = 0.0

    def __init__(self,
                 start,
                 goal,
                 obstacle_list,
                 rand_area,
                 expand_dis=10.0,
                 path_resolution=0.1,
                 goal_sample_rate=20,
                 max_iter=1000,
                 connect_circle_dist=50.0,
                 search_until_max_iter=False,
                 robot_radius=0.0):
        """
        Setting Parameter
        start:Start Position [x,y]
        goal:Goal Position [x,y]
        obstacleList:obstacle Positions [[x,y,size],...]
        randArea:Random Sampling Area [min,max]
        """
        super().__init__(start, goal, obstacle_list, rand_area, expand_dis,
                         path_resolution, goal_sample_rate, max_iter,
                         robot_radius=robot_radius)
        self.connect_circle_dist = connect_circle_dist
        self.goal_node = self.Node(goal[0], goal[1])
        self.search_until_max_iter = search_until_max_iter
        self.node_list = []

    def planning(self, animation, ax):
        """
        rrt star path planning
        animation: flag for animation on or off .
        """

        self.node_list = [self.start]
        for i in range(self.max_iter):
            print("Iter:", i, ", number of nodes:", len(self.node_list))
            rnd = self.get_random_node()
            nearest_ind = self.get_nearest_node_index(self.node_list, rnd)
            new_node = self.steer(self.node_list[nearest_ind], rnd,
                                  self.expand_dis)
            near_node = self.node_list[nearest_ind]
            new_node.cost = near_node.cost + \
                math.hypot(new_node.x-near_node.x,
                           new_node.y-near_node.y)

            if self.check_collision(
                    new_node, self.obstacle_list, self.robot_radius):
                near_inds = self.find_near_nodes(new_node)
                node_with_updated_parent = self.choose_parent(
                    new_node, near_inds)
                if node_with_updated_parent:
                    self.rewire(node_with_updated_parent, near_inds)
                    self.node_list.append(node_with_updated_parent)
                else:
                    self.node_list.append(new_node)

            if animation:
                self.draw_graph(ax, rnd)

            if ((not self.search_until_max_iter)
                    and new_node):  # if reaches goal
                last_index = self.search_best_goal_node()
                if last_index is not None:
                    return self.generate_final_course(last_index)

        print("reached max iteration")

        last_index = self.search_best_goal_node()
        if last_index is not None:
            return self.generate_final_course(last_index)

        return None

    def choose_parent(self, new_node, near_inds):
        """
        Computes the cheapest point to new_node contained in the list
        near_inds and set such a node as the parent of new_node.
            Arguments:
            --------
                new_node, Node
                    randomly generated node with a path from its neared point
                    There are not coalitions between this node and th tree.
                near_inds: list
                    Indices of indices of the nodes what are near to new_node
            Returns.
            ------
                Node, a copy of new_node
        """
        if not near_inds:
            return None

        # search nearest cost in near_inds
        costs = []
        for i in near_inds:
            near_node = self.node_list[i]
            t_node = self.steer(near_node, new_node)
            if t_node and self.check_collision(
                    t_node, self.obstacle_list, self.robot_radius):
                costs.append(self.calc_new_cost(near_node, new_node))
            else:
                costs.append(float("inf"))  # the cost of collision node
        min_cost = min(costs)

        if min_cost == float("inf"):
            print("There is no good path.(min_cost is inf)")
            return None

        min_ind = near_inds[costs.index(min_cost)]
        new_node = self.steer(self.node_list[min_ind], new_node)
        new_node.cost = min_cost

        return new_node

    def search_best_goal_node(self):
        dist_to_goal_list = [
            self.calc_dist_to_goal(n.x, n.y) for n in self.node_list
        ]
        goal_inds = [
            dist_to_goal_list.index(i) for i in dist_to_goal_list
            if i <= self.expand_dis
        ]

        safe_goal_inds = []
        for goal_ind in goal_inds:
            t_node = self.steer(self.node_list[goal_ind], self.goal_node)
            if self.check_collision(
                    t_node, self.obstacle_list, self.robot_radius):
                safe_goal_inds.append(goal_ind)

        if not safe_goal_inds:
            return None

        min_cost = min([self.node_list[i].cost for i in safe_goal_inds])
        for i in safe_goal_inds:
            if self.node_list[i].cost == min_cost:
                return i

        return None

    def find_near_nodes(self, new_node):
        """
        1) defines a ball centered on new_node
        2) Returns all nodes of the three that are inside this ball
            Arguments:
            ---------
                new_node: Node
                    new randomly generated node, without collisions between
                    its nearest node
            Returns:
            -------
                list
                    List with the indices of the nodes inside the ball of
                    radius r
        """
        nnode = len(self.node_list) + 1
        r = self.connect_circle_dist * math.sqrt(math.log(nnode) / nnode)
        # if expand_dist exists, search vertices in a range no more than
        # expand_dist
        if hasattr(self, 'expand_dis'):
            r = min(r, self.expand_dis)
        dist_list = [(node.x - new_node.x)**2 + (node.y - new_node.y)**2
                     for node in self.node_list]
        near_inds = [dist_list.index(i) for i in dist_list if i <= r**2]
        return near_inds

    def rewire(self, new_node, near_inds):
        """
            For each node in near_inds, this will check if it is cheaper to
            arrive to them from new_node.
            In such a case, this will re-assign the parent of the nodes in
            near_inds to new_node.
            Parameters:
            ----------
                new_node, Node
                    Node randomly added which can be joined to the tree
                near_inds, list of uints
                    A list of indices of the self.new_node which contains
                    nodes within a circle of a given radius.
            Remark: parent is designated in choose_parent.
        """
        for i in near_inds:
            near_node = self.node_list[i]
            edge_node = self.steer(new_node, near_node)
            if not edge_node:
                continue
            edge_node.cost = self.calc_new_cost(new_node, near_node)

            no_collision = self.check_collision(
                edge_node, self.obstacle_list, self.robot_radius)
            improved_cost = near_node.cost > edge_node.cost

            if no_collision and improved_cost:
                near_node.x = edge_node.x
                near_node.y = edge_node.y
                near_node.cost = edge_node.cost
                near_node.path_x = edge_node.path_x
                near_node.path_y = edge_node.path_y
                near_node.parent = edge_node.parent
                self.propagate_cost_to_leaves(new_node)

    def calc_new_cost(self, from_node, to_node):
        d, _ = self.calc_distance_and_angle(from_node, to_node)
        return from_node.cost + d

    def propagate_cost_to_leaves(self, parent_node):

        for node in self.node_list:
            if node.parent == parent_node:
                node.cost = self.calc_new_cost(parent_node, node)
                self.propagate_cost_to_leaves(node)


def main(sx,sy,gx,gy,obstacle_list,robot_radius,i):
    traj_df = pd.DataFrame({'X':[], 'Y':[]})

    px = 1/plt.rcParams['figure.dpi']  # pixel in inches
    fig = plt.figure(figsize=(1080*px, 1150*px))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlim([-11, 11])
    ax.set_ylim([-11, 11])
    
    # Set Initial parameters
    rrt_star = RRTStar(
        start=[sx, sy],
        goal=[gx, gy],
        rand_area=[-10,10],
        obstacle_list=obstacle_list,
        expand_dis=1,
        robot_radius=0.5)
    
    start_time = time.localtime(time.time())
    path = rrt_star.planning(show_animation, ax)
    end_time = time.localtime(time.time())

    traj_len = 0

    if path is None:
        print("Cannot find path")
        # path[0] = None
        if show_animation:
            ax.grid(color = 'black', linestyle = ':', which='both')
            plt.savefig(f'RRTstar/Outdoor_0_05/rrtstar_g{i+1}.png')
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
        traj_df.to_csv(f'RRTstar/Outdoor_0_05/rrtstar_trajectory{i+1}.csv', index=False)
        print('RRT* csv saved')

        # Draw final path
        if show_animation:
            rrt_star.draw_graph(ax)
            im = plt.imread('C:/D/ABHYAAS/Coding/imagesx10/env1.png')
            implot = plt.imshow(im, extent=(-11, 11, -11, 11))
            ax.plot([x for (x, y) in path], [y for (x, y) in path], 'yellow', linestyle = '-')
            ax.grid(color = 'black', linestyle = ':', which='both')
            # # Set the font size for x tick labels
            plt.rc('xtick', labelsize=25)
            # # Set the font size for y tick labels
            plt.rc('ytick', labelsize=25)
            plt.legend(loc='lower center', bbox_to_anchor=(0.5,-0.15), fontsize=15, ncol=2)
            plt.savefig(f'RRTstar/Outdoor_0_05/rrtstar_g{i+1}.png')
            plt.show()

        return path[0], start_time, end_time, traj_len


if __name__ == '__main__':
    main()
