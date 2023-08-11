import math

import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
import time
import pandas as pd
# import labWorld

from scipy.optimize import minimize

LEN = 0.5602 #Length of the vehicle
WIDTH = 0.254 #Width of the vehicle
WB = 0.3302 #Wheelbase of the Vehicle
TW = 0.254 #Trackwidth of the vehicle

class pRRT:
    class Car:
        def __init__(self, state, n, radius, dist, centers, u_prev):
            self.state = state
            self.n = n
            self.radius = radius
            self.dist = dist
            self.centers = centers
            self.u_prev = u_prev
            self.path_x = []
            self.path_y = []
            self.parent = None
            self.calCenter()
            # self.calBoundaries()

        def update(self, state, u):
            self.state = state
            self.u_prev = u
            self.calCenter()

        def stateupdate(self, state):
            self.state[2] = state[2]
            self.state[3] = state[3]
            self.calCenter()

        def calCenter(self):
            position = self.state[0:2]
            heading = self.state[2]
            if self.n == 1:
                self.centers = position
            else:
                R = [[np.cos(heading), -np.sin(heading)],[np.sin(heading), np.cos(heading)]]
                self.centers = [list(np.transpose(np.matmul(R,[[-self.dist],[0]]))+position),position,list(np.transpose(np.matmul(R,[[self.dist],[0]]))+position)]
                self.centers[0] = list(self.centers[0][0])
                self.centers[2] = list(self.centers[2][0])

        def calBoundaries(self):
            centers = self.centers
            rad = self.radius
            c1 = centers[0]
            c2 = centers[1]
            c3 = centers[2]
            vehicleBound = np.vstack([[c1[0] + rad, c1[1]],[c1[0]- rad, c1[1]],[c1[0], c1[1] + rad],[c1[0], c1[1] - rad],[c2[0] + rad, c2[1]],[c2[0]- rad, c2[1]],[c2[0], c2[1] + rad],[c2[0], c2[1] - rad],[c3[0] + rad, c3[1]],[c3[0]- rad, c3[1]],[c3[0], c3[1] + rad],[c3[0], c3[1] - rad]])
            return vehicleBound

        def draw(self, ax):
            positionx = self.state[0]
            positiony = self.state[1]
            heading = self.state[2]

            ax.plot(positionx, positiony, 'orangered', marker='o', markersize=10, label='AGV Initial Position')

    def __init__(self,
                 start,
                 goal,
                 obstacle_list,
                 x_val,
                 y_val,
                 pdfMap,
                 max_iter=100,
                 ):
        """
        Setting Parameter

        start:Start Position [x,y]
        goal:Goal Position [x,y]
        obstacleList:obstacle Positions [[x,y,size],...]
        randArea:Random Sampling Area [min,max]
        play_area:stay inside this area [xmin,xmax,ymin,ymax]

        """
        self.start = start
        self.goal = goal
        self.obstacle_list = obstacle_list
        self.x_val = x_val
        self.y_val = y_val
        self.pdfMap = pdfMap
        self.max_iter = max_iter
        self.node_list = []
    
    def planning(self):
        self.node_list = [self.start]
        n = 0
        terminate_flag = False
        states = []
        controls = []
        for i in range(self.max_iter):
            x0, y0 = self.get_random_sample()
            n = n + 1
            sample = [x0, y0]
            dlist = [(node.state[0] - x0)**2 + (node.state[1] - y0)**2 for node in self.node_list]
            nearest_ind = dlist.index(min(dlist))
            nearest_node = self.node_list[nearest_ind]
            new_node = self.steer(nearest_node, sample)
            if self.check_collision(new_node, self.obstacle_list):
                states.append(new_node.state)
                controls.append(new_node.u_prev)
                if np.linalg.norm([a - b for a, b in zip(new_node.state[0:2], nearest_node.state[0:2])]) <= 0.0005:
                    nearest_node.update(new_node.state, new_node.u_prev)
                    nearest_node.stateupdate(new_node.state)
                else:
                    self.node_list.append(new_node)

            if np.linalg.norm([a - b for a, b in zip(new_node.state[0:2], self.goal)]) <= 0.1:
                terminate_flag = True
                break
        if terminate_flag == True:
            return states, controls  # cannot find path
        else:
            return None, [0,0]

    def steer(self, from_node, to_pos):
        u0 = from_node.u_prev
        state = from_node.state
        vMax = 0.5
        deltaMax = 0.36
        dt = 0.1
        w = [0.4,5,1,300,10] # left and straight
        # w = [0.4,5,1,300,10] # right

        bnds = ((-deltaMax, deltaMax), (0.5, vMax))

        fun = lambda u : pRRT.objective(state,u,u0,to_pos,w,vMax,deltaMax)

        u_ans = minimize(fun, u0, bounds=bnds)
        u_ans = u_ans.x
        newState = bicycle_kinematic(state,u_ans,dt)
        newNode = self.Car(newState,3,0.14,0.145,[0,0],u_ans)
        return newNode

    def objective(state,u,u_prev,sample,w,vMax,deltaMax):
        u0 = u_prev
        lf = WB/2
        lr = WB/2
        dt = 0.1
        newState = bicycle_kinematic(state,u,dt)
        refState = bicycle_kinematic(state,[0, u0[1]],dt)
        refDist = np.linalg.norm([a - b for a, b in zip(refState[0:2], sample)])

        if u[0] == 0:
            radius2 = np.inf
        else:
            radius2 = lr**2 + ((lf+lr)*(1/math.tan(u[0])))**2

        obj_fun = w[0]*( ((u[0]-u0[0])/deltaMax*4)**2 + ((u[1]-u0[1])/vMax*2)**2 ) + w[1]*( state[3]**4/radius2 ) + w[2]*( (newState[3]-vMax)/vMax )**2 + w[3]*( np.linalg.norm([a - b for a, b in zip(newState[0:2], sample)]) - refDist )
        return obj_fun

    def get_random_sample(self):
        flat_pdf = self.pdfMap.flatten()
        ind_list = np.arange(0,len(flat_pdf),1)
        sample_ind = np.random.choice(ind_list,1,replace=True,p=flat_pdf)
        return_x = self.x_val[int(sample_ind%len(self.x_val))]
        return_y = self.y_val[int(sample_ind/len(self.y_val))]
        return return_x, return_y

    def plot_pdfMap(xx,yy,pdfMap, ax):
        ax.subplots(subplot_kw={"projection": "3d"})
        surf = ax.plot_surface(xx, yy, pdfMap, cmap=cm.coolwarm, linewidth=0, antialiased=False)
        # fig.colorbar(surf, shrink=0.5, aspect=5)
        plt.show()

    def check_collision(self, node, obstacleList):

        if node is None:
            return False

        for (ox, oy, size) in obstacleList:
            dx_list = [ox - node.state[0]]
            dy_list = [oy - node.state[1]]
            d_list = [dx * dx + dy * dy for (dx, dy) in zip(dx_list, dy_list)]

            if min(d_list) <= (size+node.radius)**2:
                return False  # collision

        return True  # safe

def generatePdfMap(mu, sigma, lamb, xx, yy, dim):
    updf = 1/(dim*dim)
    if mu.shape[0] == 1:
        [pdf,maxValue] = gauss2d(mu,sigma,xx,yy)
        pdfMap = (lamb*pdf*updf)/(maxValue) + updf
    else:
        maxs = []
        pdfvals = []
        for i in range(0,mu.shape[0]):
            [pdf,maxValue] = gauss2d(mu,sigma,xx,yy)
            pdfvals.append(pdf)
            maxs.append(maxValue)

        minValue = min(maxs[1:mu.shape[0]])
        scale = maxs/minValue
        pdfMap = (lamb*pdfvals[0])/scale[0]
        for i in range(1,len(pdfvals)):
            pdfMap = pdfMap - pdfvals[i]/scale[i]
        
        pdfMap = np.array(pdfMap)
        pdfMap = pdfMap + np.abs(np.min(pdfMap))

    return pdfMap

def gauss2d(mu, sigma, xx, yy):
    num = -((np.square(xx-mu[0][0]))/(2*(sigma[0][0]**2)) + (np.square(yy-mu[0][1]))/(2*(sigma[0][1]**2)))
    den = 2*np.pi*sigma[0][0]*sigma[0][1]
    pdfMap = (np.exp(num))/(den)
    maxVal = 1/den
    return pdfMap, maxVal

def getOVPos(laneStart, laneWidth):
    OV = pRRT.Car([1.0668+0.5588/2,0.9144+1.1176+0.5602/2,-np.pi/2,0],3,0.14,0.145,[0,0],[0,0])
    # OV = RRT.Car([laneStart+laneWidth/2,laneStart+laneWidth,-np.pi/2,0],3,0.14,0.145,[0,0],[0,0])
    OVBound = OV.calBoundaries()
    return OVBound

@staticmethod
def bicycle_kinematic(state,u,dt):
    lf = WB/2
    lr = WB/2

    newState = [0, 0, 0, 0]
    beta = math.atan(lr*math.tan(u[0])/(lf+lr))
    xdot = state[3]*math.cos(state[2]+beta)
    ydot = state[3]*math.sin(state[2]+beta)
    psidot = state[3]*math.sin(beta)/lr

    newState[0] = state[0] + xdot*dt
    newState[1] = state[1] + ydot*dt
    newState[2] = state[2] + psidot*dt
    if newState[2] > 2*np.pi:
        newState[2] = newState[2] - 2*np.pi
    elif newState[2] < 0:
        newState[2] = newState[2] + 2*np.pi
    newState[3] = u[1]

    return newState

def plot_pdfMap(xx,yy,pdfMap,ax):
    ax.subplots(subplot_kw={"projection": "3d"})
    surf = ax.plot_surface(xx, yy, pdfMap, cmap=cm.coolwarm, linewidth=0, antialiased=False)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.show()

def main(sx,sy,gx,gy,obstacle_list,robot_radius,i):
    i = i

    traj_df = pd.DataFrame({'X':[], 'Y':[]})

    px = 1/plt.rcParams['figure.dpi']  # pixel in inches
    fig = plt.figure(figsize=(1080*px, 1150*px))
    ax = fig.add_subplot(1, 1, 1)
    # ax.set_position([0, 0, 1080*px, 1080*px])
    im = plt.imread('C:/D/ABHYAAS/Coding/imagesx10/env1.png')
    implot = plt.imshow(im, extent=(-11, 11, -11, 11))
    ax.set_xlim([-11, 11])
    ax.set_ylim([-11, 11])

    # Environment Dimensions
    laneWidth = 0.5588 # Width of a single lane
    laneStart = 1.0668 # X position of the Left shoulder of the bottom part of intersection

    # Creating the world object
    # world = labWorld.LabWorld(laneWidth,laneStart)
    dd = 0.05
    dim = 2*(3)
    x = np.arange(0,dim+dd,dd)
    y = np.arange(0,dim+dd,dd)
    xx, yy = np.meshgrid(x,y)

    # pRRT Lambda parameter
    lamb = math.exp(20) # math.exp(20)

    start = [sx, sy] #laneStart+laneWidth+laneWidth/2,laneStart-LEN/2
    goalPos = [gx, gy]

    # Generating a pdfMap for pRRT
    mu = [goalPos]
    sigma = [[0.05,0.05]]
    mu = np.asarray(mu)
    sigma = np.asarray(sigma)
    pdfMap = generatePdfMap(mu, sigma, lamb, xx, yy, dim)
    pdfMap = pdfMap/np.sum(pdfMap)

    # Plotting the pdfMap
    # plot_pdfMap(xx,yy,pdfMap)
    if gx!=sx:
        theta_state = math.atan2((gy-sy),(gx-sx))
    elif gy<sy:
        theta_state = -1*np.pi/2
    else:
        theta_state = np.pi/2

    # Initializing ego vehicle object
    ego = pRRT.Car([sx,sy,theta_state,0],3,robot_radius,0.145,[0,0],[0,0])
    # ego = RRT.Car([laneStart+laneWidth,laneStart+laneWidth,(np.pi/2)+(np.pi/3),0.5],3,0.14,0.145,[0,0],[0,0])

    
    max_iter = 100 #100000
    # OVPoints = getOVPos(laneStart, laneWidth)
    # worldEdges = world.edges(1)
    # obstacle_list = [(4, 6, 1.2), (-7.5, 7.5, 1), (8.35, -8.25, 1.2), (-4, -5, 1.2), (3, -5, 1.2),
    #                 (9, 7, 1), (-7, -8, 1), (0, -8, 0.245), (-6, 0, 0.245), (7, -2, 0.245), (-1, 5, 0.245)]  # [x, y, radius]

    # Set Initial parameters
    rrt = pRRT(ego,goalPos,obstacle_list,x,y,pdfMap,max_iter)
    start_time = time.localtime(time.time())
    path, controls = rrt.planning()
    end_time = time.localtime(time.time())
    # print(controls)
    u0 = controls[0]
    # world.draw()

    ox, oy = [], []
    for j in range(0,len(obstacle_list)):
        x = obstacle_list[j][0]
        y = obstacle_list[j][1]
        R = obstacle_list[j][2]
        for theta in range(0,361):
            ox.append(x + R*np.cos(theta*np.pi/180))
            oy.append(y + R*np.sin(theta*np.pi/180))

    traj_len = 0

    ego.draw(ax)
    if path == None:
        print("No plan found")
        ax.plot(ox, oy, ".k", markersize=0.5, alpha=0.75)
        ax.plot(goalPos[0], goalPos[1], 'gold', marker='^', markersize=10, label='Goal Position')
        ax.grid(color = 'black', linestyle = ':', which='both')
        # # Set the font size for x tick labels
        plt.rc('xtick', labelsize=25)
        # # Set the font size for y tick labels
        plt.rc('ytick', labelsize=25)
        plt.legend(loc='lower center', bbox_to_anchor=(0.5,-0.15), fontsize=15, ncol=2)
        # plt.pause(0.01)  # Need for Mac
        plt.savefig(f'prrt_g{i+1}.png')
        plt.show()
        return [0,0], start_time, end_time, traj_len
    else:
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
        traj_df.to_csv(f'prrt_trajectory{i+1}.csv', index=False)
        print('pRRT csv saved')

        points_x = []
        points_y = []
        for i in range(0,len(path)):
            points_x.append(path[i][0])
            points_y.append(path[i][1])
        ax.plot(ox, oy, ".k", markersize=0.5, alpha=0.75)
        ax.plot(points_x, points_y, 'yellow', linestyle = '-')
        ax.plot(goalPos[0], goalPos[1], 'gold', marker='^', markersize=10, label='Goal Position')
        ax.grid(color = 'black', linestyle = ':', which='both')
        # # Set the font size for x tick labels
        plt.rc('xtick', labelsize=25)
        # # Set the font size for y tick labels
        plt.rc('ytick', labelsize=25)
        plt.legend(loc='lower center', bbox_to_anchor=(0.5,-0.15), fontsize=15, ncol=2)
        plt.savefig(f'prrt_g{i+1}.png')
        plt.show()
        
        return path[0], start_time, end_time, traj_len

if __name__ == "__main__":
    main()