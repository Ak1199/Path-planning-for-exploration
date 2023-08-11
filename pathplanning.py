import astar
import rrt
import rrtstar
import pRRTNew
import apf

import pandas as pd
import time


### Outdoor 0

start_pos = (0,0) #(-8, -6.5) #(1, -5) #(8.5, -2.5) #(4, 8.25) #(0, 0), #400 32
#             #  (5.0, -7.5), #460 32
#             #  (2.5, 0), #560 2
#             #  (-7.5, 5.0), #560 32
#             #  (-3.25, -8.5)] #392 32

goal_pos = [(0, 2.5), (-2.5, 5), (-1, 7.5),
            (2.5, 5), (4, 8.25), (6, 6),
            (9.5, 8.5), (7.5, 6),
            (6, 0), (8.5, -2.5),(6, -3),
            (9.5, -6.5),(6.5, -9), 
            (4, -6.5), (1, -5),(0.25, -9), (-2, -7.5),
            (-4.5, -7), (-8.5, -9.5), (-8, -6.5),(-6, -5),
            (-4.5, -1), (-7.5, 2),
            (-9.25, 6.5), (-8, 9.25)]

obstacle_list = [(4, 6, 1.2), (-7.5, 7.5, 1), (8.35, -8.25, 1.2), (-4, -5, 1.2), (3, -5, 1.2),
                    (9, 7, 1), (-7, -8, 1), (0, -8, 0.245), (-6, 0, 0.245), (7, -2, 0.245), (-1, 5, 0.245)]  # [x, y, radius]

### Outdoor 1

# start_pos = (-2.5, -4) #(6.5, -9) #(7.5, -1.75) #(3.25, 6) #(0, 0)

# # goal_pos = [(-3.5, 4), (-2, 6.75), (1.25, 7.5),
# #             (4, 8.5), (3.25, 6)]
# # goal_pos = [(5.35, 4.5), 
# #             (7.5, 6.35), (9, 1.5), (6, 0), 
# #             (7.5, -1.75)]
# # goal_pos = [(2.5, -1.65), (2, -3.75),
# #             (5.75, -5.75), (7.5, -6.5), (6.5, -9)]
# # goal_pos = [(-0.25, -9), (-4.15, -7.5), (-8.5, -7.75),
# #             (-6, -5), (-2.5, -4)]
# goal_pos = [(-0.5, -2.5), 
#             (-3, -1), (-8.75, -0.5), (-7.5, 2),
#             (-7.5, 5), (-6.5, 9)]

# obstacle_list = [(-1.75, 4.85, 1.2), (7, 4.5, 1), (8, 0, 1.2), (-2, -7.75, 1.2), (-7.75, -6, 1.2),
#                     (3.5, -3.25, 1), (-7.25, 0, 1), (-7, 6.25, 0.245), (3.5, 7.5, 0.245), (-2, -3, 0.245), (6.85, -7.25, 0.245)]  # [x, y, radius]


robot_radius = 0.5
show_animation = True

astar_df = pd.DataFrame({'Star Position X':[], 'Start Position Y':[], 
                         'Goal Position X':[], 'Goal Position Y':[],
                         'Actual Final Position X':[], 'Actual Final Position Y':[],
                         'Start Time':[], 'End Time':[], 'Computation Time (s)':[], 'Trajectory Length':[], 
                         'Start FLOPS':[], 'End FLOPS':[], 'Number of FLOPS':[]})

rrt_df = pd.DataFrame({'Star Position X': [], 'Start Position Y':[], 
                       'Goal Position X':[], 'Goal Position Y':[],
                       'Actual Final Position X':[], 'Actual Final Position Y':[],
                       'Start Time':[], 'End Time':[], 'Computation Time (s)':[], 'Trajectory Length':[], 
                       'Start FLOPS':[], 'End FLOPS':[], 'Number of FLOPS':[]})

rrtstar_df = pd.DataFrame({'Star Position X': [], 'Start Position Y':[], 
                           'Goal Position X':[], 'Goal Position Y':[],
                           'Actual Final Position X':[], 'Actual Final Position Y':[],
                           'Start Time':[], 'End Time':[], 'Computation Time (s)':[], 'Trajectory Length':[], 
                           'Start FLOPS':[], 'End FLOPS':[], 'Number of FLOPS':[]})

prrt_df = pd.DataFrame({'Star Position X': [], 'Start Position Y':[], 
                       'Goal Position X':[], 'Goal Position Y':[],
                       'Actual Final Position X':[], 'Actual Final Position Y':[],
                       'Start Time':[], 'End Time':[], 'Computation Time (s)':[], 'Trajectory Length':[], 
                       'Start FLOPS':[], 'End FLOPS':[], 'Number of FLOPS':[]})

apf_df = pd.DataFrame({'Star Position X': [], 'Start Position Y':[], 
                       'Goal Position X':[], 'Goal Position Y':[],
                       'Actual Final Position X':[], 'Actual Final Position Y':[],
                       'Start Time':[], 'End Time':[], 'Computation Time (s)':[], 'Trajectory Length':[], 
                       'Start FLOPS':[], 'End FLOPS':[], 'Number of FLOPS':[]})

# astar_df = pd.DataFrame(pd.read_csv('astar_0_1.csv'))
# rrt_df = pd.DataFrame(pd.read_csv('rrt_0_1.csv'))
# rrtstar_df = pd.DataFrame(pd.read_csv('rrtstar_0_1.csv'))
# prrt_df = pd.DataFrame(pd.read_csv('prrt_0_1.csv'))
# apf_df = pd.DataFrame(pd.read_csv('apf_0_1.csv'))

sx = start_pos[0]
sy = start_pos[1]

i = 0
for i in range(0,len(goal_pos)):
    gx = goal_pos[i][0]
    gy = goal_pos[i][1]

    
    print('start A*')
    rx, ry, start_time_astar, end_time_astar, traj_len_astar = astar.main(sx,sy,gx,gy,obstacle_list,robot_radius,i)

    comp_time_astar = (end_time_astar[3]*3600+end_time_astar[4]*60+end_time_astar[5]) - (start_time_astar[3]*3600+start_time_astar[4]*60+start_time_astar[5])
    
    astar_temp = pd.DataFrame({'Star Position X': [sx], 'Start Position Y':[sy], 
                               'Goal Position X':[gx], 'Goal Position Y':[gy],
                               'Actual Final Position X':[rx], 'Actual Final Position Y':[ry],
                               'Start Time':[str(start_time_astar[3])+'h'+str(start_time_astar[4])+'m'+str(start_time_astar[5])+'s'], 
                               'End Time':[str(end_time_astar[3])+'h'+str(end_time_astar[4])+'m'+str(end_time_astar[5])+'s'], 
                               'Computation Time (s)':[comp_time_astar],
                               'Trajectory Length':[traj_len_astar]})
    astar_df = pd.concat([astar_df, astar_temp])
    astar_df.to_csv('astar_0_05.csv', index=False)
    print('end A*')

    print('start RRT')
    path_rrt, start_time_rrt, end_time_rrt, traj_len_rrt = rrt.main(sx,sy,gx,gy,obstacle_list,robot_radius,i)

    comp_time_rrt = (end_time_rrt[3]*3600+end_time_rrt[4]*60+end_time_rrt[5]) - (start_time_rrt[3]*3600+start_time_rrt[4]*60+start_time_rrt[5])

    rrt_temp = pd.DataFrame({'Star Position X': [sx], 'Start Position Y':[sy], 
                             'Goal Position X':[gx], 'Goal Position Y':[gy],
                             'Actual Final Position X':[path_rrt[0]], 'Actual Final Position Y':[path_rrt[1]],
                             'Start Time':[str(start_time_rrt[3])+'h'+str(start_time_rrt[4])+'m'+str(start_time_rrt[5])+'s'], 
                             'End Time':[str(end_time_rrt[3])+'h'+str(end_time_rrt[4])+'m'+str(end_time_rrt[5])+'s'], 
                             'Computation Time (s)':[comp_time_rrt],
                             'Trajectory Length':[traj_len_rrt]})
    rrt_df = pd.concat([rrt_df, rrt_temp])
    rrt_df.to_csv('rrt_0_05.csv', index=False)
    print('end RRT')

    print('start RRT*')
    path_rrtstar, start_time_rrtstar, end_time_rrtstar, traj_len_rrtstar = rrtstar.main(sx,sy,gx,gy,obstacle_list,robot_radius,i)

    comp_time_rrtstar = (end_time_rrtstar[3]*3600+end_time_rrtstar[4]*60+end_time_rrtstar[5]) - (start_time_rrtstar[3]*3600+start_time_rrtstar[4]*60+start_time_rrtstar[5])

    rrtstar_temp = pd.DataFrame({'Star Position X': [sx], 'Start Position Y':[sy], 
                                'Goal Position X':[gx], 'Goal Position Y':[gy],
                                'Actual Final Position X':[path_rrtstar[0]], 'Actual Final Position Y':[path_rrtstar[1]],
                                'Start Time':[str(start_time_rrtstar[3])+'h'+str(start_time_rrtstar[4])+'m'+str(start_time_rrtstar[5])+'s'], 
                                'End Time':[str(end_time_rrtstar[3])+'h'+str(end_time_rrtstar[4])+'m'+str(end_time_rrtstar[5])+'s'], 
                                'Computation Time (s)':[comp_time_rrtstar],
                                'Trajectory Length':[traj_len_rrtstar]})
    rrtstar_df = pd.concat([rrtstar_df, rrtstar_temp])
    rrtstar_df.to_csv('rrtstar_0_05.csv', index=False)
    # print(f'Start: {sx,sy}, End: {gx, gy}, Actual end: {path_rrtstar[0], path_rrtstar[1]} \n Start Time: {str(start_time_rrtstar[3])}h{str(start_time_rrtstar[4])}m{str(start_time_rrtstar[5])}s, End Time:{str(end_time_rrtstar[3])}h{str(end_time_rrtstar[4])}m{str(end_time_rrtstar[5])}s')
    print('end RRT*')

    print('start pRRT')
    path_prrt, start_time_prrt, end_time_prrt, traj_len_prrt = pRRTNew.main(sx,sy,gx,gy,obstacle_list,robot_radius,i)

    comp_time_prrt = (end_time_prrt[3]*3600+end_time_prrt[4]*60+end_time_prrt[5]) - (start_time_prrt[3]*3600+start_time_prrt[4]*60+start_time_prrt[5])

    prrt_temp = pd.DataFrame({'Star Position X': [sx], 'Start Position Y':[sy], 
                             'Goal Position X':[gx], 'Goal Position Y':[gy],
                             'Actual Final Position X':[path_prrt[0]], 'Actual Final Position Y':[path_prrt[1]],
                             'Start Time':[str(start_time_prrt[3])+'h'+str(start_time_prrt[4])+'m'+str(start_time_prrt[5])+'s'], 
                             'End Time':[str(end_time_prrt[3])+'h'+str(end_time_prrt[4])+'m'+str(end_time_prrt[5])+'s'], 
                             'Computation Time (s)':[comp_time_prrt],
                             'Trajectory Length':[traj_len_prrt]})
    prrt_df = pd.concat([prrt_df, prrt_temp])
    prrt_df.to_csv('prrt_0_05.csv', index=False)
    print('end pRRT')

    print('start APF')
    rx_apf, ry_apf, start_time_apf, end_time_apf, traj_len_apf = apf.main(sx,sy,gx,gy,obstacle_list,robot_radius,i)

    comp_time_apf = (end_time_apf[3]*3600+end_time_apf[4]*60+end_time_apf[5]) - (start_time_apf[3]*3600+start_time_apf[4]*60+start_time_apf[5])

    apf_temp = pd.DataFrame({'Star Position X': [sx], 'Start Position Y':[sy], 
                             'Goal Position X':[gx], 'Goal Position Y':[gy],
                             'Actual Final Position X':[rx_apf], 'Actual Final Position Y':[ry_apf],
                             'Start Time':[str(start_time_apf[3])+'h'+str(start_time_apf[4])+'m'+str(start_time_apf[5])+'s'], 
                             'End Time':[str(end_time_apf[3])+'h'+str(end_time_apf[4])+'m'+str(end_time_apf[5])+'s'], 
                             'Computation Time (s)':[comp_time_apf],
                             'Trajectory Length':[traj_len_apf]})
    apf_df = pd.concat([apf_df, apf_temp])
    apf_df.to_csv('apf_0_05.csv', index=False)
    print('end APF')

    sx = gx
    sy = gy
    

    
    

    

