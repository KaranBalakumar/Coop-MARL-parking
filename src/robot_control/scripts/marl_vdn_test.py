#!/usr/bin/python3

import rclpy
from rclpy.node import Node
import numpy as np
import random
import cv2
import time
from gazebo_msgs.msg import ModelStates, LinkStates, EntityState
from std_msgs.msg import Float64MultiArray
from sensor_msgs.msg import Image
from gazebo_msgs.srv import SetEntityState
from std_srvs.srv import Empty
import threading
import math
from math import cos, sin, atan, atan2, pi
from cv_bridge import CvBridge
import argparse
import os
import sys
from copy import deepcopy
import torch
from torch.utils.tensorboard import SummaryWriter

sys.path.append('../')
from configs.arguments import get_common_args
from configs.vdn_config import VDNConfig
from marltoolkit.agents.vdn_agent import VDNAgent
from marltoolkit.data.ma_replaybuffer import ReplayBuffer
from marltoolkit.modules.actors import RNNActor
from marltoolkit.modules.mixers import VDNMixer
from marltoolkit.utils.transforms import OneHotTransform
from marltoolkit.runners.episode_runner import (run_evaluate_episode, run_train_episode)
from marltoolkit.utils import (ProgressBar, TensorboardLogger,
                               get_outdir, get_root_logger)

bridge = CvBridge()

observation = np.array([
                        0.0, 0.0, 0.0, #robot1 x, y, yaw angle
                        0.0, 0.0, 0.0, #robot2 
                        0.0, 0.0, 0.0  #robot3
                        ], float)

arrow_positions = np.array([[0.0, -3.0, 0.0], # blue
                            [0.0, 0.0, 0.0],  # red
                            [0.0, 3.0, 0.0]   # green
                            ], float)

total_observation = np.array([[0.0, 0.0, 0.0, # robot1
                               0.0, 0.0, 0.0, # 2
                               0.0, 0.0, 0.0, # 3
                               0.0, 0.0, 0.0],# target diff1
                              [0.0, 0.0, 0.0, # robot2
                               0.0, 0.0, 0.0, # 1
                               0.0, 0.0, 0.0, # 3
                               0.0, 0.0, 0.0],# target diff2
                              [0.0, 0.0, 0.0, # robot3
                               0.0, 0.0, 0.0, # 1
                               0.0, 0.0, 0.0, # 2
                               0.0, 0.0, 0.0] # target diff3
                               ], float)

clash_sum = 0
LIMIT = 300
TIME_DELTA = 0.1

class GazeboEnv(Node):

    def __init__(self):
        super().__init__('env')

        self.set_state = self.create_client(SetEntityState, "/gazebo/set_entity_state")
        self.unpause = self.create_client(Empty, "/unpause_physics")
        self.pause = self.create_client(Empty, "/pause_physics")
        self.reset_world = self.create_client(Empty, "/reset_world")

        self.knuckle_pos1 = np.array([0,0], float) #left right
        self.wheel_vel1 = np.array([0,0], float) #left right
        self.publisher_pos1 = self.create_publisher(Float64MultiArray, '/robot_1/forward_position_controller/commands', 10)
        self.publisher_vel1 = self.create_publisher(Float64MultiArray, '/robot_1/forward_velocity_controller/commands', 10)

        self.knuckle_pos2 = np.array([0,0], float) #left right
        self.wheel_vel2 = np.array([0,0], float) #left right
        self.publisher_pos2 = self.create_publisher(Float64MultiArray, '/robot_2/forward_position_controller/commands', 10)
        self.publisher_vel2 = self.create_publisher(Float64MultiArray, '/robot_2/forward_velocity_controller/commands', 10)

        self.knuckle_pos3 = np.array([0,0], float) #left right
        self.wheel_vel3 = np.array([0,0], float) #left right
        self.publisher_pos3 = self.create_publisher(Float64MultiArray, '/robot_3/forward_position_controller/commands', 10)
        self.publisher_vel3 = self.create_publisher(Float64MultiArray, '/robot_3/forward_velocity_controller/commands', 10)

        #robot1 state
        self.set_robot_1_state = EntityState()
        self.set_robot_1_state.name = "robot_1"
        self.set_robot_1_state.pose.position.x = 0.0
        self.set_robot_1_state.pose.position.y = -3.0
        self.set_robot_1_state.pose.position.z = 0.7
        self.set_robot_1_state.pose.orientation.x = 0.0
        self.set_robot_1_state.pose.orientation.y = 0.0
        self.set_robot_1_state.pose.orientation.z = 0.0
        self.set_robot_1_state.pose.orientation.w = 1.0
        self.robot_1_state = SetEntityState.Request()

        #robot2 state
        self.set_robot_2_state = EntityState()
        self.set_robot_2_state.name = "robot_2"
        self.set_robot_2_state.pose.position.x = 0.0
        self.set_robot_2_state.pose.position.y = 0.0
        self.set_robot_2_state.pose.position.z = 0.7
        self.set_robot_2_state.pose.orientation.x = 0.0
        self.set_robot_2_state.pose.orientation.y = 0.0
        self.set_robot_2_state.pose.orientation.z = 0.0
        self.set_robot_2_state.pose.orientation.w = 1.0
        self.robot_2_state = SetEntityState.Request()

        #robot3 state
        self.set_robot_3_state = EntityState()
        self.set_robot_3_state.name = "robot_3"
        self.set_robot_3_state.pose.position.x = 0.0
        self.set_robot_3_state.pose.position.y = 3.0
        self.set_robot_3_state.pose.position.z = 0.7
        self.set_robot_3_state.pose.orientation.x = 0.0
        self.set_robot_3_state.pose.orientation.y = 0.0
        self.set_robot_3_state.pose.orientation.z = 0.0
        self.set_robot_3_state.pose.orientation.w = 1.0
        self.robot_3_state = SetEntityState.Request()
        
        self.robot1_done = False
        self.robot2_done = False
        self.robot3_done = False
        self.done = False

        self.res_observation = np.array([
                                         0.0, 0.0, 0.0, #robot1 x, y, theta
                                         0.0, 0.0, 0.0, #robot2 
                                         0.0, 0.0, 0.0  #robot3
                                        ], float)
        
        self.state = np.array([
                            0.0, 0.0, 0.0, # robot1 x,y,theta
                            0.0, 0.0, 0.0,  
                            0.0, 0.0, 0.0,
                            0.0, 0.0, 0.0, # robot1 difference with goal x, y, theta
                            0.0, 0.0, 0.0,  
                            0.0, 0.0, 0.0,
                            ], float)
        
        self.boundary_x = 7.0
        self.boundary_y = 7.0

        self.image_L = 700 # 900 - 2*100 
        self.H = self.image_L
        self.W = self.image_L
        self.C_H = int(self.H/2) # center of the image
        self.C_W = int(self.W/2)
        self.L_m = 1.8
        self.T_m = 1.1
        self.Rw = 0.3
        self.pix2m = 0.02 #1 pixel = 0.02m
        self.robot_L = int(2.8/self.pix2m) 
        self.robot_W = int(1.3/self.pix2m) 
        margin = int(1.0/self.pix2m) 
        self.OBL = self.robot_L + 2*margin # outer_boundary_L
        self.OBW = self.robot_W + 2*margin # outer_boundary_W

        self.n_agents = 3
        self.n_actions = 11
        self.actions_one_hot_transform = OneHotTransform(self.n_actions)

        self.action1 = [0,0]
        self.action2 = [0,0]
        self.action3 = [0,0]
        self.omega1 = 0.6
        self.omega2 = 0.3

        self.time_step = 0

    def step(self, action):
        global observation, clash_sum, arrow_positions

        reward = 0

        ## robot1
        if  (action[0] == 0):self.action1 = [1, self.omega1]
        elif(action[0] == 1):self.action1 = [1, self.omega2]
        elif(action[0] == 2):self.action1 = [1, 0.0]    
        elif(action[0] == 3):self.action1 = [1, -self.omega2]  
        elif(action[0] == 4):self.action1 = [1, -self.omega1]  
        elif(action[0] == 5):self.action1 = [-1, self.omega1] 
        elif(action[0] == 6):self.action1 = [-1, self.omega2]     
        elif(action[0] == 7):self.action1 = [-1, 0.0]   
        elif(action[0] == 8):self.action1 = [-1, -self.omega2]  
        elif(action[0] == 9):self.action1 = [-1, -self.omega1]   
        elif(action[0] == 10):self.action1 = [0.0, 0.0]  

        try:
            self.knuckle_pos1[0] = atan(self.action1[1]*self.L_m/(2*self.action1[0] - self.action1[1]*self.T_m))
        except:
            self.knuckle_pos1[0] = 0
        
        try:
            self.knuckle_pos1[1] = atan(self.action1[1]*self.L_m/(2*self.action1[0] + self.action1[1]*self.T_m))
        except:
            self.knuckle_pos1[1] = 0

        self.wheel_vel1[0] = (self.action1[0] - self.action1[1]*self.T_m/2)/self.Rw
        self.wheel_vel1[1] = (self.action1[0] + self.action1[1]*self.T_m/2)/self.Rw    

        ## robot2
        if  (action[1] == 0):self.action2 = [1, self.omega1]
        elif(action[1] == 1):self.action2 = [1, self.omega2]
        elif(action[1] == 2):self.action2 = [1, 0.0]    
        elif(action[1] == 3):self.action2 = [1, -self.omega2]  
        elif(action[1] == 4):self.action2 = [1, -self.omega1]  
        elif(action[1] == 5):self.action2 = [-1, self.omega1] 
        elif(action[1] == 6):self.action2 = [-1, self.omega2]     
        elif(action[1] == 7):self.action2 = [-1, 0.0]   
        elif(action[1] == 8):self.action2 = [-1, -self.omega2]  
        elif(action[1] == 9):self.action2 = [-1, -self.omega1]   
        elif(action[1] == 10):self.action2 = [0.0, 0.0]  

        try:
            self.knuckle_pos2[0] = atan(self.action2[1]*self.L_m/(2*self.action2[0] - self.action2[1]*self.T_m))
        except:
            self.knuckle_pos2[0] = 0

        try:
            self.knuckle_pos2[1] = atan(self.action2[1]*self.L_m/(2*self.action2[0] + self.action2[1]*self.T_m))
        except:
            self.knuckle_pos2[1] = 0

        self.wheel_vel2[0] = (self.action2[0] - self.action2[1]*self.T_m/2)/self.Rw
        self.wheel_vel2[1] = (self.action2[0] + self.action2[1]*self.T_m/2)/self.Rw    

        ## robot3
        if  (action[2] == 0):self.action3 = [1, self.omega1]
        elif(action[2] == 1):self.action3 = [1, self.omega2]
        elif(action[2] == 2):self.action3 = [1, 0.0]    
        elif(action[2] == 3):self.action3 = [1, -self.omega2]  
        elif(action[2] == 4):self.action3 = [1, -self.omega1]  
        elif(action[2] == 5):self.action3 = [-1, self.omega1] 
        elif(action[2] == 6):self.action3 = [-1, self.omega2]     
        elif(action[2] == 7):self.action3 = [-1, 0.0]   
        elif(action[2] == 8):self.action3 = [-1, -self.omega2]  
        elif(action[2] == 9):self.action3 = [-1, -self.omega1]   
        elif(action[2] == 10):self.action3 = [0.0, 0.0]  

        try:
            self.knuckle_pos3[0] = atan(self.action3[1]*self.L_m/(2*self.action3[0] - self.action3[1]*self.T_m))
        except:
            self.knuckle_pos3[0] = 0

        try:
            self.knuckle_pos3[1] = atan(self.action3[1]*self.L_m/(2*self.action3[0] + self.action3[1]*self.T_m))
        except:
            self.knuckle_pos3[1] = 0

        self.wheel_vel3[0] = (self.action3[0] - self.action3[1]*self.T_m/2)/self.Rw
        self.wheel_vel3[1] = (self.action3[0] + self.action3[1]*self.T_m/2)/self.Rw    

        knuckle_pos_array1 = Float64MultiArray(data=self.knuckle_pos1)    
        wheel_vel_array1 = Float64MultiArray(data=self.wheel_vel1) 
        self.publisher_pos1.publish(knuckle_pos_array1)     
        self.publisher_vel1.publish(wheel_vel_array1)  

        knuckle_pos_array2 = Float64MultiArray(data=self.knuckle_pos2)    
        wheel_vel_array2 = Float64MultiArray(data=self.wheel_vel2) 
        self.publisher_pos2.publish(knuckle_pos_array2)     
        self.publisher_vel2.publish(wheel_vel_array2)  

        knuckle_pos_array3 = Float64MultiArray(data=self.knuckle_pos3)    
        wheel_vel_array3 = Float64MultiArray(data=self.wheel_vel3) 
        self.publisher_pos3.publish(knuckle_pos_array3)     
        self.publisher_vel3.publish(wheel_vel_array3) 

        while not self.unpause.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('service not available, waiting again...')
        try:
            self.unpause.call_async(Empty.Request())
        except:
            self.get_logger().info("/unpause_physics service call failed")

        time.sleep(TIME_DELTA)

        while not self.pause.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('service not available, waiting again...')
        try:
            self.pause.call_async(Empty.Request())
        except (rclpy.ServiceException) as e:
            self.get_logger().info("/gazebo/pause_physics service call failed")

        robot1_dif_x = observation[0] - arrow_positions[0][0]
        robot1_dif_y = observation[1] - arrow_positions[0][1]
        robot1_dif_theta = observation[2] - arrow_positions[0][2]
        if(abs(robot1_dif_x) < 0.5 and
           abs(robot1_dif_y) < 0.5 and
           abs(robot1_dif_theta) < 0.2):
            self.robot1_done = True
            reward += 0.33333
        else:
            reward += (10 - math.sqrt(robot1_dif_x**2 + robot1_dif_y**2) - abs(robot1_dif_theta))/30

        robot2_dif_x = observation[3] - arrow_positions[1][0]
        robot2_dif_y = observation[4] - arrow_positions[1][1]
        robot2_dif_theta = observation[5] - arrow_positions[1][2]
        if(abs(robot2_dif_x) < 0.5 and
           abs(robot2_dif_y) < 0.5 and
           abs(robot2_dif_theta) < 0.2):
            self.robot2_done = True
            reward += 0.3333
        else:
            reward += (10 - math.sqrt(robot2_dif_x**2 + robot2_dif_y**2) - abs(robot2_dif_theta))/30

        robot3_dif_x = observation[6] - arrow_positions[2][0]
        robot3_dif_y = observation[7] - arrow_positions[2][1]
        robot3_dif_theta = observation[8] - arrow_positions[2][2]
        if(abs(robot3_dif_x) < 0.5 and
           abs(robot3_dif_y) < 0.5 and
           abs(robot3_dif_theta) < 0.2):
            self.robot3_done = True
            reward += 0.3333
        else:
            reward += (10 - math.sqrt(robot3_dif_x**2 + robot3_dif_y**2) - abs(robot3_dif_theta))/30


        # clash penalty
        reward -= clash_sum/20000

        if(self.robot1_done == True and self.robot2_done == True and self.robot3_done == True):
            self.done = True
            reward = 1.0

        # if out of the boundary
        # robot1
        if(observation[0] > self.boundary_x or observation[0] < -self.boundary_x or
           observation[1] > self.boundary_y or observation[1] < -self.boundary_y):
            self.done = True
            reward = -1.0
        # robot2
        elif(observation[3] > self.boundary_x or observation[3] < -self.boundary_x or
           observation[4] > self.boundary_y or observation[4] < -self.boundary_y):
            self.done = True
            reward = -1.0
        # robot3
        elif(observation[6] > self.boundary_x or observation[6] < -self.boundary_x or
           observation[7] > self.boundary_y or observation[7] < -self.boundary_y):
            self.done = True
            reward = -1.0

        if reward > 1:
            reward = 1
        elif reward < -1:
            reward = -1

        if(self.time_step == LIMIT-1):
            self.done = True

        # creating observations for newral network
        self.state[0] = observation[0]/self.boundary_x
        self.state[1] = observation[1]/self.boundary_y
        self.state[2] = observation[2]/pi
        self.state[3] = observation[3]/self.boundary_x
        self.state[4] = observation[4]/self.boundary_y
        self.state[5] = observation[5]/pi
        self.state[6] = observation[6]/self.boundary_x
        self.state[7] = observation[7]/self.boundary_y
        self.state[8] = observation[8]/pi
        self.state[9] = robot1_dif_x/self.boundary_x
        self.state[10] = robot1_dif_y/self.boundary_y
        self.state[11] = robot1_dif_theta/pi
        self.state[12] = robot2_dif_x/self.boundary_x
        self.state[13] = robot2_dif_y/self.boundary_y
        self.state[14] = robot2_dif_theta/pi
        self.state[15] = robot3_dif_x/self.boundary_x
        self.state[16] = robot3_dif_y/self.boundary_y
        self.state[17] = robot3_dif_theta/pi

        # robot1
        total_observation[0][:] = self.state[:12]
        
        # robot2
        total_observation[1][:3] = self.state[3:6]
        total_observation[1][3:6] = self.state[:3]
        total_observation[1][6:9] = self.state[6:9]
        total_observation[1][9:12] = self.state[12:15]

        # robot3
        total_observation[2][:3] = self.state[6:9]
        total_observation[2][3:9] = self.state[:6]
        total_observation[2][9:12] = self.state[15:18]  

        self.time_step += 1

        return self.state, total_observation, reward, self.done

    def reset(self):
        global arrow_positions

        self.robot1_done = False
        self.robot2_done = False
        self.robot3_done = False
        self.done = False
        self.time_step = 0

        while not self.reset_world.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('reset : service not available, waiting again...')

        try:
            self.get_logger().info('Resetting the world')
            self.reset_world.call_async(Empty.Request())
        except:
            import traceback
            traceback.print_exc()

        # set random positions for arraw markers
        # there are 6 possible combinations for marker placement
        placement = random.randint(0, 5)
        if(placement==0):
            arrow_positions[0][1] = -3.0
            arrow_positions[1][1] = 0.0
            arrow_positions[2][1] = 3.0
        elif(placement==1):
            arrow_positions[0][1] = 0.0
            arrow_positions[1][1] = -3.0
            arrow_positions[2][1] = 3.0
        elif(placement==2):
            arrow_positions[0][1] = 3.0
            arrow_positions[1][1] = 0.0
            arrow_positions[2][1] = -3.0
        elif(placement==3):
            arrow_positions[0][1] = -3.0
            arrow_positions[1][1] = 3.0
            arrow_positions[2][1] = 0.0
        elif(placement==4):
            arrow_positions[0][1] = 3.0
            arrow_positions[1][1] = -3.0
            arrow_positions[2][1] = 0.0
        elif(placement==5):
            arrow_positions[0][1] = 0.0
            arrow_positions[1][1] = 3.0
            arrow_positions[2][1] = -3.0

        # set random x position for an arraw marker
        # x coordinate is set in range of -2~2m randomly
        arrow_positions[0][0] = (random.random() - 0.5)*4
        arrow_positions[1][0] = (random.random() - 0.5)*4
        arrow_positions[2][0] = (random.random() - 0.5)*4       

        # set random (0 or 180 deg) orientation for each arraw marker
        orientation_b = random.randint(0, 1)
        if orientation_b==0:
            arrow_positions[0][2] = 0
        else:
            arrow_positions[0][2] = pi

        orientation_r = random.randint(0, 1)
        if orientation_r==0:
            arrow_positions[1][2] = 0
        else:
            arrow_positions[1][2] = pi

        orientation_g = random.randint(0, 1)
        if orientation_g==0:
            arrow_positions[2][2] = 0
        else:  
            arrow_positions[2][2] = pi

        ## set random positions for a robot
        rng = np.random.default_rng()

        self.robot1_region = np.zeros((self.H, self.W), np.int8)
        self.robot2_region = np.zeros((self.H, self.W), np.int8)
        self.robot3_region = np.zeros((self.H, self.W), np.int8)
        
        margin = 80
        robot_angles_rng = rng.integers(0, 18, size = (1, 3))*20*pi/180
        robot_angles = robot_angles_rng[0]
        rand_coord1_rng = rng.integers(int(-self.image_L/2) + margin, int(self.image_L/2) - margin, size = (1, 2))
        rand_coord1 = rand_coord1_rng[0]
        self.robot1_ob = [[int(cos(robot_angles[0])*self.OBL/2 - sin(robot_angles[0])*self.OBW/2 + self.C_W + rand_coord1[0]),       int(sin(robot_angles[0])*self.OBL/2 + cos(robot_angles[0])*self.OBW/2 + self.C_H + rand_coord1[1])],
                          [int(cos(robot_angles[0])*self.OBL/2 - sin(robot_angles[0])*(-self.OBW/2) + self.C_W + rand_coord1[0]),    int(sin(robot_angles[0])*self.OBL/2 + cos(robot_angles[0])*(-self.OBW/2) + self.C_H + rand_coord1[1])],
                          [int(cos(robot_angles[0])*(-self.OBL/2) - sin(robot_angles[0])*(-self.OBW/2) + self.C_W + rand_coord1[0]), int(sin(robot_angles[0])*(-self.OBL/2) + cos(robot_angles[0])*(-self.OBW/2) + self.C_H + rand_coord1[1])],
                          [int(cos(robot_angles[0])*(-self.OBL/2) - sin(robot_angles[0])*self.OBW/2 + self.C_W + rand_coord1[0]),    int(sin(robot_angles[0])*(-self.OBL/2) + cos(robot_angles[0])*self.OBW/2 + self.C_H + rand_coord1[1])]]        
        pts1_ob = np.array(self.robot1_ob, np.int32)
        cv2.fillPoly(self.robot1_region, [pts1_ob], 255)
        self.res_observation[0] = rand_coord1[0]*self.pix2m
        self.res_observation[1] = rand_coord1[1]*self.pix2m
        self.res_observation[2] = robot_angles[0]
        self.set_robot_1_state.pose.position.x = self.res_observation[0]
        self.set_robot_1_state.pose.position.y = self.res_observation[1]
        self.set_robot_1_state.pose.orientation.z = sin(self.res_observation[2]/2)
        self.set_robot_1_state.pose.orientation.w = cos(self.res_observation[2]/2)

        # set initial position for robot2
        while True:
            self.robot2_region[:,:] = 0
            rand_coord2_rng = rng.integers(int(-self.image_L/2) + margin, int(self.image_L/2) - margin, size = (1, 2))
            rand_coord2 = rand_coord2_rng[0]
            self.robot2_ob = [[int(cos(robot_angles[1])*self.OBL/2 - sin(robot_angles[1])*self.OBW/2 + self.C_W + rand_coord2[0]),   int(sin(robot_angles[1])*self.OBL/2 + cos(robot_angles[1])*self.OBW/2 + self.C_H + rand_coord2[1])],
                        [int(cos(robot_angles[1])*self.OBL/2 - sin(robot_angles[1])*(-self.OBW/2) + self.C_W + rand_coord2[0]),    int(sin(robot_angles[1])*self.OBL/2 + cos(robot_angles[1])*(-self.OBW/2) + self.C_H + rand_coord2[1])],
                        [int(cos(robot_angles[1])*(-self.OBL/2) - sin(robot_angles[1])*(-self.OBW/2) + self.C_W + rand_coord2[0]), int(sin(robot_angles[1])*(-self.OBL/2) + cos(robot_angles[1])*(-self.OBW/2) + self.C_H + rand_coord2[1])],
                        [int(cos(robot_angles[1])*(-self.OBL/2) - sin(robot_angles[1])*self.OBW/2 + self.C_W + rand_coord2[0]),    int(sin(robot_angles[1])*(-self.OBL/2) + cos(robot_angles[1])*self.OBW/2 + self.C_H + rand_coord2[1])]]        
            pts2_ob = np.array(self.robot2_ob, np.int32)
            cv2.fillPoly(self.robot2_region, [pts2_ob], 255)
            common_part12 = cv2.bitwise_and(self.robot1_region, self.robot2_region)
            sum = cv2.countNonZero(common_part12)
            if(sum == 0):
                self.res_observation[3] = rand_coord2[0]*self.pix2m
                self.res_observation[4] = rand_coord2[1]*self.pix2m
                self.res_observation[5] = robot_angles[1]
                self.set_robot_2_state.pose.position.x = self.res_observation[3]
                self.set_robot_2_state.pose.position.y = self.res_observation[4]
                self.set_robot_2_state.pose.orientation.z = sin(self.res_observation[5]/2)
                self.set_robot_2_state.pose.orientation.w = cos(self.res_observation[5]/2)
                break

        # set initial position for robot3
        while True:
            self.robot3_region[:,:] = 0
            rand_coord3_rng = rng.integers(int(-self.image_L/2) + margin, int(self.image_L/2) - margin, size = (1, 2))
            rand_coord3 = rand_coord3_rng[0]
            self.robot3_ob = [[int(cos(robot_angles[2])*self.OBL/2 - sin(robot_angles[2])*self.OBW/2 + self.C_W + rand_coord3[0]),       int(sin(robot_angles[2])*self.OBL/2 + cos(robot_angles[2])*self.OBW/2 + self.C_H + rand_coord3[1])],
                            [int(cos(robot_angles[2])*self.OBL/2 - sin(robot_angles[2])*(-self.OBW/2) + self.C_W + rand_coord3[0]),    int(sin(robot_angles[2])*self.OBL/2 + cos(robot_angles[2])*(-self.OBW/2) + self.C_H + rand_coord3[1])],
                            [int(cos(robot_angles[2])*(-self.OBL/2) - sin(robot_angles[2])*(-self.OBW/2) + self.C_W + rand_coord3[0]), int(sin(robot_angles[2])*(-self.OBL/2) + cos(robot_angles[2])*(-self.OBW/2) + self.C_H + rand_coord3[1])],
                            [int(cos(robot_angles[2])*(-self.OBL/2) - sin(robot_angles[2])*self.OBW/2 + self.C_W + rand_coord3[0]),    int(sin(robot_angles[2])*(-self.OBL/2) + cos(robot_angles[2])*self.OBW/2 + self.C_H + rand_coord3[1])]]        
            pts3_ob = np.array(self.robot3_ob, np.int32)
            cv2.fillPoly(self.robot3_region, [pts3_ob], 255)
            common_part13 = cv2.bitwise_and(self.robot1_region, self.robot3_region)
            common_part23 = cv2.bitwise_and(self.robot2_region, self.robot3_region)
            sum1 = cv2.countNonZero(common_part13)
            sum2 = cv2.countNonZero(common_part23)
            if(sum1 == 0 and sum2 == 0):
                self.res_observation[6] = rand_coord3[0]*self.pix2m
                self.res_observation[7] = rand_coord3[1]*self.pix2m
                self.res_observation[8] = robot_angles[2]
                self.set_robot_3_state.pose.position.x = self.res_observation[6]
                self.set_robot_3_state.pose.position.y = self.res_observation[7]
                self.set_robot_3_state.pose.orientation.z = sin(self.res_observation[8]/2)
                self.set_robot_3_state.pose.orientation.w = cos(self.res_observation[8]/2)
                break

        self.robot_1_state = SetEntityState.Request()
        self.robot_1_state._state = self.set_robot_1_state
        self.robot_2_state = SetEntityState.Request()
        self.robot_2_state._state = self.set_robot_2_state
        self.robot_3_state = SetEntityState.Request()
        self.robot_3_state._state = self.set_robot_3_state

        while not self.set_state.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('reset : service not available, waiting again...')

        try:
            self.set_state.call_async(self.robot_1_state)
            self.set_state.call_async(self.robot_2_state)
            self.set_state.call_async(self.robot_3_state)
        except rclpy.ServiceException as e:
            self.get_logger().info("/gazebo/reset_simulation service call failed")

        # creating observations for newral network
        self.state[0] = self.res_observation[0]/self.boundary_x
        self.state[1] = self.res_observation[1]/self.boundary_y
        self.state[2] = self.res_observation[2]/pi
        self.state[3] = self.res_observation[3]/self.boundary_x
        self.state[4] = self.res_observation[4]/self.boundary_y
        self.state[5] = self.res_observation[5]/pi
        self.state[6] = self.res_observation[6]/self.boundary_x
        self.state[7] = self.res_observation[7]/self.boundary_y
        self.state[8] = self.res_observation[8]/pi
        self.state[9] = (self.res_observation[0] - arrow_positions[0][0])/self.boundary_x
        self.state[10] = (self.res_observation[1] - arrow_positions[0][1])/self.boundary_y
        self.state[11] = (self.res_observation[2] - arrow_positions[0][2])/pi
        self.state[12] = (self.res_observation[3] - arrow_positions[1][0])/self.boundary_x
        self.state[13] = (self.res_observation[4] - arrow_positions[1][1])/self.boundary_y
        self.state[14] = (self.res_observation[5] - arrow_positions[1][2])/pi
        self.state[15] = (self.res_observation[6] - arrow_positions[2][0])/self.boundary_x
        self.state[16] = (self.res_observation[7] - arrow_positions[2][1])/self.boundary_y
        self.state[17] = (self.res_observation[8] - arrow_positions[2][2])/pi
        
        # robot1
        total_observation[0][:] = self.state[:12]
        
        # robot2
        total_observation[1][:3] = self.state[3:6]
        total_observation[1][3:6] = self.state[:3]
        total_observation[1][6:9] = self.state[6:9]
        total_observation[1][9:12] = self.state[12:15]

        # robot3
        total_observation[2][:3] = self.state[6:9]
        total_observation[2][3:9] = self.state[:6]
        total_observation[2][9:12] = self.state[15:18]        

        return self.state, total_observation # return state, obs

    def _get_actions_one_hot(self, actions):
        actions_one_hot = []
        for action in actions:
            one_hot = self.actions_one_hot_transform(action)
            actions_one_hot.append(one_hot)
        return np.array(actions_one_hot)

    def get_available_actions(self):
        available_actions = []
        for agent_id in range(self.n_agents):
            available_actions.append(gz_env.get_avail_agent_actions(agent_id))
        return np.array(available_actions)
    
    def get_avail_agent_actions(self, agent_id):
        """Returns the available actions for agent_id."""
        avail_actions = [1,1,1,1,1,1,1,1,1,1,1]
        return avail_actions


class Get_modelstate(Node):

    def __init__(self):
        super().__init__('get_modelstate')
        self.subscription = self.create_subscription(
            ModelStates,
            '/gazebo/model_states',
            self.listener_callback,
            10)
        self.subscription

    def listener_callback(self, data):
        global observation

        robot_1_id = data.name.index('robot_1')
        robot_2_id = data.name.index('robot_2')
        robot_3_id = data.name.index('robot_3')

        observation[0] = data.pose[robot_1_id].position.x
        observation[1] = data.pose[robot_1_id].position.y
        q0 = data.pose[robot_1_id].orientation.x
        q1 = data.pose[robot_1_id].orientation.y
        q2 = data.pose[robot_1_id].orientation.z
        q3 = data.pose[robot_1_id].orientation.w
        observation[2] = -atan2(2*(q0*q1 + q2*q3), (q0**2 - q1**2 - q2**2 + q3**2))

        observation[3] = data.pose[robot_2_id].position.x
        observation[4] = data.pose[robot_2_id].position.y
        q0 = data.pose[robot_2_id].orientation.x
        q1 = data.pose[robot_2_id].orientation.y
        q2 = data.pose[robot_2_id].orientation.z
        q3 = data.pose[robot_2_id].orientation.w
        observation[5] = -atan2(2*(q0*q1 + q2*q3), (q0**2 - q1**2 - q2**2 + q3**2))

        observation[6] = data.pose[robot_3_id].position.x
        observation[7] = data.pose[robot_3_id].position.y
        q0 = data.pose[robot_3_id].orientation.x
        q1 = data.pose[robot_3_id].orientation.y
        q2 = data.pose[robot_3_id].orientation.z
        q3 = data.pose[robot_3_id].orientation.w
        observation[8] = -atan2(2*(q0*q1 + q2*q3), (q0**2 - q1**2 - q2**2 + q3**2))


class Clash_calculation(Node):

    def __init__(self):
        super().__init__('clash_calculation')
        self.time_interval = 0.05

        self.image_L = 900
        self.H = self.image_L
        self.W = self.image_L
        self.C_H = int(self.H/2) 
        self.C_W = int(self.W/2)
        self.pix2m = 0.02
        self.robot_L = int(2.8/self.pix2m) 
        self.robot_W = int(1.3/self.pix2m) 
  
        margin = int(0.8/self.pix2m)
        self.OBL = self.robot_L + 2*margin # outer_boundary_Length
        self.OBW = self.robot_W + 2*margin # outer_boundary_Width
        
        self.image = np.zeros((self.H, self.W, 3), np.uint8)
        self.robot1_region = np.zeros((self.H, self.W), np.uint8)
        self.robot2_region = np.zeros((self.H, self.W), np.uint8)
        self.robot3_region = np.zeros((self.H, self.W), np.uint8)
        self.outer_region = np.zeros((self.H, self.W), np.uint8)

        self.pub_sum_img = self.create_publisher(Image, '/sum_image', 10)
        self.pub_common_part_img = self.create_publisher(Image, '/common_part_image', 10)
        self.timer = self.create_timer(self.time_interval, self.timer_callback)

    def timer_callback(self):
        global observation, clash_sum, arrow_positions

        self.image[:,:,:,] = 255
        self.image[0:100, :, :] = 0
        self.image[800:900, :, :] = 0
        self.image[:, 0:100, :] = 0
        self.image[:, 800:900, :] = 0

        self.robot1_region[:,:] = 0
        self.robot2_region[:,:] = 0
        self.robot3_region[:,:] = 0
        self.outer_region[:,:] = 255
        self.outer_region[100:800, 100:800] = 0 # outer boundary is 255

        #### ROBOT1
        #outer boundary1
        theta1 = observation[2]
        self.robot1_ob = [[int(cos(theta1)*self.OBL/2 - sin(theta1)*self.OBW/2 + self.C_W + observation[0]/self.pix2m),       int(sin(theta1)*self.OBL/2 + cos(theta1)*self.OBW/2 + self.C_H - observation[1]/self.pix2m)],
                        [int(cos(theta1)*self.OBL/2 - sin(theta1)*(-self.OBW/2) + self.C_W + observation[0]/self.pix2m),    int(sin(theta1)*self.OBL/2 + cos(theta1)*(-self.OBW/2) + self.C_H - observation[1]/self.pix2m)],
                        [int(cos(theta1)*(-self.OBL/2) - sin(theta1)*(-self.OBW/2) + self.C_W + observation[0]/self.pix2m), int(sin(theta1)*(-self.OBL/2) + cos(theta1)*(-self.OBW/2) + self.C_H - observation[1]/self.pix2m)],
                        [int(cos(theta1)*(-self.OBL/2) - sin(theta1)*self.OBW/2 + self.C_W + observation[0]/self.pix2m),    int(sin(theta1)*(-self.OBL/2) + cos(theta1)*self.OBW/2 + self.C_H - observation[1]/self.pix2m)]]        
        pts1_ob = np.array(self.robot1_ob, np.int32)
        cv2.fillPoly(self.image, [pts1_ob], (80, 80, 80))
        cv2.fillPoly(self.robot1_region, [pts1_ob], 255)
        # vehicle1
        self.robot1_coord = [[int(cos(theta1)*self.robot_L/2 - sin(theta1)*self.robot_W/2 + self.C_W + observation[0]/self.pix2m),       int(sin(theta1)*self.robot_L/2 + cos(theta1)*self.robot_W/2 + self.C_H - observation[1]/self.pix2m)],
                           [int(cos(theta1)*self.robot_L/2 - sin(theta1)*(-self.robot_W/2) + self.C_W + observation[0]/self.pix2m),    int(sin(theta1)*self.robot_L/2 + cos(theta1)*(-self.robot_W/2) + self.C_H - observation[1]/self.pix2m)],
                           [int(cos(theta1)*(-self.robot_L/2) - sin(theta1)*(-self.robot_W/2) + self.C_W + observation[0]/self.pix2m), int(sin(theta1)*(-self.robot_L/2) + cos(theta1)*(-self.robot_W/2) + self.C_H - observation[1]/self.pix2m)],
                           [int(cos(theta1)*(-self.robot_L/2) - sin(theta1)*self.robot_W/2 + self.C_W + observation[0]/self.pix2m),    int(sin(theta1)*(-self.robot_L/2) + cos(theta1)*self.robot_W/2 + self.C_H - observation[1]/self.pix2m)]]
        pts1 = np.array(self.robot1_coord, np.int32)
        robot1_c = np.sum(pts1, axis=0)/4
        cv2.fillPoly(self.image, [pts1], (255, 165, 0))
        cv2.line(self.image, (int(cos(theta1)*self.robot_L/2 - sin(theta1)*self.robot_W/2 + self.C_W + observation[0]/self.pix2m), int(sin(theta1)*self.robot_L/2 + cos(theta1)*self.robot_W/2 + self.C_H - observation[1]/self.pix2m)),
                             (int(cos(theta1)*self.robot_L/2 - sin(theta1)*(-self.robot_W/2) + self.C_W + observation[0]/self.pix2m), int(sin(theta1)*self.robot_L/2 + cos(theta1)*(-self.robot_W/2) + self.C_H - observation[1]/self.pix2m)),
                              color=(0, 0, 255), thickness=3, lineType=cv2.LINE_4, shift=0)
        
        #### ROBOT2
        #outer boundary2
        theta2 = observation[5]
        self.robot2_ob = [[int(cos(theta2)*self.OBL/2 - sin(theta2)*self.OBW/2 + self.C_W + observation[3]/self.pix2m),       int(sin(theta2)*self.OBL/2 + cos(theta2)*self.OBW/2 + self.C_H - observation[4]/self.pix2m)],
                        [int(cos(theta2)*self.OBL/2 - sin(theta2)*(-self.OBW/2) + self.C_W + observation[3]/self.pix2m),    int(sin(theta2)*self.OBL/2 + cos(theta2)*(-self.OBW/2) + self.C_H - observation[4]/self.pix2m)],
                        [int(cos(theta2)*(-self.OBL/2) - sin(theta2)*(-self.OBW/2) + self.C_W + observation[3]/self.pix2m), int(sin(theta2)*(-self.OBL/2) + cos(theta2)*(-self.OBW/2) + self.C_H - observation[4]/self.pix2m)],
                        [int(cos(theta2)*(-self.OBL/2) - sin(theta2)*self.OBW/2 + self.C_W + observation[3]/self.pix2m),    int(sin(theta2)*(-self.OBL/2) + cos(theta2)*self.OBW/2 + self.C_H - observation[4]/self.pix2m)]]        
        pts2_ob = np.array(self.robot2_ob, np.int32)
        cv2.fillPoly(self.image, [pts2_ob], (80, 80, 80))
        cv2.fillPoly(self.robot2_region, [pts2_ob], 255)
        # vehicle2
        self.robot2_coord = [[int(cos(theta2)*self.robot_L/2 - sin(theta2)*self.robot_W/2 + self.C_W + observation[3]/self.pix2m),       int(sin(theta2)*self.robot_L/2 + cos(theta2)*self.robot_W/2 + self.C_H - observation[4]/self.pix2m)],
                           [int(cos(theta2)*self.robot_L/2 - sin(theta2)*(-self.robot_W/2) + self.C_W + observation[3]/self.pix2m),    int(sin(theta2)*self.robot_L/2 + cos(theta2)*(-self.robot_W/2) + self.C_H - observation[4]/self.pix2m)],
                           [int(cos(theta2)*(-self.robot_L/2) - sin(theta2)*(-self.robot_W/2) + self.C_W + observation[3]/self.pix2m), int(sin(theta2)*(-self.robot_L/2) + cos(theta2)*(-self.robot_W/2) + self.C_H - observation[4]/self.pix2m)],
                           [int(cos(theta2)*(-self.robot_L/2) - sin(theta2)*self.robot_W/2 + self.C_W + observation[3]/self.pix2m),    int(sin(theta2)*(-self.robot_L/2) + cos(theta2)*self.robot_W/2 + self.C_H - observation[4]/self.pix2m)]]
        pts2 = np.array(self.robot2_coord, np.int32)
        robot2_c = np.sum(pts2, axis=0)/4
        cv2.fillPoly(self.image, [pts2], (166, 171, 248))
        cv2.line(self.image, (int(cos(theta2)*self.robot_L/2 - sin(theta2)*self.robot_W/2 + self.C_W + observation[3]/self.pix2m), int(sin(theta2)*self.robot_L/2 + cos(theta2)*self.robot_W/2 + self.C_H - observation[4]/self.pix2m)),
                             (int(cos(theta2)*self.robot_L/2 - sin(theta2)*(-self.robot_W/2) + self.C_W + observation[3]/self.pix2m), int(sin(theta2)*self.robot_L/2 + cos(theta2)*(-self.robot_W/2) + self.C_H - observation[4]/self.pix2m)),
                              color=(0, 0, 255), thickness=3, lineType=cv2.LINE_4, shift=0)     

        #### ROBOT3
        #outer boundary3
        theta3 = observation[8]
        self.robot3_ob = [[int(cos(theta3)*self.OBL/2 - sin(theta3)*self.OBW/2 + self.C_W + observation[6]/self.pix2m),       int(sin(theta3)*self.OBL/2 + cos(theta3)*self.OBW/2 + self.C_H - observation[7]/self.pix2m)],
                        [int(cos(theta3)*self.OBL/2 - sin(theta3)*(-self.OBW/2) + self.C_W + observation[6]/self.pix2m),    int(sin(theta3)*self.OBL/2 + cos(theta3)*(-self.OBW/2) + self.C_H - observation[7]/self.pix2m)],
                        [int(cos(theta3)*(-self.OBL/2) - sin(theta3)*(-self.OBW/2) + self.C_W + observation[6]/self.pix2m), int(sin(theta3)*(-self.OBL/2) + cos(theta3)*(-self.OBW/2) + self.C_H - observation[7]/self.pix2m)],
                        [int(cos(theta3)*(-self.OBL/2) - sin(theta3)*self.OBW/2 + self.C_W + observation[6]/self.pix2m),    int(sin(theta3)*(-self.OBL/2) + cos(theta3)*self.OBW/2 + self.C_H - observation[7]/self.pix2m)]]        
        pts3_ob = np.array(self.robot3_ob, np.int32)
        cv2.fillPoly(self.image, [pts3_ob], (80, 80, 80))
        cv2.fillPoly(self.robot3_region, [pts3_ob], 255)
        # vehicle3
        self.robot3_coord = [[int(cos(theta3)*self.robot_L/2 - sin(theta3)*self.robot_W/2 + self.C_W + observation[6]/self.pix2m),       int(sin(theta3)*self.robot_L/2 + cos(theta3)*self.robot_W/2 + self.C_H - observation[7]/self.pix2m)],
                           [int(cos(theta3)*self.robot_L/2 - sin(theta3)*(-self.robot_W/2) + self.C_W + observation[6]/self.pix2m),    int(sin(theta3)*self.robot_L/2 + cos(theta3)*(-self.robot_W/2) + self.C_H - observation[7]/self.pix2m)],
                           [int(cos(theta3)*(-self.robot_L/2) - sin(theta3)*(-self.robot_W/2) + self.C_W + observation[6]/self.pix2m), int(sin(theta3)*(-self.robot_L/2) + cos(theta3)*(-self.robot_W/2) + self.C_H - observation[7]/self.pix2m)],
                           [int(cos(theta3)*(-self.robot_L/2) - sin(theta3)*self.robot_W/2 + self.C_W + observation[6]/self.pix2m),    int(sin(theta3)*(-self.robot_L/2) + cos(theta3)*self.robot_W/2 + self.C_H - observation[7]/self.pix2m)]]
        pts3 = np.array(self.robot3_coord, np.int32)
        robot3_c = np.sum(pts3, axis=0)/4
        cv2.fillPoly(self.image, [pts3], (51, 204, 153))
        cv2.line(self.image, (int(cos(theta3)*self.robot_L/2 - sin(theta3)*self.robot_W/2 + self.C_W + observation[6]/self.pix2m),    int(sin(theta3)*self.robot_L/2 + cos(theta3)*self.robot_W/2 + self.C_H - observation[7]/self.pix2m)),
                             (int(cos(theta3)*self.robot_L/2 - sin(theta3)*(-self.robot_W/2) + self.C_W + observation[6]/self.pix2m), int(sin(theta3)*self.robot_L/2 + cos(theta3)*(-self.robot_W/2) + self.C_H - observation[7]/self.pix2m)),
                              color=(0, 0, 255), thickness=3, lineType=cv2.LINE_4, shift=0) 
        
        common_part12 = cv2.bitwise_and(self.robot1_region, self.robot2_region)
        common_part13 = cv2.bitwise_and(self.robot1_region, self.robot3_region)
        common_part23 = cv2.bitwise_and(self.robot2_region, self.robot3_region)
        common_robot1_out = cv2.bitwise_and(self.robot1_region, self.outer_region)
        common_robot2_out = cv2.bitwise_and(self.robot2_region, self.outer_region)
        common_robot3_out = cv2.bitwise_and(self.robot3_region, self.outer_region)
        temp1 = cv2.bitwise_or(common_part12, common_part13)
        temp2 = cv2.bitwise_or(temp1, common_part23)
        temp3 = cv2.bitwise_or(temp2, common_robot1_out)
        temp4 = cv2.bitwise_or(temp3, common_robot2_out)
        result_img = cv2.bitwise_or(temp4, common_robot3_out)
        clash_sum = cv2.countNonZero(result_img)

        cv2.putText(self.image, 'R1', (int(robot1_c[0]), int(robot1_c[1])), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(self.image, 'R2', (int(robot2_c[0]), int(robot2_c[1])), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(self.image, 'R3', (int(robot3_c[0]), int(robot3_c[1])), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.circle(self.image, center=(int(self.C_W + arrow_positions[0][0]/self.pix2m), int(self.C_H + arrow_positions[0][1]/self.pix2m)), radius=10, color=(255, 0, 0), thickness=cv2.FILLED)
        cv2.circle(self.image, center=(int(self.C_W + arrow_positions[1][0]/self.pix2m), int(self.C_H + arrow_positions[1][1]/self.pix2m)), radius=10, color=(0, 0, 255), thickness=cv2.FILLED)
        cv2.circle(self.image, center=(int(self.C_W + arrow_positions[2][0]/self.pix2m), int(self.C_H + arrow_positions[2][1]/self.pix2m)), radius=10, color=(0, 255, 0), thickness=cv2.FILLED)        
        
        if(arrow_positions[0][2] == 0):
            cv2.line(self.image, (int(self.C_W + arrow_positions[0][0]/self.pix2m),      int(self.C_H + arrow_positions[0][1]/self.pix2m)),
                                 (int(self.C_W + arrow_positions[0][0]/self.pix2m) + 20, int(self.C_H + arrow_positions[0][1]/self.pix2m)),
                                 color=(255, 0, 0), thickness=3, lineType=cv2.LINE_4, shift=0) 
        else:
            cv2.line(self.image, (int(self.C_W + arrow_positions[0][0]/self.pix2m),      int(self.C_H + arrow_positions[0][1]/self.pix2m)),
                                 (int(self.C_W + arrow_positions[0][0]/self.pix2m) - 20, int(self.C_H + arrow_positions[0][1]/self.pix2m)),
                                 color=(255, 0, 0), thickness=3, lineType=cv2.LINE_4, shift=0) 

        if(arrow_positions[1][2] == 0):
            cv2.line(self.image, (int(self.C_W + arrow_positions[1][0]/self.pix2m),      int(self.C_H + arrow_positions[1][1]/self.pix2m)),
                                 (int(self.C_W + arrow_positions[1][0]/self.pix2m) + 20, int(self.C_H + arrow_positions[1][1]/self.pix2m)),
                                 color=(0, 0, 255), thickness=3, lineType=cv2.LINE_4, shift=0) 
        else:
            cv2.line(self.image, (int(self.C_W + arrow_positions[1][0]/self.pix2m),      int(self.C_H + arrow_positions[1][1]/self.pix2m)),
                                 (int(self.C_W + arrow_positions[1][0]/self.pix2m) - 20, int(self.C_H + arrow_positions[1][1]/self.pix2m)),
                                 color=(0, 0, 255), thickness=3, lineType=cv2.LINE_4, shift=0) 
            
        if(arrow_positions[2][2] == 0):
            cv2.line(self.image, (int(self.C_W + arrow_positions[2][0]/self.pix2m),      int(self.C_H + arrow_positions[2][1]/self.pix2m)),
                                 (int(self.C_W + arrow_positions[2][0]/self.pix2m) + 20, int(self.C_H + arrow_positions[2][1]/self.pix2m)),
                                 color=(0, 255, 0), thickness=3, lineType=cv2.LINE_4, shift=0) 
        else:
            cv2.line(self.image, (int(self.C_W + arrow_positions[2][0]/self.pix2m),      int(self.C_H + arrow_positions[2][1]/self.pix2m)),
                                 (int(self.C_W + arrow_positions[2][0]/self.pix2m) - 20, int(self.C_H + arrow_positions[2][1]/self.pix2m)),
                                 color=(0, 255, 0), thickness=3, lineType=cv2.LINE_4, shift=0) 

        img_msg = bridge.cv2_to_imgmsg(self.image)  
        img_common_part = bridge.cv2_to_imgmsg(result_img)  
        self.pub_common_part_img.publish(img_common_part)
        self.pub_sum_img.publish(img_msg) 
   

if __name__ == '__main__':
    rclpy.init(args=None)

    gz_env = GazeboEnv()
    get_modelstate = Get_modelstate()
    clash_calculation = Clash_calculation()

    executor = rclpy.executors.MultiThreadedExecutor()
    executor.add_node(gz_env)
    executor.add_node(get_modelstate)
    executor.add_node(clash_calculation)

    executor_thread = threading.Thread(target=executor.spin, daemon=True)
    executor_thread.start()

    vdn_config = VDNConfig()
    common_args = get_common_args()
    args = argparse.Namespace(**vars(common_args), **vars(vdn_config))
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    args.episode_limit = 750 #15sec
    args.obs_shape = 12
    args.state_shape = 18
    args.n_agents = 3
    args.n_actions = 11
    args.device = device

    # init the logger before other steps
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    # log
    log_name = os.path.join(args.project, args.scenario, args.algo_name, timestamp).replace(os.path.sep, '_')
    log_path = os.path.join(args.log_dir, args.project, args.scenario, args.algo_name)
    tensorboard_log_path = get_outdir(log_path, 'tensorboard_log_dir')
    log_file = os.path.join(log_path, log_name + '.log')
    text_logger = get_root_logger(log_file=log_file, log_level='INFO')
    model_path = os.getcwd() + '/cooperative_marl_ros/src/robot_control/scripts/vdn_models'

    writer = SummaryWriter(tensorboard_log_path)
    writer.add_text('args', str(args))
    logger = TensorboardLogger(writer)

    rpm = ReplayBuffer(
        max_size=args.replay_buffer_size,
        episode_limit=args.episode_limit,
        state_shape=args.state_shape,
        obs_shape=args.obs_shape,
        num_agents=args.n_agents,
        num_actions=args.n_actions,
        batch_size=args.batch_size,
        device=device)

    agent_model = RNNActor(
        input_shape=args.obs_shape,
        rnn_hidden_dim=args.rnn_hidden_dim,
        n_actions=args.n_actions,
    )

    mixer_model = VDNMixer()

    marl_agent = VDNAgent(
        agent_model=agent_model,
        mixer_model=mixer_model,
        n_agents=args.n_agents,
        double_q=args.double_q,
        total_steps=args.total_steps,
        gamma=args.gamma,
        learning_rate=args.learning_rate,
        min_learning_rate=args.min_learning_rate,
        exploration_start=args.exploration_start,
        min_exploration=args.min_exploration,
        update_target_interval=args.update_target_interval,
        update_learner_freq=args.update_learner_freq,
        clip_grad_norm=args.clip_grad_norm,
        device=args.device,
    )

    steps_cnt = 0
    episode_cnt = 0
    marl_agent.restore(model_path)

    rate = gz_env.create_rate(2)
    try:
        while rclpy.ok():
            eval_rewards, eval_steps = run_evaluate_episode(
                gz_env, marl_agent, num_eval_episodes=5)

    except KeyboardInterrupt:
        print("KeyboardInterrupt")

    rclpy.shutdown()
    executor_thread.join()

