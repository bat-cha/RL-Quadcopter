"""Hover task."""
from collections import deque

import numpy as np
from gym import spaces
from geometry_msgs.msg import Vector3, Point, Quaternion, Pose, Twist, Wrench
from quad_controller_rl.tasks.base_task import BaseTask


class Hover(BaseTask):

    def __init__(self):
        # State space: <position_x, .._y, .._z, orientation_x, .._y, .._z, .._w>
        cube_size = 300.0  # env is cube_size x cube_size x cube_size
        self.observation_space = spaces.Box(
            np.array([- cube_size / 2, - cube_size / 2, 0.0, -1.0, -1.0, -1.0, -1.0]),
            np.array([cube_size / 2, cube_size / 2, cube_size, 1.0, 1.0, 1.0, 1.0]))
        print("Hover(): observation_space = {}".format(self.observation_space))  # [debug]
        self.observation_space_range = self.observation_space.high - self.observation_space.low

        # Action space: <force_x, .._y, .._z, torque_x, .._y, .._z>
        max_force = 25.0
        max_torque = 25.0
        self.action_space = spaces.Box(
            np.array([-max_force, -max_force, -max_force, -max_torque, -max_torque, -max_torque]),
            np.array([max_force, max_force, max_force, max_torque, max_torque, max_torque]))
        print("Hover(): action_space = {}".format(self.action_space))  # [debug]
        self.action_space_range = self.action_space.high - self.action_space.low

        # Task-specific parameters
        self.max_duration = 10.0  # secs
        self.count = 0
        self.action = None
        self.last_time = 0.0
        self.start = np.array([0.0, 0.0, 10.0])
        self.target = np.array([0.0, 0.0, 10.0])
        self.last_position = np.array([0.0, 0.0, 0.0])

    def reset(self):
        self.action = None
        self.last_time = 0.0
        z_start = np.random.normal(1.0, 0.5)
        self.start[2] = z_start
        self.last_position = self.start
        return Pose(
            position=Point(0.0, 0.0, self.start[2]),
            orientation=Quaternion(0.0, 0.0, 0.0, 0.0),
        ), Twist(
            linear=Vector3(0.0, 0.0, 0.0),
            angular=Vector3(0.0, 0.0, 0.0)
        )

    def update(self, timestamp, pose, angular_velocity, linear_acceleration):
        state = np.array([
            pose.position.x, pose.position.y, pose.position.z,
            pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w])

        current_position = np.array([pose.position.x, pose.position.y, pose.position.z])
        distance_to_target = np.linalg.norm(self.target - current_position)

        self.last_position = current_position

        acceleration = np.array([linear_acceleration.x, linear_acceleration.y, linear_acceleration.z])
        sum_acceleration = np.linalg.norm(acceleration)
        done = False
        reward = (5. - distance_to_target) * 0.3 - sum_acceleration * 0.05
        print('reward={:.4} distance_to_target={:.4}'.format(reward, distance_to_target), end='\r')

        if timestamp > self.max_duration:  # agent has run out of time
            done = True

        self.action = self.agent.step(state, reward, done)

        # Convert to proper force command (a Wrench object) and return it
        action = self.action * self.action_space_range[2] * 0.5  # only z force
        if action is not None:
            action = np.clip(action.flatten(), self.action_space.low,
                             self.action_space.high)  # flatten, clamp to action space limits
            return Wrench(
                force=Vector3(action[0], action[1], action[2]),
                torque=Vector3(action[3], action[4], action[5])
            ), done
        else:
            return Wrench(), done