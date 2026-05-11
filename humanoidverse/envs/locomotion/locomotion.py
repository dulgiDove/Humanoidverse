from time import time
from warnings import WarningMessage
import numpy as np
import os

from humanoidverse.utils.torch_utils import *
# from isaacgym import gymtorch, gymapi, gymutil

import torch
from torch import Tensor
from typing import Tuple, Dict
from rich.progress import Progress

from humanoidverse.envs.env_utils.general import class_to_dict
from humanoidverse.utils.spatial_utils.rotations import quat_apply_yaw, wrap_to_pi
from humanoidverse.envs.legged_base_task.legged_robot_base import LeggedRobotBase
# from humanoidverse.envs.env_utils.command_generator import CommandGenerator


class LeggedRobotLocomotion(LeggedRobotBase):
    def __init__(self, config, device):
        self.init_done = False
        super().__init__(config, device)
        self.init_done = True
        # import ipdb; ipdb.set_trace()
        if config.robot.motion.get("hips_link", None):
            self.hips_dof_id = [self.simulator._body_list.index(link) - 1 for link in config.robot.motion.hips_link] # Yuanhang: -1 for the base link (pelvis)

    def _init_buffers(self):
        super()._init_buffers()
        self.commands = torch.zeros(
            (self.num_envs, 4), dtype=torch.float32, device=self.device
        )
        self.command_ranges = self.config.locomotion_command_ranges
        self.target_pos = torch.zeros((self.num_envs, 2), dtype=torch.float32, device=self.device)
        # Stage3: 장애물 버퍼
        self.obstacle_pos = torch.zeros((self.num_envs, 3, 2), dtype=torch.float32, device=self.device)
        self.obstacle_radius = 0.3
        self.num_scan_rays = 36
        self.lidar_max_range = 5.0

    def _setup_simulator_control(self):
        self.simulator.commands = self.commands

    def _update_tasks_callback(self):
        """ Callback called before computing terminations, rewards, and observations
            Default behaviour: Compute ang vel command based on target and heading, compute measured terrain heights and randomly push robots
        """
        # 
        super()._update_tasks_callback()

        # commands
        if not self.is_evaluating:
            env_ids = (self.episode_length_buf % int(self.config.locomotion_command_resampling_time / self.dt)==0).nonzero(as_tuple=False).flatten()
            #self._resample_commands(env_ids)
        robot_xy = self.simulator.robot_root_states[:, :2]
        to_target = self.target_pos - robot_xy
        target_heading = torch.atan2(to_target[:, 1], to_target[:, 0])
        self.commands[:, 3] = target_heading
        self.commands[:, 0] = 1.0
        self.commands[:, 1] = 0.0 
        forward = quat_apply(self.base_quat, self.forward_vec)
        heading = torch.atan2(forward[:, 1], forward[:, 0])
        self.commands[:, 2] = torch.clip(
            3.0 * wrap_to_pi(self.commands[:, 3] - heading), 
            self.command_ranges["ang_vel_yaw"][0], 
            self.command_ranges["ang_vel_yaw"][1]
        )
        dist = torch.norm(to_target, dim=1)
        reached = (dist < 0.5).nonzero(as_tuple=False).flatten()
        if len(reached) > 0:
            logger.info(f"[GOAL REACHED] env {reached.tolist()} | dist: {dist[reached].tolist()}")
            self._resample_target(reached)
            if self.num_envs == 1:
                self._resample_obstacles(reached)

        if self.num_envs == 1:
            tx = self.target_pos[0, 0].item()
            ty = self.target_pos[0, 1].item()
            self.simulator.target_pos_for_cam = (tx, ty)

    def _resample_target(self, env_ids):
        robot_pos = self.simulator.robot_root_states[env_ids, :2]
        rand_dist = torch_rand_float(3.0, 8.0, (len(env_ids), 1), device=self.device).squeeze(1)
        rand_angle = torch_rand_float(-3.14159, 3.14159, (len(env_ids), 1), device=self.device).squeeze(1)
        self.target_pos[env_ids, 0] = robot_pos[:, 0] + rand_dist * torch.cos(rand_angle)
        self.target_pos[env_ids, 1] = robot_pos[:, 1] + rand_dist * torch.sin(rand_angle)

    def _resample_commands(self, env_ids):
        self.commands[env_ids, 0] = torch_rand_float(self.command_ranges["lin_vel_x"][0], self.command_ranges["lin_vel_x"][1], (len(env_ids), 1), device=str(self.device)).squeeze(1)
        self.commands[env_ids, 1] = torch_rand_float(self.command_ranges["lin_vel_y"][0], self.command_ranges["lin_vel_y"][1], (len(env_ids), 1), device=str(self.device)).squeeze(1)
        self.commands[env_ids, 3] = torch_rand_float(self.command_ranges["heading"][0], self.command_ranges["heading"][1], (len(env_ids), 1), device=self.device).squeeze(1)

        # set small commands to zero
        self.commands[env_ids, :2] *= (torch.norm(self.commands[env_ids, :2], dim=1) > 0.2).unsqueeze(1)


    def _reset_tasks_callback(self, env_ids):
        super()._reset_tasks_callback(env_ids)
        #if not self.is_evaluating:
            #self._resample_commands(env_ids)
        self._resample_target(env_ids)
        self._resample_obstacles(env_ids)

    def _resample_obstacles(self, env_ids):
        robot_pos = self.simulator.robot_root_states[env_ids, :2]
        for i in range(3):
            rand_dist = torch_rand_float(1.5, 5.0, (len(env_ids), 1), device=self.device).squeeze(1)
            rand_angle = torch_rand_float(-3.14159, 3.14159, (len(env_ids), 1), device=self.device).squeeze(1)
            ox = robot_pos[:, 0] + rand_dist * torch.cos(rand_angle)
            oy = robot_pos[:, 1] + rand_dist * torch.sin(rand_angle)
            self.obstacle_pos[env_ids, i, 0] = ox
            self.obstacle_pos[env_ids, i, 1] = oy
        # num_envs=1일 때 마커 위치 업데이트
        if self.num_envs == 1:
            for i in range(3):
                ox = self.obstacle_pos[0, i, 0].item()
                oy = self.obstacle_pos[0, i, 1].item()
                self.simulator.obstacle_markers[i].set_pos([[ox, oy, 0.76]])
            tx = self.target_pos[0, 0].item()
            ty = self.target_pos[0, 1].item()
            self.simulator.target_marker.set_pos([[tx, ty, 0.3]])

    def set_is_evaluating(self, command=None):
        super().set_is_evaluating()
        self.commands = torch.zeros((self.num_envs, 4), dtype=torch.float32, device=self.device)
        if command is not None:
            self.commands[:, :3] = torch.tensor(command).to(self.device)  # only set the first 3 commands

    ########################### TRACKING REWARDS ###########################

    def _reward_tracking_lin_vel(self):
        # Tracking of linear velocity commands (xy axes)
        lin_vel_error = torch.sum(torch.square(self.commands[:, :2] - self.base_lin_vel[:, :2]), dim=1)
        return torch.exp(-lin_vel_error/self.config.rewards.reward_tracking_sigma.lin_vel)
    
    def _reward_tracking_ang_vel(self):
        # Tracking of angular velocity commands (yaw) 
        ang_vel_error = torch.square(self.commands[:, 2] - self.base_ang_vel[:, 2])
        return torch.exp(-ang_vel_error/self.config.rewards.reward_tracking_sigma.ang_vel)

    def _reward_goal_reached(self):
        robot_xy = self.simulator.robot_root_states[:, :2]
        dist = torch.norm(self.target_pos - robot_xy, dim=1)
        return (dist < 0.5).float()

    ########################### PENALTY REWARDS ###########################

    def _reward_penalty_lin_vel_z(self):
        # Penalize z axis base linear velocity
        return torch.square(self.base_lin_vel[:, 2])
    
    def _reward_penalty_ang_vel_xy(self):
        # Penalize xy axes base angular velocity
        return torch.sum(torch.square(self.base_ang_vel[:, :2]), dim=1)

    def _reward_penalty_ang_vel_xy_torso(self):
        # Penalize xy axes base angular velocity

        torso_ang_vel = quat_rotate_inverse(self.simulator._rigid_body_rot[:, self.torso_index], self.simulator._rigid_body_ang_vel[:, self.torso_index])
        return torch.sum(torch.square(torso_ang_vel[:, :2]), dim=1)
    

    def _reward_penalty_feet_contact_forces(self):
        # penalize high contact forces
        return torch.sum((torch.norm(self.simulator.contact_forces[:, self.feet_indices, :], dim=-1) -  self.config.rewards.locomotion_max_contact_force).clip(min=0.), dim=1)

    ########################### FEET REWARDS ###########################

    def _reward_feet_air_time(self):
        # Reward long steps
        # Need to filter the contacts because the contact reporting of PhysX is unreliable on meshes
        contact = self.simulator.contact_forces[:, self.feet_indices, 2] > 1.
        contact_filt = torch.logical_or(contact, self.last_contacts) 
        self.last_contacts = contact
        first_contact = (self.feet_air_time > 0.) * contact_filt
        self.feet_air_time += self.dt
        rew_airTime = torch.sum((self.feet_air_time - 0.5) * first_contact, dim=1) # reward only on first contact with the ground
        rew_airTime *= torch.norm(self.commands[:, :2], dim=1) > 0.1 #no reward for zero command
        self.feet_air_time *= ~contact_filt
        return rew_airTime
    
    def _reward_penalty_in_the_air(self):
        contact = self.simulator.contact_forces[:, self.feet_indices, 2] > 1.
        contact_filt = torch.logical_or(contact, self.last_contacts) 
        first_foot_contact = contact_filt[:,0]
        second_foot_contact = contact_filt[:,1]
        reward = ~(first_foot_contact | second_foot_contact)
        return reward



    def _reward_penalty_stumble(self):
        # Penalize feet hitting vertical surfaces
        return torch.any(torch.norm(self.simulator.contact_forces[:, self.feet_indices, :2], dim=2) >\
             5 *torch.abs(self.simulator.contact_forces[:, self.feet_indices, 2]), dim=1)


    def _reward_penalty_feet_ori(self):
        left_quat = self.simulator._rigid_body_rot[:, self.feet_indices[0]]
        left_gravity = quat_rotate_inverse(left_quat, self.gravity_vec)
        right_quat = self.simulator._rigid_body_rot[:, self.feet_indices[1]]
        right_gravity = quat_rotate_inverse(right_quat, self.gravity_vec)
        return torch.sum(torch.square(left_gravity[:, :2]), dim=1)**0.5 + torch.sum(torch.square(right_gravity[:, :2]), dim=1)**0.5 

    def _reward_base_height(self):
        # Penalize base height away from target

        base_height = self.simulator.robot_root_states[:, 2]
        return torch.square(base_height - self.config.rewards.desired_base_height)

    def _reward_penalty_hip_pos(self):
        # Penalize the hip joints (only roll and yaw)
        hips_roll_yaw_indices = self.hips_dof_id[1:3] + self.hips_dof_id[4:6]
        hip_pos = self.simulator.dof_pos[:, hips_roll_yaw_indices]
        return torch.sum(torch.square(hip_pos), dim=1)

    def _reward_feet_heading_alignment(self):
        left_quat = self.simulator._rigid_body_rot[:, self.feet_indices[0]]
        right_quat = self.simulator._rigid_body_rot[:, self.feet_indices[1]]

        forward_left_feet = quat_apply(left_quat, self.forward_vec)
        heading_left_feet = torch.atan2(forward_left_feet[:, 1], forward_left_feet[:, 0])
        forward_right_feet = quat_apply(right_quat, self.forward_vec)
        heading_right_feet = torch.atan2(forward_right_feet[:, 1], forward_right_feet[:, 0])


        root_forward = quat_apply(self.base_quat, self.forward_vec)
        heading_root = torch.atan2(root_forward[:, 1], root_forward[:, 0])

        heading_diff_left = torch.abs(wrap_to_pi(heading_left_feet - heading_root))
        heading_diff_right = torch.abs(wrap_to_pi(heading_right_feet - heading_root))
        
        return heading_diff_left + heading_diff_right
    
    def _reward_feet_ori(self):
        left_quat = self.simulator._rigid_body_rot[:, self.feet_indices[0]]
        left_gravity = quat_rotate_inverse(left_quat, self.gravity_vec)
        right_quat = self.simulator._rigid_body_rot[:, self.feet_indices[1]]
        right_gravity = quat_rotate_inverse(right_quat, self.gravity_vec)
        return torch.sum(torch.square(left_gravity[:, :2]), dim=1)**0.5 + torch.sum(torch.square(right_gravity[:, :2]), dim=1)**0.5 

    def _reward_penalty_feet_slippage(self):
        # assert self.simulator._rigid_body_vel.shape[1] == 20
        foot_vel = self.simulator._rigid_body_vel[:, self.feet_indices]
        return torch.sum(torch.norm(foot_vel, dim=-1) * (torch.norm(self.simulator.contact_forces[:, self.feet_indices, :], dim=-1) > 1.), dim=1)
    

    def _reward_penalty_feet_height(self):
        # Penalize base height away from target
        feet_height = self.simulator._rigid_body_pos[:,self.feet_indices, 2]
        dif = torch.abs(feet_height - self.config.rewards.feet_height_target)
        dif = torch.min(dif, dim=1).values # [num_env], # select the foot closer to target 
        return torch.clip(dif - 0.02, min=0.) # target - 0.02 ~ target + 0.02 is acceptable 
    
    def _reward_penalty_close_feet_xy(self):
        # returns 1 if two feet are too close
        left_foot_xy = self.simulator._rigid_body_pos[:, self.feet_indices[0], :2]
        right_foot_xy = self.simulator._rigid_body_pos[:, self.feet_indices[1], :2]
        feet_distance_xy = torch.norm(left_foot_xy - right_foot_xy, dim=1)
        return (feet_distance_xy < self.config.rewards.close_feet_threshold) * 1.0
    

    def _reward_penalty_close_knees_xy(self):
        # returns 1 if two knees are too close
        left_knee_xy = self.simulator._rigid_body_pos[:, self.knee_indices[0], :2]
        right_knee_xy = self.simulator._rigid_body_pos[:, self.knee_indices[1], :2]
        self.knee_distance_xy = torch.norm(left_knee_xy - right_knee_xy, dim=1)
        return (self.knee_distance_xy < self.config.rewards.close_knees_threshold)* 1.0
    

    def _reward_upperbody_joint_angle_freeze(self):
        # returns keep the upper body joint angles close to the default
        assert self.config.robot.has_upper_body_dof
        deviation = torch.abs(self.simulator.dof_pos[:, self.upper_dof_indices] - self.default_dof_pos[:,self.upper_dof_indices])
        return torch.sum(deviation, dim=1)
    
    ######################### Observations #########################
    def _get_obs_command_lin_vel(self):
        return self.commands[:, :2]
    
    def _get_obs_command_ang_vel(self):
        return self.commands[:, 2:3]

    ######################### Stage3: LiDAR & Obstacle #########################
    def _compute_lidar_scan(self):
        robot_xy = self.simulator.robot_root_states[:, :2]
        forward = quat_apply(self.base_quat, self.forward_vec)
        robot_yaw = torch.atan2(forward[:, 1], forward[:, 0])
        angles = torch.linspace(0, 2 * 3.14159, self.num_scan_rays + 1,
                        device=self.device, dtype=torch.float32)[:-1]
        world_angles = robot_yaw.unsqueeze(1) + angles.unsqueeze(0)
        ray_dx = torch.cos(world_angles)
        ray_dy = torch.sin(world_angles)
        scan = torch.full((self.num_envs, self.num_scan_rays),
                          self.lidar_max_range, device=self.device)
        r = self.obstacle_radius
        for i in range(3):
            dx = self.obstacle_pos[:, i, 0] - robot_xy[:, 0]
            dy = self.obstacle_pos[:, i, 1] - robot_xy[:, 1]
            proj = dx.unsqueeze(1) * ray_dx + dy.unsqueeze(1) * ray_dy
            perp_sq = (dx.unsqueeze(1)**2 + dy.unsqueeze(1)**2) - proj**2
            valid = (proj > 0) & (perp_sq < r**2)
            hit_dist = proj - torch.sqrt(torch.clamp(r**2 - perp_sq, min=0.0))
            hit_dist = torch.clamp(hit_dist, min=0.0, max=self.lidar_max_range)
            scan = torch.where(valid & (hit_dist < scan), hit_dist, scan)
        return scan

    def _get_obs_lidar_scan(self):
        return self._compute_lidar_scan() / self.lidar_max_range

    def _reward_penalty_obstacle_collision(self):
        robot_xy = self.simulator.robot_root_states[:, :2]
        min_dist = torch.full((self.num_envs,), self.lidar_max_range, device=self.device)
        for i in range(3):
            d = torch.norm(robot_xy - self.obstacle_pos[:, i], dim=1)
            min_dist = torch.minimum(min_dist, d)
        return (min_dist < self.obstacle_radius + 0.3).float()