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
from loguru import logger

from humanoidverse.envs.env_utils.general import class_to_dict
from humanoidverse.utils.spatial_utils.rotations import quat_apply_yaw, wrap_to_pi
from humanoidverse.envs.legged_base_task.legged_robot_base import LeggedRobotBase


class LeggedRobotLocomotion(LeggedRobotBase):
    def __init__(self, config, device):
        self.init_done = False
        super().__init__(config, device)
        self.init_done = True
        if config.robot.motion.get("hips_link", None):
            self.hips_dof_id = [self.simulator._body_list.index(link) - 1 for link in config.robot.motion.hips_link]

    def _init_buffers(self):
        super()._init_buffers()
        # E2E: commands 텐서는 내부 참조용으로만 유지 (velocity 명령으로 쓰지 않음)
        self.commands = torch.zeros(
            (self.num_envs, 4), dtype=torch.float32, device=self.device
        )
        self.command_ranges = self.config.locomotion_command_ranges
        self.target_pos = torch.zeros((self.num_envs, 2), dtype=torch.float32, device=self.device)
        # 장애물 버퍼
        self.obstacle_pos = torch.zeros((self.num_envs, 3, 2), dtype=torch.float32, device=self.device)
        self.obstacle_radius = 0.3
        self.num_scan_rays = 36
        self.lidar_max_range = 5.0
        # Progress reward용: 이전 스텝 목표까지 거리 버퍼
        self.prev_dist_to_target = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        # 목표 도달 보너스용 1-step buffer
        # HumanoidVerse의 callback/reward 계산 순서가 바뀌어도 보너스가 누락되지 않도록 분리한다.
        self.goal_reached_reward_buf = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self.goal_reach_threshold = 0.5

        # Stable goal 판정 파라미터
        # goal 반경에 들어왔더라도 자세가 무너지거나 크게 흔들리면 성공으로 인정하지 않는다.
        self.goal_stable_min_height_ratio = 0.80
        self.goal_stable_max_orientation_error = 0.50
        self.goal_stable_max_ang_vel_xy = 2.50
        self.goal_stable_max_lin_vel_z = 1.00

        # Goal Distance Curriculum
        self.max_goal_dist = 1.5               # 초기 목표 거리 (m)
        self.min_goal_dist = 1.5               # 항상 1.5m ~ max_goal_dist 범위에서 샘플링
        self.goal_dist_curriculum_step = 0.075 # 증가량 (m)
        self.goal_dist_max_limit = 8.0         # 최대 상한 (m)
        # 누적 에피소드 성공률 기반 커리큘럼
        # 1000 에피소드 누적 성공률이 70% 이상이면 거리 증가
        self.goal_reached_this_episode = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self.total_episodes_count = 0
        self.total_success_count = 0
        self.curriculum_check_interval = 1000  # 1000 에피소드마다 판단
        self.curriculum_threshold = 0.70       # 성공률 기준

        # ── [스타일 보상] mocap reference motion 로드 (클립 단위, 2026.07.20 수정) ──
        # 기존 cmu_walk_h1.npy는 서로 다른 사람의 걷기 클립을 그냥 이어붙인 형태라
        # 클립 경계에서 관절 각도가 순간이동하듯 튀는 문제가 있었음 (46곳 발견).
        # cmu_walk_h1_full19_clips.npz: clips(num_clips, max_len, 19) + lengths(num_clips,)
        # 2026.07.21: 하체10 + 상체9(토르소, 양쪽 어깨3축, 팔꿈치) = 19 DOF로 확장
        # 에피소드마다 클립 하나를 랜덤하게 골라, 그 클립 안에서만 위상이 순환하도록 함
        # (클립 간 순간이동 없음)
        _motion_path = os.path.join(
            os.path.dirname(__file__),          # envs/locomotion/
            "..", "..", "data", "motions", "cmu_walk_h1_full19_clips.npz"
        )
        _motion_path = os.path.normpath(_motion_path)
        if os.path.exists(_motion_path):
            _npz = np.load(_motion_path)
            self.ref_motion_clips = torch.tensor(
                _npz["clips"], dtype=torch.float32, device=self.device
            )                                                       # (num_clips, max_len, 10)
            self.ref_motion_clip_lens = torch.tensor(
                _npz["lengths"], dtype=torch.long, device=self.device
            )                                                       # (num_clips,)
            self.num_ref_clips = self.ref_motion_clips.shape[0]
            # 각 env마다: 현재 배정된 클립 idx + 그 클립 안에서의 위상(phase) idx
            self.motion_clip_idx = torch.zeros(
                self.num_envs, dtype=torch.long, device=self.device
            )
            self.motion_phase_idx = torch.zeros(
                self.num_envs, dtype=torch.long, device=self.device
            )
            self._has_ref_motion = True
        else:
            self._has_ref_motion = False
            print(f"[경고] reference motion 파일을 찾을 수 없습니다: {_motion_path}")

    def _setup_simulator_control(self):
        self.simulator.commands = self.commands

    def _update_tasks_callback(self):
        """E2E: velocity command 설정 없이 goal 도달 여부만 체크"""
        super()._update_tasks_callback()

        # ── 목표 도달 체크 및 리샘플 ──
        robot_xy = self.simulator.robot_root_states[:, :2]
        to_target = self.target_pos - robot_xy
        dist = torch.norm(to_target, dim=1)

        pos_reached = dist < self.goal_reach_threshold
        stable_goal_mask = self._compute_stable_goal_mask(pos_reached)
        unstable_goal_mask = pos_reached & ~stable_goal_mask
        reached = stable_goal_mask.nonzero(as_tuple=False).flatten()

        if unstable_goal_mask.any():
            self.log_dict["unstable_goal_count"] = unstable_goal_mask.float().sum()

        if len(reached) > 0:
            self.goal_reached_this_episode[reached] = True
            # Goal bonus는 progress reward와 분리한다.
            # reward 계산이 callback 전/후 어느 쪽에서 호출되어도 다음 reward pass에서 1회만 지급된다.
            self.goal_reached_reward_buf[reached] = 1.0
            logger.info(
                f"[GOAL REACHED] env {reached.tolist()} | "
                f"dist: {dist[reached].tolist()} | "
                f"max_goal_dist: {self.max_goal_dist:.2f}m"
            )
            self._resample_target(reached)
            if self.num_envs == 1:
                self._resample_obstacles(reached)

            # target을 즉시 바꾸면 signed progress가 큰 음수로 튀는 문제가 생긴다.
            # 새 target 기준 거리로 prev buffer를 재초기화해서 리샘플 직후 보상 spike를 막는다.
            reached_robot_xy = self.simulator.robot_root_states[reached, :2]
            self.prev_dist_to_target[reached] = torch.norm(
                self.target_pos[reached] - reached_robot_xy, dim=1
            )

        # 텐서보드 로깅
        self.log_dict["max_goal_dist"] = torch.tensor(self.max_goal_dist, dtype=torch.float)
        self.log_dict["goal_reach_threshold"] = torch.tensor(self.goal_reach_threshold, dtype=torch.float)
        if self.total_episodes_count > 0:
            self.log_dict["goal_reach_rate"] = torch.tensor(
                self.total_success_count / max(self.total_episodes_count, 1), dtype=torch.float
            )

        # ── Genesis 시각화 마커 업데이트 (eval 1-env 전용) ──
        if self.num_envs == 1:
            tx = self.target_pos[0, 0].item()
            ty = self.target_pos[0, 1].item()
            self.simulator.target_pos_for_cam = (tx, ty)
            self.simulator.target_marker.set_pos([[tx, ty, 0.3]])

    def _compute_stable_goal_mask(self, pos_reached):
        """목표 반경 안에서도 안정적인 자세일 때만 goal success로 인정한다."""
        base_height = self.simulator.robot_root_states[:, 2]
        desired_base_height = float(getattr(self.config.rewards, "desired_base_height", 0.98))
        min_height = desired_base_height * self.goal_stable_min_height_ratio
        height_ok = base_height > min_height

        base_gravity = quat_rotate_inverse(self.base_quat, self.gravity_vec)
        orientation_error = torch.sum(torch.square(base_gravity[:, :2]), dim=1)
        ori_ok = orientation_error < self.goal_stable_max_orientation_error

        ang_vel_xy = torch.norm(self.base_ang_vel[:, :2], dim=1)
        ang_vel_ok = ang_vel_xy < self.goal_stable_max_ang_vel_xy

        lin_vel_z_ok = torch.abs(self.base_lin_vel[:, 2]) < self.goal_stable_max_lin_vel_z

        return pos_reached & height_ok & ori_ok & ang_vel_ok & lin_vel_z_ok

    def _resample_target(self, env_ids):
        """현재 로봇 위치 기준으로 goal을 다시 뽑는다.

        기존 코드는 world origin 기준으로 target을 뽑고 있었기 때문에, 로봇이 origin에서
        멀어진 뒤에는 curriculum의 goal distance가 실제 로봇-target 거리와 달라질 수 있었다.
        PPO-only navigation에서는 이 차이가 reward/observation을 흔들기 쉬우므로 로봇 기준
        상대 좌표로 target을 생성한다.
        """
        min_dist = 0.8
        # target은 항상 현재 로봇 위치 기준 self.min_goal_dist ~ self.max_goal_dist에서 샘플링한다.
        # curriculum이 올라가도 가까운 목표와 중간 목표를 계속 섞어 catastrophic forgetting을 줄인다.
        robot_xy = self.simulator.robot_root_states[env_ids, :2]

        for _ in range(20):
            # Goal Distance Curriculum: 실제 로봇 위치 기준 거리
            rand_dist = torch_rand_float(
                self.min_goal_dist, self.max_goal_dist,
                (len(env_ids), 1), device=self.device
            ).squeeze(1)
            rand_angle = torch_rand_float(-3.14159, 3.14159, (len(env_ids), 1), device=self.device).squeeze(1)

            tx = robot_xy[:, 0] + rand_dist * torch.cos(rand_angle)
            ty = robot_xy[:, 1] + rand_dist * torch.sin(rand_angle)

            obs_ok = torch.ones(len(env_ids), dtype=torch.bool, device=self.device)
            for i in range(3):
                d = torch.norm(torch.stack([tx - self.obstacle_pos[env_ids, i, 0],
                                            ty - self.obstacle_pos[env_ids, i, 1]], dim=1), dim=1)
                obs_ok &= (d >= min_dist)

            if obs_ok.all():
                break

        self.target_pos[env_ids, 0] = tx
        self.target_pos[env_ids, 1] = ty

    def _resample_commands(self, env_ids):
        # E2E에서는 사용하지 않지만 부모 클래스 호환을 위해 유지
        self.commands[env_ids, 0] = torch_rand_float(
            self.command_ranges["lin_vel_x"][0],
            self.command_ranges["lin_vel_x"][1],
            (len(env_ids), 1), device=str(self.device)
        ).squeeze(1)
        self.commands[env_ids, 1] = torch_rand_float(
            self.command_ranges["lin_vel_y"][0],
            self.command_ranges["lin_vel_y"][1],
            (len(env_ids), 1), device=str(self.device)
        ).squeeze(1)

    def _reset_tasks_callback(self, env_ids):
        # ── 누적 성공률 계산 (커리큘럼) ──
        n_done = len(env_ids)
        n_success = self.goal_reached_this_episode[env_ids].sum().item()
        self.total_episodes_count += n_done
        self.total_success_count += n_success

        # 플래그 리셋
        self.goal_reached_this_episode[env_ids] = False
        self.goal_reached_reward_buf[env_ids] = 0.0

        # 1000 에피소드마다 판단
        if self.total_episodes_count >= self.curriculum_check_interval:
            avg_success = self.total_success_count / self.total_episodes_count
            if avg_success >= self.curriculum_threshold:
                self.max_goal_dist = min(
                    self.max_goal_dist + self.goal_dist_curriculum_step,
                    self.goal_dist_max_limit
                )
                self.min_goal_dist = 1.5
                logger.info(
                    f"[CURRICULUM UP] max_goal_dist: {self.max_goal_dist:.2f}m | "
                    f"avg_success: {avg_success:.2%} | threshold: {self.curriculum_threshold:.2%}"
                )
            else:
                logger.info(
                    f"[CURRICULUM HOLD] max_goal_dist: {self.max_goal_dist:.2f}m | "
                    f"avg_success: {avg_success:.2%} | threshold: {self.curriculum_threshold:.2%}"
                )
            self.total_episodes_count = 0
            self.total_success_count = 0

        super()._reset_tasks_callback(env_ids)
        self._resample_target(env_ids)
        self._resample_obstacles(env_ids)
        # 리셋 시 이전 거리 버퍼 초기화
        robot_xy = self.simulator.robot_root_states[env_ids, :2]
        self.prev_dist_to_target[env_ids] = torch.norm(
            self.target_pos[env_ids] - robot_xy, dim=1
        )

        # ── [스타일 보상] 리셋 시 클립을 랜덤하게 배정하고, 그 클립 안에서 랜덤 위상으로 초기화 ──
        # (master 브랜치 병합 시 누락됐던 부분 + 클립 단위 구조로 재작성, 2026.07.20)
        if self._has_ref_motion:
            n = len(env_ids)
            new_clip_idx = torch.randint(0, self.num_ref_clips, (n,), device=self.device)
            self.motion_clip_idx[env_ids] = new_clip_idx
            clip_lens = self.ref_motion_clip_lens[new_clip_idx]
            # 클립별로 길이가 달라서 각 env마다 다른 상한으로 랜덤 정수를 뽑아야 함
            rand_frac = torch.rand(n, device=self.device)
            self.motion_phase_idx[env_ids] = (rand_frac * clip_lens.float()).long()

    def _resample_obstacles(self, env_ids):
        robot_pos = self.simulator.robot_root_states[env_ids, :2]
        min_dist = 0.8

        for i in range(3):
            for _ in range(20):
                rand_dist = torch_rand_float(1.5, 5.0, (len(env_ids), 1), device=self.device).squeeze(1)
                rand_angle = torch_rand_float(-3.14159, 3.14159, (len(env_ids), 1), device=self.device).squeeze(1)

                # 장애물도 world origin이 아니라 현재 로봇 위치 기준으로 배치한다.
                ox = robot_pos[:, 0] + rand_dist * torch.cos(rand_angle)
                oy = robot_pos[:, 1] + rand_dist * torch.sin(rand_angle)

                dist_robot = torch.norm(
                    torch.stack([ox - robot_pos[:, 0], oy - robot_pos[:, 1]], dim=1), dim=1)
                dist_target = torch.norm(
                    torch.stack([ox - self.target_pos[env_ids, 0], oy - self.target_pos[env_ids, 1]], dim=1), dim=1)

                dist_obs_ok = torch.ones(len(env_ids), dtype=torch.bool, device=self.device)
                for j in range(i):
                    d = torch.norm(
                        torch.stack([ox - self.obstacle_pos[env_ids, j, 0],
                                     oy - self.obstacle_pos[env_ids, j, 1]], dim=1), dim=1)
                    dist_obs_ok &= (d >= min_dist)

                if (dist_robot >= 0.5).all() and (dist_target >= min_dist).all() and dist_obs_ok.all():
                    break

            self.obstacle_pos[env_ids, i, 0] = ox
            self.obstacle_pos[env_ids, i, 1] = oy

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

    ########################### TRACKING REWARDS ###########################
    # E2E: tracking_lin_vel / tracking_ang_vel은 yaml에서 scale=0이므로 호출 안 됨.
    # 혹시 future use를 위해 함수는 유지.

    def _reward_tracking_lin_vel(self):
        lin_vel_error = torch.sum(
            torch.square(self.commands[:, :2] - self.base_lin_vel[:, :2]), dim=1)
        return torch.exp(-lin_vel_error / self.config.rewards.reward_tracking_sigma.lin_vel)

    def _reward_tracking_ang_vel(self):
        ang_vel_error = torch.square(self.commands[:, 2] - self.base_ang_vel[:, 2])
        return torch.exp(-ang_vel_error / self.config.rewards.reward_tracking_sigma.ang_vel)

    def _reward_goal_reached(self):
        """Signed Progress Reward.

        이전 스텝보다 목표에 가까워지면 양수, 멀어지면 음수다.
        E2E PPO에서는 clamp(min=0)를 제거해야 목표에서 멀어지는 행동을 빨리 버린다.
        YAML 호환을 위해 함수명은 goal_reached로 유지한다.
        """
        robot_xy = self.simulator.robot_root_states[:, :2]
        dist = torch.norm(self.target_pos - robot_xy, dim=1)
        progress = self.prev_dist_to_target - dist
        self.prev_dist_to_target = dist.clone()

        # 빠른 질주 자체가 추가 보상으로 커지는 것을 막는다.
        # 전역 speed penalty가 아니라 progress reward 상한만 제한하므로 균형 회복 행동은 덜 방해한다.
        max_progress_speed = 1.0
        max_progress_per_step = max_progress_speed * self.dt
        return torch.clamp(progress, min=-0.25, max=max_progress_per_step)

    def _reward_goal_bonus(self):
        """목표 도달 순간 1회 지급되는 sparse bonus.

        target은 callback에서 즉시 리샘플되므로, distance check만으로는 reward 계산 순서에 따라
        보너스가 누락될 수 있다. 따라서 _update_tasks_callback에서 채운 1-step buffer를 사용한다.
        """
        bonus = self.goal_reached_reward_buf.clone()
        self.goal_reached_reward_buf[:] = 0.0
        return bonus

    def _reward_lin_vel_toward_goal(self):
        """실제로 목표 방향으로 이동하는 속도에 비례한 보상.

        목표 방향이면 양수, 반대 방향이면 음수다. 단순 속력(norm(v))이 아니라
        goal direction과 base linear velocity의 내적을 사용한다.
        """
        goal_dir = self._get_obs_goal_direction()       # 로컬 프레임 단위벡터 (N, 2)
        forward_vel = self.base_lin_vel[:, 0]            # 로봇 전방 실제 속도
        lateral_vel = self.base_lin_vel[:, 1]            # 로봇 측면 실제 속도
        vel_toward_goal = forward_vel * goal_dir[:, 0] + lateral_vel * goal_dir[:, 1]
        return torch.clamp(vel_toward_goal, min=-1.0, max=1.5)

    def _reward_goal_heading(self):
        """몸의 전방이 목표를 향할수록 보상.

        _get_obs_goal_direction의 local_x는 목표가 정면이면 1, 뒤쪽이면 -1이다.
        작은 weight로 사용하면 불필요하게 크게 도는 행동을 줄이고 제자리 회전을 유도한다.
        """
        goal_dir = self._get_obs_goal_direction()
        return goal_dir[:, 0]

    # ── [스타일 보상] (master 브랜치에서 병합) ──────────────────────────────
    def _reward_style_imitation(self):
        """
        CMU mocap 걷기 데이터를 reference motion으로 사용하는 스타일 보상.

        로봇 관절 각도와 reference motion의 동일 위상 관절 각도를 비교하여,
        차이가 작을수록 높은 보상을 준다. ref_motion_clips의 마지막 차원 크기에 맞춰
        하체 10 DOF 또는 전신 19 DOF(하체+상체)를 자동으로 비교한다.

        수식: r = exp(-w * ||q_current - q_ref||^2)
          - w: 민감도 계수 (config의 style_imitation_sigma로 조절)
          - 차이가 0이면 r=1.0 (최대), 차이가 클수록 r→0

        매 스텝마다 motion_phase_idx를 1씩 전진시켜, 각 env에 배정된 클립 "안에서만" 순환함
        (클립 경계를 넘어가지 않으므로 서로 다른 사람 클립 간 순간이동 문제가 없음).
        리셋 시에는 _reset_tasks_callback에서 클립을 새로 뽑고 랜덤 위상으로 초기화.
        """
        if not self._has_ref_motion:
            return torch.zeros(self.num_envs, device=self.device)

        # 현재 env마다 배정된 클립에서, 그 위상에 해당하는 reference 관절 각도 가져오기
        # ref_motion_clips: (num_clips, max_len, ref_dof) - ref_dof는 10(하체) 또는 19(전신)
        ref_pose = self.ref_motion_clips[self.motion_clip_idx, self.motion_phase_idx]  # (num_envs, ref_dof)

        # 현재 로봇 관절 각도를 ref_pose와 동일한 DOF 수만큼만 비교
        # dof_pos shape: (num_envs, 총 DOF 수)
        ref_dof = ref_pose.shape[-1]
        current_pose = self.simulator.dof_pos[:, :ref_dof]       # (num_envs, ref_dof)

        # 관절 각도 오차의 제곱합
        pose_diff_sq = torch.sum(torch.square(current_pose - ref_pose), dim=1)  # (num_envs,)

        # sigma 값으로 민감도 조절 (클수록 관대함)
        sigma = getattr(self.config.rewards, "style_imitation_sigma", 0.5)
        reward = torch.exp(-pose_diff_sq / sigma)

        # 위상 인덱스 1 전진 (각 env가 배정된 클립의 길이 안에서만 순환)
        current_clip_lens = self.ref_motion_clip_lens[self.motion_clip_idx]
        self.motion_phase_idx = (self.motion_phase_idx + 1) % current_clip_lens

        return reward

    ########################### PENALTY REWARDS ###########################

    def _reward_penalty_lin_vel_z(self):
        return torch.square(self.base_lin_vel[:, 2])

    def _reward_penalty_ang_vel_xy(self):
        return torch.sum(torch.square(self.base_ang_vel[:, :2]), dim=1)

    def _reward_penalty_ang_vel_xy_torso(self):
        torso_ang_vel = quat_rotate_inverse(
            self.simulator._rigid_body_rot[:, self.torso_index],
            self.simulator._rigid_body_ang_vel[:, self.torso_index]
        )
        return torch.sum(torch.square(torso_ang_vel[:, :2]), dim=1)

    def _reward_penalty_unstable_goal(self):
        """goal 반경 안에 들어왔지만 안정 조건을 통과하지 못하면 벌점."""
        robot_xy = self.simulator.robot_root_states[:, :2]
        dist = torch.norm(self.target_pos - robot_xy, dim=1)
        pos_reached = dist < self.goal_reach_threshold
        stable_goal_mask = self._compute_stable_goal_mask(pos_reached)
        return (pos_reached & ~stable_goal_mask).float()

    def _reward_penalty_speed_limit(self):
        """현재는 scale 0으로 둔다. 필요 시 속도 상한 실험용으로만 사용."""
        speed_xy = torch.norm(self.base_lin_vel[:, :2], dim=1)
        speed_limit = 1.5
        excess_speed = torch.clamp(speed_xy - speed_limit, min=0.0)
        return excess_speed ** 2

    def _reward_termination(self):
        """timeout이 아닌 reset/fall에 대한 termination penalty."""
        if not hasattr(self, "reset_buf"):
            return torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)

        reset = self.reset_buf.bool()
        if hasattr(self, "time_out_buf"):
            timeout = self.time_out_buf.bool()
            return (reset & ~timeout).float()
        return reset.float()

    def _reward_penalty_feet_contact_forces(self):
        return torch.sum(
            (torch.norm(self.simulator.contact_forces[:, self.feet_indices, :], dim=-1)
             - self.config.rewards.locomotion_max_contact_force).clip(min=0.), dim=1)

    ########################### FEET REWARDS ###########################

    def _reward_feet_air_time(self):
        """E2E: commands 대신 실제 선속도로 필터"""
        contact = self.simulator.contact_forces[:, self.feet_indices, 2] > 1.
        contact_filt = torch.logical_or(contact, self.last_contacts)
        self.last_contacts = contact
        first_contact = (self.feet_air_time > 0.) * contact_filt
        self.feet_air_time += self.dt
        rew_airTime = torch.sum((self.feet_air_time - 0.5) * first_contact, dim=1)
        # velocity command 대신 실제 선속도 기준으로 필터 (정지 상태 보상 방지)
        rew_airTime *= torch.norm(self.base_lin_vel[:, :2], dim=1) > 0.1
        self.feet_air_time *= ~contact_filt
        return rew_airTime

    def _reward_penalty_in_the_air(self):
        contact = self.simulator.contact_forces[:, self.feet_indices, 2] > 1.
        contact_filt = torch.logical_or(contact, self.last_contacts)
        first_foot_contact = contact_filt[:, 0]
        second_foot_contact = contact_filt[:, 1]
        reward = ~(first_foot_contact | second_foot_contact)
        return reward

    def _reward_penalty_stumble(self):
        return torch.any(
            torch.norm(self.simulator.contact_forces[:, self.feet_indices, :2], dim=2)
            > 5 * torch.abs(self.simulator.contact_forces[:, self.feet_indices, 2]), dim=1)

    def _reward_penalty_feet_ori(self):
        left_quat = self.simulator._rigid_body_rot[:, self.feet_indices[0]]
        left_gravity = quat_rotate_inverse(left_quat, self.gravity_vec)
        right_quat = self.simulator._rigid_body_rot[:, self.feet_indices[1]]
        right_gravity = quat_rotate_inverse(right_quat, self.gravity_vec)
        return (torch.sum(torch.square(left_gravity[:, :2]), dim=1) ** 0.5
                + torch.sum(torch.square(right_gravity[:, :2]), dim=1) ** 0.5)

    def _reward_base_height(self):
        base_height = self.simulator.robot_root_states[:, 2]
        return torch.square(base_height - self.config.rewards.desired_base_height)

    def _reward_penalty_hip_pos(self):
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
        return (torch.sum(torch.square(left_gravity[:, :2]), dim=1) ** 0.5
                + torch.sum(torch.square(right_gravity[:, :2]), dim=1) ** 0.5)

    def _reward_penalty_feet_slippage(self):
        foot_vel = self.simulator._rigid_body_vel[:, self.feet_indices]
        return torch.sum(
            torch.norm(foot_vel, dim=-1)
            * (torch.norm(self.simulator.contact_forces[:, self.feet_indices, :], dim=-1) > 1.), dim=1)

    def _reward_penalty_feet_height(self):
        feet_height = self.simulator._rigid_body_pos[:, self.feet_indices, 2]
        dif = torch.abs(feet_height - self.config.rewards.feet_height_target)
        dif = torch.min(dif, dim=1).values
        return torch.clip(dif - 0.02, min=0.)

    def _reward_penalty_close_feet_xy(self):
        left_foot_xy = self.simulator._rigid_body_pos[:, self.feet_indices[0], :2]
        right_foot_xy = self.simulator._rigid_body_pos[:, self.feet_indices[1], :2]
        feet_distance_xy = torch.norm(left_foot_xy - right_foot_xy, dim=1)
        return (feet_distance_xy < self.config.rewards.close_feet_threshold) * 1.0

    def _reward_penalty_close_knees_xy(self):
        left_knee_xy = self.simulator._rigid_body_pos[:, self.knee_indices[0], :2]
        right_knee_xy = self.simulator._rigid_body_pos[:, self.knee_indices[1], :2]
        self.knee_distance_xy = torch.norm(left_knee_xy - right_knee_xy, dim=1)
        return (self.knee_distance_xy < self.config.rewards.close_knees_threshold) * 1.0

    def _reward_upperbody_joint_angle_freeze(self):
        assert self.config.robot.has_upper_body_dof
        deviation = torch.abs(
            self.simulator.dof_pos[:, self.upper_dof_indices]
            - self.default_dof_pos[:, self.upper_dof_indices])
        return torch.sum(deviation, dim=1)

    ######################### Observations #########################

    # ── E2E: goal_direction / goal_distance로 교체 ──
    # command_lin_vel / command_ang_vel getter는 삭제

    def _get_obs_goal_direction(self):
        """
        로봇 로컬 프레임 기준 목표 방향 단위벡터 (2D)
        - local_x > 0: 목표가 앞쪽  / < 0: 뒤쪽
        - local_y > 0: 목표가 왼쪽  / < 0: 오른쪽
        """
        robot_xy = self.simulator.robot_root_states[:, :2]
        to_target = self.target_pos - robot_xy                          # (N, 2) 월드 벡터
        dist = torch.norm(to_target, dim=1, keepdim=True).clamp(min=0.01)
        goal_dir_world = to_target / dist                               # 월드 단위벡터

        # 로봇 heading으로 로컬 프레임 변환
        forward = quat_apply(self.base_quat, self.forward_vec)
        cos_h = forward[:, 0:1]
        sin_h = forward[:, 1:2]
        local_x =  goal_dir_world[:, 0:1] * cos_h + goal_dir_world[:, 1:2] * sin_h
        local_y = -goal_dir_world[:, 0:1] * sin_h + goal_dir_world[:, 1:2] * cos_h
        return torch.cat([local_x, local_y], dim=1)                     # (N, 2)

    def _get_obs_goal_distance(self):
        """
        목표까지 거리, lidar_max_range(5m) 기준으로 0~1 정규화 (1D)
        """
        robot_xy = self.simulator.robot_root_states[:, :2]
        dist = torch.norm(self.target_pos - robot_xy, dim=1, keepdim=True)
        return torch.clamp(dist / self.lidar_max_range, 0.0, 1.0)      # (N, 1)

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
            perp_sq = (dx.unsqueeze(1) ** 2 + dy.unsqueeze(1) ** 2) - proj ** 2
            valid = (proj > 0) & (perp_sq < r ** 2)
            hit_dist = proj - torch.sqrt(torch.clamp(r ** 2 - perp_sq, min=0.0))
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

    def _reward_penalty_obstacle_proximity(self):
        """충돌 전부터 장애물에 가까워지는 행동을 약하게 벌점화한다.

        collision penalty만 있으면 PPO는 실제로 부딪힐 때까지 회피 신호를 거의 받지 못한다.
        이 함수는 obstacle_radius+1.0m 안쪽으로 들어오면 0~1 사이의 soft penalty를 준다.
        YAML에서 음수 scale을 붙여 사용한다.
        """
        robot_xy = self.simulator.robot_root_states[:, :2]
        min_dist = torch.full((self.num_envs,), self.lidar_max_range, device=self.device)
        for i in range(3):
            d = torch.norm(robot_xy - self.obstacle_pos[:, i], dim=1)
            min_dist = torch.minimum(min_dist, d)

        collision_margin = self.obstacle_radius + 0.3
        safe_margin = self.obstacle_radius + 1.0
        proximity = (safe_margin - min_dist) / (safe_margin - collision_margin)
        return torch.clamp(proximity, min=0.0, max=1.0) ** 2
