H1 10DoF E2E Goal-Reaching Locomotion

HumanoidVerse 기반 H1 10DoF 휴머노이드의 PPO-only end-to-end goal reaching locomotion 실험입니다.

기존 velocity command tracking 방식에서 벗어나, 로봇이 goal direction, goal distance, LiDAR scan을 관측하고 직접 목표 지점까지 이동하며 장애물을 회피하도록 학습합니다.

Main Changes
1. Velocity Command Tracking 제거

기존에는 lin_vel_x, lin_vel_y, ang_vel_yaw command를 추종하는 구조였지만, 현재는 외부 속도 command를 사용하지 않습니다.

대신 다음 관측값을 기반으로 목표까지 직접 이동합니다.

goal direction
goal distance
LiDAR scan
robot state observations
2. Goal-Based Reward 구조 적용

기존 속도 추종 중심 reward에서 목표 도달 중심 reward로 변경했습니다.

주요 reward scale:

goal_reached: 80.0
goal_bonus: 40.0
goal_heading: 2.0
lin_vel_toward_goal: 0.0
goal_reached: 이전 step보다 목표에 가까워진 정도를 보상하는 progress reward
goal_bonus: 목표에 안정적으로 도달했을 때 1회 지급되는 sparse reward
goal_heading: 몸의 전방이 목표를 향하도록 유도
lin_vel_toward_goal: 기존 목표 방향 속도 보상은 제거
3. Stable Goal Condition 추가

단순히 goal radius 안에 들어오는 것만으로는 성공으로 인정하지 않습니다.

목표 반경 안에 들어온 뒤에도 다음 조건을 만족해야 goal success로 처리합니다.

base height 유지
orientation 안정
angular velocity 안정
vertical velocity 안정

이를 통해 목표 지점에 몸을 던져서 도달하는 reward hacking을 줄였습니다.

4. Goal Distance Curriculum 수정

기존에는 낮은 성공률에서도 max_goal_dist가 증가할 수 있었고, curriculum update 이후 목표가 현재 max distance 근처로만 샘플링되는 문제가 있었습니다.

현재는 다음과 같이 수정했습니다.

self.min_goal_dist = 1.5
self.max_goal_dist = curriculum value
self.curriculum_threshold = 0.70
self.goal_dist_curriculum_step = 0.075

목표는 항상 다음 범위에서 샘플링됩니다.

1.5m ~ max_goal_dist

즉, 쉬운 거리부터 긴 거리까지 섞어서 학습합니다.

5. Progress Reward Cap 추가

빠르게 질주하는 행동이 과도하게 보상되지 않도록 progress reward에 cap을 적용했습니다.

max_progress_speed = 1.0
max_progress_per_step = max_progress_speed * self.dt

이를 통해 목표 방향으로 너무 빠르게 달려가다 넘어지는 행동을 줄이고자 했습니다.

6. Obstacle Avoidance 강화

장애물 충돌 penalty와 proximity penalty를 강화했습니다.

penalty_obstacle_collision: -40.0
penalty_obstacle_proximity: -2.0

장애물과 실제로 충돌했을 때뿐 아니라, 가까워지는 행동도 soft penalty로 억제합니다.

7. Penalty Curriculum 비활성화

기존에는 penalty curriculum이 켜져 있어 학습 초반 penalty가 약하게 적용되었습니다.

현재는 penalty scale이 항상 고정되도록 변경했습니다.

reward_penalty_curriculum: False
8. Termination / Unstable Goal Penalty 추가

조기 reset과 불안정한 goal 접근을 억제하기 위해 다음 penalty를 추가했습니다.

termination: -70.0
penalty_unstable_goal: -10.0
termination: time-out이 아닌 실패 reset 발생 시 penalty
penalty_unstable_goal: goal radius 안에 들어왔지만 stable condition을 만족하지 못한 경우 penalty
