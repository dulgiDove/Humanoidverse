# H1 10DoF E2E Goal-Reaching Locomotion

HumanoidVerse 기반 H1 10DoF 휴머노이드의 PPO-only end-to-end goal reaching locomotion 실험입니다.

기존 velocity command tracking 방식에서 벗어나, 로봇이 `goal direction`, `goal distance`, `LiDAR scan`을 관측하고 직접 목표 지점까지 이동하며 장애물을 회피하도록 학습합니다.

---

## Main Changes

### 1. Velocity Command Tracking 제거

기존에는 `lin_vel_x`, `lin_vel_y`, `ang_vel_yaw` command를 추종하는 구조였지만, 현재는 외부 속도 command를 사용하지 않습니다.

대신 다음 관측값을 기반으로 목표까지 직접 이동합니다.

- `goal_direction`
- `goal_distance`
- `lidar_scan`
- robot state observations

---

### 2. Goal-Based Reward 구조 적용

기존 속도 추종 중심 reward에서 목표 도달 중심 reward로 변경했습니다.

주요 reward scale은 다음과 같습니다.

```yaml
goal_reached: 80.0
goal_bonus: 40.0
goal_heading: 2.0
lin_vel_toward_goal: 0.0
```

각 reward의 의미는 다음과 같습니다.

| Reward | Description |
|---|---|
| `goal_reached` | 이전 step보다 목표에 가까워진 정도를 보상하는 progress reward |
| `goal_bonus` | 목표에 안정적으로 도달했을 때 1회 지급되는 sparse reward |
| `goal_heading` | 로봇의 몸 방향이 목표를 향하도록 유도 |
| `lin_vel_toward_goal` | 기존 목표 방향 속도 보상. 현재는 제거하여 `0.0`으로 설정 |

---

### 3. Stable Goal Condition 추가

단순히 goal radius 안에 들어오는 것만으로는 성공으로 인정하지 않습니다.

목표 반경 안에 들어온 뒤에도 다음 조건을 만족해야 goal success로 처리합니다.

- base height 유지
- orientation 안정
- angular velocity 안정
- vertical velocity 안정

이를 통해 목표 지점에 몸을 던져서 도달하는 reward hacking을 줄였습니다.

---

### 4. Goal Distance Curriculum 수정

기존에는 낮은 성공률에서도 `max_goal_dist`가 증가할 수 있었고, curriculum update 이후 목표가 현재 max distance 근처로만 샘플링되는 문제가 있었습니다.

현재는 다음과 같이 수정했습니다.

```python
self.min_goal_dist = 1.5
self.max_goal_dist = curriculum value
self.curriculum_threshold = 0.70
self.goal_dist_curriculum_step = 0.075
```

목표는 항상 다음 범위에서 샘플링됩니다.

```text
1.5m ~ max_goal_dist
```

즉, 쉬운 거리부터 긴 거리까지 섞어서 학습합니다.

---

### 5. Progress Reward Cap 추가

빠르게 질주하는 행동이 과도하게 보상되지 않도록 progress reward에 cap을 적용했습니다.

```python
max_progress_speed = 1.0
max_progress_per_step = max_progress_speed * self.dt
```

이를 통해 목표 방향으로 너무 빠르게 달려가다 넘어지는 행동을 줄이고자 했습니다.

---

### 6. Obstacle Avoidance 강화

장애물 충돌 penalty와 proximity penalty를 강화했습니다.

```yaml
penalty_obstacle_collision: -40.0
penalty_obstacle_proximity: -2.0
```

장애물과 실제로 충돌했을 때뿐 아니라, 가까워지는 행동도 soft penalty로 억제합니다.

---

### 7. Penalty Curriculum 비활성화

기존에는 penalty curriculum이 켜져 있어 학습 초반 penalty가 약하게 적용되었습니다.

현재는 penalty scale이 항상 고정되도록 변경했습니다.

```yaml
reward_penalty_curriculum: False
```

---

### 8. Termination / Unstable Goal Penalty 추가

조기 reset과 불안정한 goal 접근을 억제하기 위해 다음 penalty를 추가했습니다.

```yaml
termination: -70.0
penalty_unstable_goal: -10.0
```

| Penalty | Description |
|---|---|
| `termination` | time-out이 아닌 실패 reset 발생 시 penalty |
| `penalty_unstable_goal` | goal radius 안에 들어왔지만 stable condition을 만족하지 못한 경우 penalty |

---

## Training

Fresh training example:

```bash
python humanoidverse/train_agent.py \
  +simulator=genesis \
  +exp=locomotion \
  +domain_rand=NO_domain_rand \
  +rewards=loco/reward_h1_locomotion \
  +robot=h1/h1_10dof \
  +terrain=terrain_locomotion_plane \
  +obs=loco/leggedloco_obs_singlestep_withlinvel \
  num_envs=512 \
  project_name=H1_E2E \
  experiment_name=H1_10dof_E2E_v22_verified_fresh \
  headless=True \
  rewards.reward_penalty_curriculum=False
```

---

## Evaluation

```bash
python humanoidverse/eval_agent.py \
  "+checkpoint=logs/H1_E2E/<run_name>/model_xxxxx.pt" \
  +headless=True \
  +num_envs=1
```

Checkpoint는 policy weight를 복원하지만, `locomotion.py` 같은 환경 Python 코드는 현재 프로젝트 파일을 사용합니다.

따라서 정확한 evaluation을 위해서는 학습 당시 사용한 코드와 현재 코드가 일치해야 합니다.

---

## TensorBoard Metrics

중점적으로 확인할 metric은 다음과 같습니다.

| Metric | Meaning |
|---|---|
| `Env/goal_reach_rate` | 목표 도달 성공률 |
| `Env/max_goal_dist` | 현재 curriculum의 최대 목표 거리 |
| `Train/mean_episode_length` | 평균 episode 길이 |
| `Train/mean_reward` | 평균 reward |
| `Episode/rew_goal_bonus` | 실제 stable goal 성공 보상 |
| `Episode/rew_goal_reached` | 목표 접근 progress reward |
| `Episode/rew_penalty_unstable_goal` | 불안정한 goal 접근 penalty |
| `Episode/rew_termination` | 조기 reset penalty |
| `Episode/rew_penalty_obstacle_collision` | 장애물 충돌 penalty |

좋은 학습 흐름:

```text
goal_reach_rate slowly increases
max_goal_dist increases step by step
mean_episode_length stays stable or increases
goal_bonus gradually increases
termination penalty does not worsen
```

나쁜 학습 흐름:

```text
max_goal_dist increases
goal_reach_rate collapses
mean_episode_length drops
mean_reward drops
goal_bonus stays near zero
termination or unstable_goal penalty increases
```

---

## Summary of Major Differences

| Component | Previous | Current |
|---|---:|---:|
| Goal curriculum threshold | `0.40` | `0.70` |
| Curriculum step | `0.05` | `0.075` |
| Target sampling | near max distance | `1.5m ~ max_goal_dist` |
| Goal success | distance only | stable goal condition |
| Progress reward | strong, uncapped | capped |
| `goal_reached` scale | `120.0` | `80.0` |
| `goal_bonus` scale | `20.0` | `40.0` |
| `lin_vel_toward_goal` | `1.5` | `0.0` |
| `goal_heading` | `0.5` | `2.0` |
| Penalty curriculum | enabled | disabled |
| Obstacle collision penalty | `-10.0` | `-40.0` |
| Obstacle proximity penalty | `-1.0` | `-2.0` |
| Termination penalty | not explicitly used | `-70.0` |
| Unstable goal penalty | none | `-10.0` |

---

## Notes

This experiment focuses on learning goal-reaching locomotion without explicit velocity commands.

The main objective is not just reaching the goal position, but reaching it in a stable state while avoiding obstacles.
