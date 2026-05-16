# HumanoidVerse - Stage3: 장애물 회피

Unitree H1 10DoF 휴머노이드 로봇의 목표 지점 이동 및 장애물 회피 강화학습 프로젝트입니다.

---

## 학습 단계

| 단계 | 내용 | 상태 |
|------|------|------|
| Stage1 | 기본 보행 학습 | ✅ 완료 |
| Stage2 | 목표 지점 이동 (속도 커맨드 방식) | ✅ 완료 |
| Stage3 | 장애물 회피 + 목표 지점 이동 | ✅ 완료 |

---

## 환경

```
OS:       Windows 11 + WSL2 Ubuntu
GPU:      NVIDIA RTX 4060 Ti (8GB)
Python:   3.10 (conda 환경: hgen)
알고리즘:  PPO (rsl-rl 기반)
시뮬레이터: Genesis
```

---

## 주요 구현

### Observation (78차원)
```
base_lin_vel(3) + base_ang_vel(3) + projected_gravity(3)
+ command_lin_vel(2) + command_ang_vel(1)
+ dof_pos(10) + dof_vel(10) + actions(10)
+ lidar_scan(36)
```

### LiDAR
- 수학적 레이캐스팅 (PyTorch 텐서 연산)
- 36방향 장애물 거리 계산
- 원통형 장애물 3개와 레이-원통 교차 검출

### 보상 함수
```yaml
tracking_lin_vel:          1.0   # 전진 속도 추종
tracking_ang_vel:          0.5   # 회전 속도 추종
goal_reached:              5.0   # 목표 도달 (Progress Reward)
penalty_obstacle_collision: -10.0 # 장애물 접근 패널티
```

### 장애물/목표 배치
```
목표:   원점(0,0) 기준 3~8m 랜덤
장애물: 원점(0,0) 기준 1.5~5m 랜덤
```

---

## 수정 파일

| 파일 | 주요 변경사항 |
|------|-------------|
| `humanoidverse/envs/locomotion/locomotion.py` | LiDAR, 목표 추종, 장애물 회피 보상 |
| `humanoidverse/simulator/genesis/genesis.py` | 장애물 추가, 카메라 녹화 |
| `humanoidverse/config/rewards/loco/reward_h1_locomotion.yaml` | 보상 스케일 |
| `humanoidverse/config/obs/loco/leggedloco_obs_singlestep_withlinvel.yaml` | lidar_scan 추가 |

---

## 학습 명령어

```bash
conda activate hgen
cd /mnt/c/Users/pc123/HumanoidVerse
export LD_LIBRARY_PATH=/usr/lib/wsl/lib:$LD_LIBRARY_PATH

python humanoidverse/train_agent.py \
  +simulator=genesis \
  +exp=locomotion \
  +domain_rand=NO_domain_rand \
  +rewards=loco/reward_h1_locomotion \
  +robot=h1/h1_10dof \
  +terrain=terrain_locomotion_plane \
  +obs=loco/leggedloco_obs_singlestep_withlinvel \
  num_envs=512 \
  project_name=H1_Stage3 \
  experiment_name=H1_10dof_obstacle_v7 \
  headless=True \
  rewards.reward_penalty_curriculum=True \
  rewards.reward_initial_penalty_scale=0.5
```

---

## Eval 명령어

```bash
python humanoidverse/eval_agent.py \
  "+checkpoint=logs/H1_Stage3/폴더명/model_XXXXX.pt" \
  +headless=True \
  +num_envs=1
```
