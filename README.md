# HumanoidVerse - Stage3: WebSocket Unity 연동

Stage3 장애물 회피 학습 결과를 Unity와 실시간으로 연동하는 WebSocket 서버가 추가된 브랜치입니다.

---

## 학습 단계

| 단계 | 내용 | 상태 |
|------|------|------|
| Stage1 | 기본 보행 학습 | ✅ 완료 |
| Stage2 | 목표 지점 이동 | ✅ 완료 |
| Stage3 | 장애물 회피 + 목표 지점 이동 | ✅ 완료 |
| Unity 연동 | WebSocket 실시간 시각화 | ✅ 완료 |

---

## 환경

```
OS:       Windows 11 + WSL2 Ubuntu
GPU:      NVIDIA RTX 4060 Ti (8GB)
Python:   3.10 (conda 환경: hgen)
알고리즘:  PPO (rsl-rl 기반)
시뮬레이터: Genesis
통신:      WebSocket (localhost:8765)
```

---

## 시스템 구조

```
Python (Genesis 물리 시뮬레이션)
  ├── RL Policy 실행
  ├── 물리 계산 (관절각도, 위치, 방향)
  └── WebSocket 서버 (localhost:8765)
          ↕
Unity (시각화)
  ├── 캐릭터 애니메이션
  ├── 목표 지점 클릭 입력
  └── WebSocket 클라이언트
```

---

## 전송 데이터 (50Hz)

```
Python → Unity:
  joint_angles (10개): 관절 각도 (라디안)
  robot_pos    (3개):  로봇 XYZ 위치
  robot_quat   (4개):  로봇 방향 쿼터니언
  target_pos   (2개):  목표 XY 위치
  obstacle_pos (6개):  장애물 3개 XY 위치

Unity → Python:
  type: "set_target"
  x, y: 목표 좌표
```

---

## 추가된 파일

| 파일 | 설명 |
|------|------|
| `humanoidverse/websocket_server.py` | WebSocket 서버 |
| `run_backend.bat` | Windows 실행 스크립트 |
| `run_backend.sh` | WSL Ubuntu 실행 스크립트 |

---

## 실행 방법

### 1. 패키지 설치

```bash
pip install websockets --break-system-packages
```

### 2. Python 백엔드 실행

```bash
conda activate hgen
cd /mnt/c/Users/pc123/HumanoidVerse
export LD_LIBRARY_PATH=/usr/lib/wsl/lib:$LD_LIBRARY_PATH

python humanoidverse/eval_agent.py \
  "+checkpoint=logs/H1_Stage3/폴더명/model_XXXXX.pt" \
  +headless=True \
  +num_envs=1
```

또는 `run_backend.bat` 더블클릭

### 3. Unity 실행

[HumanoidVerse-Unity](https://github.com/dulgiDove/HumanoidVerse-Unity) 레포 참고

---

## Stage3 학습 내용

`stage3/obstacle-avoidance` 브랜치 참고
