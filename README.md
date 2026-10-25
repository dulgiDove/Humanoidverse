# Humanoid Navigation Challenge

## 1. 프로젝트 개요

본 프로젝트는 **HumanoidVerse**와 **Genesis Simulator**를 기반으로, H1 휴머노이드 로봇이 목표 지점까지 이동하면서 주변 장애물을 회피하도록 학습시키는 **End-to-End(E2E) 강화학습 프로젝트**입니다.

초기에는 **H1 하체 10DoF 기반 PPO 정책**으로 목표 도달과 장애물 회피 환경을 구축했습니다. 이후 **전신 19DoF**로 확장하고, **LAFAN1 모션 데이터와 Adversarial Motion Priors(AMP)**를 적용해 보행 스타일을 개선했습니다. 장애물 회피 보상과 전신 충돌 판정도 추가로 개선했습니다.

최종 시연은 학습된 **C_fullbody_collision 체크포인트(`model_32000.pt`)**를 이용합니다. **Genesis Viewer**에서 3D 움직임을 확인하고, **로컬 웹 컨트롤 패널**에서 목표·장애물 좌표를 바꾸거나 시뮬레이션을 제어할 수 있습니다.

---

## 2. 주요 기능

- **H1 전신 19DoF** 휴머노이드 시뮬레이션 (초기 개발 모델: 하체 10DoF)
- **E2E 목표 도달**: 목표 방향과 거리, 장애물 정보 등을 관측하여 정책이 관절 행동을 결정
- **PPO + AMP 학습**: 목표 이동·장애물 회피 보상에 모션 스타일 보상을 결합
- **LAFAN1 기반 보행 학습**: BVH → FK → IK 변환을 통해 H1 관절 모션 참조 데이터 구축
- **장애물 3개 회피** 및 충돌 전 근접 페널티를 통한 점진적 회피 유도
- **전신 링크 기반 장애물 거리 판정**: 골반뿐 아니라 팔·다리 등 전신 링크를 고려
- **목표 거리 커리큘럼**: 초기 근거리 목표부터 최대 8m까지 점진적으로 확장
- **Genesis Viewer 기반 3D 시각화**
- **로컬 웹 컨트롤 패널**: Target/Obstacle 좌표 입력·랜덤 생성 및 Start/Pause/Reset/Info/Fit Camera
- 목표 도달 후 새로운 목표 및 장애물 배치 갱신

---

## 3. 학습 시스템 구성

### 3.1 E2E 목표 도달 및 장애물 회피

기존의 속도 명령 추종 방식과 달리, 목표 속도나 조건별 제자리 회전 명령을 별도로 지정하지 않고 **하나의 정책이 관측값에서 직접 관절 행동을 생성**하도록 구성했습니다.

주요 관측값은 다음과 같습니다.

- 로봇의 선속도·각속도 및 중력 방향
- 로봇 기준 목표 방향(`goal_direction`)과 목표 거리(`goal_distance`)
- 관절 위치·관절 속도 및 이전 행동
- **36방향 LiDAR 스캔** 기반 장애물 정보(`lidar_scan`)

주요 보상은 다음과 같습니다.

| 구분 | 보상 항목 | 역할 |
| --- | --- | --- |
| 목표 접근 | `goal_reached` | 이전 스텝보다 목표에 가까워지면 양수, 멀어지면 음수를 주는 signed progress reward |
| 목표 달성 | `goal_bonus` | 안정성 조건을 충족하며 목표에 도달했을 때 추가 보상 |
| 방향 정렬 | `goal_heading` | 로봇이 목표 방향을 향하도록 유도 |
| 장애물 안전 | `penalty_obstacle_collision` | 장애물과의 거리 기반 충돌 판정에 페널티 부여 |
| 선제적 회피 | `penalty_obstacle_proximity` | 장애물에 너무 가까워지기 전부터 연속적인 페널티 부여 |
| 보행 안정성 | 자세·높이·토크·관절 가속도 등 | 안정적인 보행과 불필요한 행동 변화 억제 |

> 최종 설정에서 `lin_vel_toward_goal`의 가중치는 **0**입니다. 목표 접근 보상과 기능이 중복되어 비활성화했습니다.

### 3.2 전신 19DoF 및 AMP 보행 학습

초기 하체 10DoF에서 토르소와 양팔 관절을 추가한 **19DoF H1 모델**로 확장했습니다.

보행 학습은 다음과 같은 과정을 거쳤습니다.

1. **CMU 모션 캡처 기반 Style Imitation**: 관절 자세와 참조 모션 간 오차에 따른 보상을 실험
2. **LAFAN1 모션 데이터 전처리**: BVH 모션을 파싱하고 FK·좌표계 변환·IK를 거쳐 H1 관절 모션으로 변환
3. **AMP 적용**: 실제 참조 모션의 상태 전이와 정책이 생성한 상태 전이를 판별기로 비교하고, 그 결과를 PPO 학습에 추가 보상으로 사용
4. **보행 안정화**: 판별기 replay buffer, R1 gradient penalty, 관절 속도 관측 및 상체 비틀림 페널티 적용

최종 C 체크포인트에서는 **AMP 보상을 사용하며, 직접적인 `style_imitation` 보상 가중치는 0**으로 설정했습니다. 즉 Style Imitation과 AMP를 동시에 활성화한 것이 아니라, 초기 모방 보상 실험을 거쳐 AMP 기반 학습으로 전환한 결과입니다.

---

## 4. 실행 환경

프로젝트에서 사용한 주요 환경은 다음과 같습니다.

- Windows 11 / WSL2 Ubuntu
- NVIDIA GPU 및 CUDA 사용 환경
- Miniconda 또는 Anaconda
- Python 3.10
- PyTorch (CUDA 12.1 설치 스크립트 기준)
- Genesis World 0.2.1
- HumanoidVerse / PPO / AMP

**주의:** Genesis Viewer의 GUI 출력과 GPU 연산을 위해 WSL2에서 NVIDIA 드라이버·CUDA·OpenGL이 정상 동작해야 합니다. 시스템 환경에 따라 추가 설정이 필요할 수 있습니다.

---

## 5. 사전 준비

WSL2 Ubuntu, NVIDIA GPU 드라이버 및 Miniconda/Anaconda를 미리 설치합니다.

GPU 인식 여부:

```bash
nvidia-smi
```

WSL에서 GUI 프로그램 실행 가능 여부:

```bash
xeyes
```

`xeyes`가 없다면 다음과 같이 설치할 수 있습니다.

```bash
sudo apt update
sudo apt install -y x11-apps
```

---

## 6. 설치 방법

현재 최종 **19DoF AMP 코드 및 체크포인트는 `transfer-temp` 브랜치**에 있습니다. GitHub에서 저장소를 받을 때 해당 브랜치를 선택합니다.

```bash
git clone --branch transfer-temp https://github.com/dulgiDove/Humanoidverse.git
cd Humanoidverse
```

이미 저장소를 내려받았다면 최종 브랜치로 전환한 후 프로젝트 루트에서 진행합니다.

```bash
git fetch origin
git switch transfer-temp
```

설치 스크립트 실행 권한을 부여합니다.

```bash
chmod +x scripts/setup_env.sh
```

최초 1회만 설치 스크립트를 실행합니다.

```bash
./scripts/setup_env.sh
```

이 스크립트는 다음 작업을 수행합니다.

- Ubuntu OpenGL / Genesis Viewer 관련 패키지 설치
- Conda 가상환경 생성 및 Python 3.10 설정
- PyTorch CUDA 12.1 설치
- HumanoidVerse 프로젝트 및 Genesis World 0.2.1 설치
- NumPy 1.26.4, `rsl_rl` 및 기타 의존성 설치
- `libigl`, `pyglet`, `PyOpenGL` 버전 고정

기본 Conda 환경 이름은 **`hgen`**입니다. 다른 이름으로 설치할 경우:

```bash
ENV_NAME=hgen_submit_test ./scripts/setup_env.sh
```

---

## 7. 최종 모델 시연 방법

별도의 재학습 없이 저장소에 포함된 최종 체크포인트로 시연할 수 있습니다. **프로젝트 루트 디렉터리**에서 실행합니다.

### 7.1 Conda 환경 활성화

```bash
conda activate hgen
```

다른 이름으로 설치했다면 해당 환경을 활성화합니다.

### 7.2 Genesis Viewer 및 웹 컨트롤 패널 실행

```bash
python humanoidverse/eval_web_control.py \
  +checkpoint=logs/LAFAN1_BVH_IK/20261003_215621-C_fullbody_collision-locomotion-h1/model_32000.pt
```

평가 스크립트는 해당 체크포인트와 함께 저장된 **`config.yaml`**을 읽어 **H1 19DoF 및 AMP 설정**을 불러옵니다. 실행 중 Genesis Viewer가 열리고, 웹 제어 서버가 실행됩니다.

### 7.3 웹 패널 접속

브라우저에서 다음 주소에 접속합니다.

```text
http://localhost:8080
```

---

## 8. 웹 컨트롤 패널 사용법

### Coordinate Setup

- **Target X/Y**: 목표 지점 좌표 입력
- **Obstacle 1/2/3 X/Y**: 장애물 3개의 좌표 입력
- **Random**: 해당 목표 또는 장애물 좌표를 랜덤 생성
- **Random All**: 목표와 모든 장애물 좌표를 랜덤 생성
- **Apply**: 입력한 좌표를 Genesis 시뮬레이션에 적용

### Simulation Control

- **Start**: 학습된 정책 실행 시작
- **Pause**: 시뮬레이션 진행 일시정지
- **Reset**: 로봇과 환경 초기화. 마지막으로 Apply한 좌표 배치가 있으면 해당 배치를 유지
- **Info**: 로봇 위치, 목표 위치, 장애물 좌표, 목표까지의 거리 등 표시
- **Fit Camera**: 로봇·목표·장애물을 화면에 담도록 카메라 위치 조정

웹 패널은 Python 평가 스크립트 내부의 로컬 HTTP 서버와 통신합니다. **실제 물리 시뮬레이션 및 로봇 렌더링은 Genesis Viewer에서 진행**되며, 웹 페이지는 좌표 설정 및 시뮬레이션 제어를 담당합니다.

---

## 9. Random 좌표 생성 범위

Random 버튼은 **현재 로봇 위치로부터의 거리**를 기준으로 좌표를 생성합니다.

- Target Random: 로봇 기준 **1.5m ~ 8.0m**
- Obstacle Random: 로봇 기준 **1.5m ~ 5.0m**
- 목표와 장애물 및 장애물끼리는 최소 **0.8m** 이상 떨어지도록 생성

**주의:** X/Y 좌표값 각각을 위 범위로 제한하는 것이 아니라, 로봇 위치에서의 거리를 기준으로 생성합니다.

---

## 10. 최종 보상 설정으로 새 학습 시작하기

다음은 조원이 공유한 **C안과 동일한 주요 설정으로 체크포인트 없이 새 학습을 시작하는 명령어**입니다. `experiment_name`은 구분 가능한 이름으로 바꿔 사용할 수 있습니다.

```bash
python humanoidverse/train_agent.py \
  +simulator=genesis +exp=locomotion_amp +domain_rand=NO_domain_rand \
  +rewards=loco/reward_h1_locomotion +robot=h1/h1 +terrain=terrain_locomotion_plane \
  +obs=loco/leggedloco_obs_singlestep_withlinvel \
  num_envs=480 headless=True auto_load_latest=False \
  rewards.reward_scales.style_imitation=0 \
  algo.config.amp.gp_type=r1 algo.config.amp.grad_penalty_coef=1.0 algo.config.amp.use_velocity=True \
  checkpoint=null \
  project_name=LAFAN1_BVH_IK experiment_name=C_retrain \
  rewards.obstacle_safe_margin_extra=1.8 \
  rewards.reward_scales.penalty_obstacle_proximity=-4.0
```

주요 설정:

| 항목 | 설정 |
| --- | --- |
| 로봇 | `h1/h1` (19DoF) |
| 알고리즘 | `locomotion_amp` (AMP + PPO) |
| 병렬 학습 환경 | `num_envs=480` |
| 모션 스타일 | AMP 판별기 보상 사용, `style_imitation=0` |
| AMP 안정화 | `gp_type=r1`, `grad_penalty_coef=1.0` |
| AMP 관절 속도 관측 | `use_velocity=True` |
| 장애물 근접 보상 가중치 | `-4.0` |
| 장애물 안전거리 추가값 | `1.8` |
| 학습 시작 지점 | `checkpoint=null` (새 학습) |

**학습 이력 구분:** 위 명령어는 **처음부터 새 학습을 시작하는 재학습용 명령어**입니다. 저장소에 포함된 **`model_32000.pt`는 학습 기록상 이전 `model_21900.pt` 체크포인트에서 이어서 학습한 C 실험 결과**입니다. 따라서 위 명령어가 `model_32000.pt`를 처음부터 생성할 때 사용된 명령어라는 뜻은 아닙니다.

---

## 11. 장애물 회피 개선: A/B/C 실험

초기에는 장애물이 물리 충돌이 없는 시각적 오브젝트였고, 로봇이 장애물을 스치거나 관통하듯 이동해도 순간적인 페널티만 받는 문제가 있었습니다. 이를 개선하기 위해 세 가지 가설을 비교했습니다.

| 실험 | 변경 사항 | 시연에서 관찰된 동작 |
| --- | --- | --- |
| **A** | 장애물 충돌 시 에피소드 종료 | 충돌 후 초기화되어 같은 장면이 반복되는 경향 |
| **B** | 충돌 페널티 강화 | 장애물에 가까이 접근한 후 급격하게 방향을 바꾸는 경향 |
| **C (최종 채택)** | 근접 페널티 강화 및 회피 시작 거리 확장 | 장애물에 접근하면서 경로가 점진적으로 휘어지는 경향 |

최종 C 실험에서 사용한 주요 설정:

```yaml
rewards:
  obstacle_safe_margin_extra: 1.8
  reward_scales:
    penalty_obstacle_collision: -40.0
    penalty_obstacle_proximity: -4.0
```

또한 충돌·근접 판정 기준을 기존 **골반(root) 위치**에서 **전신 링크 중 장애물과 가장 가까운 위치**로 변경하여, 팔·무릎 등이 장애물을 스치는 상황도 판정에 반영했습니다.

**평가 범위:** A/B/C의 상대적인 동작 차이는 체크포인트별 동일 장면 재생을 통한 **시각적 비교**로 판단했습니다. 세 실험의 충돌률이나 평균 회피 거리를 동일 프로토콜로 정량 비교한 결과는 아닙니다.

---

## 12. 최종 학습 결과

최종 제출 체크포인트: **`C_fullbody_collision` / `model_32000.pt`**

조원의 학습 정리 문서에 기록된 최종 학습 로그(`32,058 iteration`) 기준 지표는 다음과 같습니다.

| 지표 | 결과 | 설명 |
| --- | --- | --- |
| `Env/goal_reach_rate` | **0.668** (마지막 기록값 **0.679**) | 목표 도달 성공률: 약 66.8% / 마지막 기록값 67.9% |
| `Train/mean_episode_length` | **968.2 / 1000 step** | 평균 에피소드 길이 |
| `Train/mean_reward` | **46.5** | 평균 보상 |
| `Episode/rew_penalty_obstacle_collision` | **-0.022** | 충돌 관련 평균 보상 항목 |
| `Episode/rew_penalty_obstacle_proximity` | **-0.064** | 근접 관련 평균 보상 항목 |
| `Loss/Discriminator` | **0.0070** | AMP 판별기 손실 |

표의 평균 지표는 주로 **마지막 50 iteration 평균**이며, 목표 도달률의 **0.679**는 별도로 기록된 마지막 값입니다. 페널티 보상값은 **충돌률(%)과 동일한 지표가 아닙니다.**

최종 시연에서는 목표 지점으로 이동하면서 장애물을 회피하는 동작을 확인했습니다. 단, 본 수치만으로 새로운 환경에서의 일반화 성능이나 실제 로봇 적용 가능성을 입증한 것은 아닙니다.

---

## 13. 포함 파일 구조 (주요 파일)

```text
Humanoidverse/
├─ assets/
├─ humanoidverse/
│  ├─ agents/
│  │  ├─ modules/
│  │  │  └─ amp_discriminator.py       # AMP 판별기
│  │  └─ ppo/
│  │     ├─ ppo.py                     # 기본 PPO
│  │     └─ amp_ppo.py                 # PPO + AMP
│  ├─ config/
│  │  ├─ algo/amp_ppo.yaml
│  │  ├─ exp/locomotion_amp.yaml
│  │  ├─ rewards/loco/reward_h1_locomotion.yaml
│  │  └─ robot/h1/
│  │     ├─ h1.yaml                   # 전신 19DoF
│  │     └─ h1_10dof.yaml             # 초기 10DoF
│  ├─ data/motions/
│  │  ├─ cmu_walk_h1_full19_clips.npz
│  │  └─ lafan1_walk_h1_ik.npz
│  ├─ envs/locomotion/locomotion.py
│  ├─ simulator/genesis/genesis.py
│  ├─ utils/
│  │  ├─ motion_loader.py
│  │  └─ lafan1_ik_pipeline.py
│  ├─ train_agent.py
│  ├─ eval_web_control.py
│  ├─ eval_visualize.py
│  └─ eval_agent.py
├─ logs/
│  ├─ H1_E2E/                       # 초기 10DoF PPO 학습 기록
│  └─ LAFAN1_BVH_IK/
│     └─ 20261003_215621-C_fullbody_collision-locomotion-h1/
│        ├─ config.yaml
│        ├─ model_32000.pt
│        └─ .hydra/
├─ scripts/
│  └─ setup_env.sh
├─ FULLBODY_STYLE_NOTES.md
├─ LICENSE
├─ README.md
└─ pyproject.toml
```

> 위 목록은 주요 파일만 발췌한 구조입니다. AMP 참조 모션 데이터와 최종 체크포인트는 `transfer-temp` 브랜치에 포함되어 있습니다.

---

## 14. 주요 실행 및 구현 파일

- **`humanoidverse/train_agent.py`**: 강화학습 학습 실행 파일
- **`humanoidverse/eval_web_control.py`**: 최종 시연 파일. Genesis Viewer와 로컬 웹 패널 실행
- **`humanoidverse/eval_visualize.py`**: 웹 패널 없이 Viewer 중심으로 평가하는 파일
- **`humanoidverse/eval_agent.py`**: 기존 평가 실행 파일
- **`humanoidverse/agents/ppo/amp_ppo.py`**: PPO와 AMP 보상·판별기 학습 결합
- **`humanoidverse/agents/modules/amp_discriminator.py`**: 참조 모션/정책 모션 구분 및 스타일 보상 계산
- **`humanoidverse/envs/locomotion/locomotion.py`**: 목표·장애물·보상함수·커리큘럼·전신 충돌 판정
- **`humanoidverse/utils/lafan1_ik_pipeline.py`**: LAFAN1 BVH → H1 관절 모션 데이터 생성
- **`humanoidverse/simulator/genesis/genesis.py`**: Genesis 환경과 Viewer 마커 구성
- **`scripts/setup_env.sh`**: 최초 실행 환경 설치

---

## 15. Checkpoint 및 데이터

최종 시연에 사용하는 체크포인트는 다음 경로에 포함되어 있습니다.

```text
logs/LAFAN1_BVH_IK/20261003_215621-C_fullbody_collision-locomotion-h1/model_32000.pt
```

평가 스크립트는 같은 학습 폴더의 `config.yaml`을 로드하므로 **별도의 19DoF/AMP 설정을 직접 입력하지 않아도 됩니다.**

AMP 판별기의 참조 모션 데이터는 다음 경로입니다.

```text
humanoidverse/data/motions/lafan1_walk_h1_ik.npz
```

현재 코드에서는 AMP 알고리즘의 초기화 과정에서 참조 모션 데이터를 로드하므로, **평가 시에도 해당 파일이 필요합니다.**

---

## 16. 문제 해결

### 16.1 `conda activate hgen`이 동작하지 않는 경우

Conda가 Bash에 초기화되어 있는지 확인합니다.

```bash
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate hgen
```

Anaconda를 사용하는 경우 경로를 `$HOME/anaconda3/etc/profile.d/conda.sh`로 변경합니다.

### 16.2 웹 페이지가 열리지 않는 경우

기본 접속 주소는 `http://localhost:8080`입니다. 시연 스크립트가 실행된 상태인지 확인합니다.

이전 웹 서버 프로세스가 8080 포트를 점유한 경우 해당 프로세스를 종료한 뒤 **7절의 시연 명령어**를 다시 실행합니다.

```bash
pkill -f eval_web_control.py
```

### 16.3 체크포인트를 찾지 못하는 경우

프로젝트 루트에서 다음 경로가 존재하는지 확인합니다.

```bash
ls -lh logs/LAFAN1_BVH_IK/20261003_215621-C_fullbody_collision-locomotion-h1/model_32000.pt
```

`transfer-temp` 브랜치인지, 현재 터미널 위치가 프로젝트 루트인지 확인합니다.

### 16.4 AMP 참조 모션 데이터 파일을 찾지 못하는 경우

```bash
ls -lh humanoidverse/data/motions/lafan1_walk_h1_ik.npz
```

해당 파일이 없다면 `transfer-temp` 브랜치의 최신 내용을 받아야 합니다.

### 16.5 Genesis Viewer가 뜨지 않는 경우

WSL2, NVIDIA GPU 드라이버, CUDA, OpenGL 환경을 확인합니다.

```bash
nvidia-smi
xeyes
```

WSL2 환경에서 그래픽 라이브러리 설정이 필요한 경우, 기존 시연 스크립트에서 사용하던 아래 환경변수를 적용한 뒤 다시 실행할 수 있습니다.

```bash
export PYOPENGL_PLATFORM=glx
export LIBGL_ALWAYS_INDIRECT=0
export GALLIUM_DRIVER=d3d12
export MESA_D3D12_DEFAULT_ADAPTER_NAME=NVIDIA
export LD_LIBRARY_PATH="/usr/lib/wsl/lib:${LD_LIBRARY_PATH:-}"
```

이 설정은 **WSL2 + NVIDIA 환경에서 사용하는 선택적 문제 해결 방법**이며, 모든 시스템에 필요한 것은 아닙니다.

### 16.6 설치 스크립트 실행 권한 오류

```bash
chmod +x scripts/setup_env.sh
```

---

## 17. 빠른 실행 요약

최초 설치:

```bash
git clone --branch transfer-temp https://github.com/dulgiDove/Humanoidverse.git
cd Humanoidverse
chmod +x scripts/setup_env.sh
./scripts/setup_env.sh
```

설치 완료 후 시연:

```bash
conda activate hgen
python humanoidverse/eval_web_control.py \
  +checkpoint=logs/LAFAN1_BVH_IK/20261003_215621-C_fullbody_collision-locomotion-h1/model_32000.pt
```

브라우저 접속:

```text
http://localhost:8080
```

---

## 18. 최종 제출물 설명

본 프로젝트는 **H1 19DoF 휴머노이드가 강화학습 기반 E2E 정책으로 목표 지점에 접근하고 장애물을 회피하는 시스템**입니다. 초기 10DoF PPO 내비게이션 환경을 확장하여, **LAFAN1 기반 AMP 보행 학습**, **장애물 근접 보상 개선**, **전신 링크 거리 판정**을 적용했습니다.

최종 결과물로 `C_fullbody_collision/model_32000.pt` 체크포인트와 관련 설정·모션 데이터가 포함되어 있으며, **Genesis Viewer + 웹 컨트롤 패널**을 통해 목표 및 장애물 배치를 변경하면서 동작을 시연할 수 있습니다.

최종 학습·시연 방법과 실험 결과는 위 절을 참고해 주세요.
