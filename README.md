# Humanoid Navigation Challenge

## 1. 프로젝트 개요

본 프로젝트는 **HumanoidVerse**와 **Genesis Simulator**를 기반으로 한 휴머노이드 로봇 주행 데모입니다.

H1 10DoF 휴머노이드 로봇이 학습된 강화학습 정책을 사용하여 목표 지점까지 이동하고, 주변 장애물을 회피하는 과정을 Genesis Viewer에서 시각화합니다.

또한 로컬 웹 컨트롤 패널을 제공하여 목표 좌표와 장애물 좌표를 직접 입력하거나 랜덤으로 생성한 뒤 시뮬레이션에 적용할 수 있습니다.

---

## 2. 주요 기능

- H1 10DoF 휴머노이드 로봇 시뮬레이션
- 학습된 PPO 정책 기반 목표 지점 이동
- 3개의 장애물 회피
- Genesis Viewer 기반 3D 시각화
- 로컬 웹 컨트롤 패널 제공
- Target / Obstacle 좌표 직접 입력
- Target / Obstacle 좌표 랜덤 생성
- Start / Pause / Reset / Info / Fit Camera 기능
- 목표 도달 후 새로운 목표 및 장애물 자동 갱신

---

## 3. 실행 환경

본 프로젝트는 다음 환경에서 테스트되었습니다.

- Windows 11
- WSL2 Ubuntu
- NVIDIA GPU
- Miniconda
- Python 3.10
- PyTorch CUDA 12.1
- Genesis World 0.2.1
- HumanoidVerse

주의: 본 프로젝트는 Genesis Viewer와 CUDA 기반 PyTorch를 사용하므로, Windows 환경에서는 WSL2와 NVIDIA GPU 드라이버가 정상적으로 설정되어 있어야 합니다.

---

## 4. 사전 준비

실행 전에 다음 항목이 설치되어 있어야 합니다.

- WSL2 Ubuntu
- Miniconda 또는 Anaconda
- NVIDIA GPU Driver
- WSL에서 CUDA 사용 가능 환경

CUDA 사용 가능 여부는 다음 명령으로 확인할 수 있습니다.

```bash
nvidia-smi
```

WSL에서 GUI 앱이 실행 가능한지도 확인하는 것이 좋습니다.

```bash
xeyes
```

`xeyes`가 없는 경우 아래 명령으로 설치할 수 있습니다.

```bash
sudo apt update
sudo apt install -y x11-apps
```

---

## 5. 설치 방법

압축 파일을 해제한 뒤 프로젝트 폴더로 이동합니다.

```bash
cd HumanoidVerse_submit_clean
```

압축 해제 과정에서 `.sh` 파일의 실행 권한이 사라질 수 있으므로, 먼저 실행 권한을 부여합니다.

```bash
chmod +x scripts/setup_env.sh scripts/run_demo.sh
```

처음 한 번만 환경 설치 스크립트를 실행합니다.

```bash
./scripts/setup_env.sh
```

이 스크립트는 다음 작업을 수행합니다.

- Ubuntu OpenGL / viewer 관련 패키지 설치
- Conda 환경 생성
- Python 3.10 환경 구성
- PyTorch CUDA 12.1 설치
- HumanoidVerse 로컬 프로젝트 설치
- Genesis World 0.2.1 설치
- numpy 1.26.4 고정
- rsl_rl 설치
- tensorboard, pynput, rich, termcolor 등 추가 의존성 설치
- libigl, pyglet, PyOpenGL 버전 고정

기본 Conda 환경 이름은 `hgen`입니다.

다른 환경 이름을 사용하려면 다음처럼 실행할 수 있습니다.

```bash
ENV_NAME=hgen_submit_test ./scripts/setup_env.sh
```

---

## 6. 실행 방법

설치가 완료되면 다음 명령으로 데모를 실행합니다.

```bash
./scripts/run_demo.sh
```

다른 Conda 환경 이름으로 설치했다면 실행 시에도 같은 환경 이름을 지정합니다.

```bash
ENV_NAME=hgen_submit_test ./scripts/run_demo.sh
```

정상 실행 시 Genesis Viewer가 열리고, 터미널에 다음과 같은 웹 패널 주소가 출력됩니다.

```text
http://localhost:8080
```

웹 브라우저에서 위 주소로 접속하면 로컬 웹 컨트롤 패널을 사용할 수 있습니다.

---

## 7. 웹 컨트롤 패널 사용법

웹 브라우저에서 다음 주소에 접속합니다.

```text
http://localhost:8080
```

### Coordinate Setup

- **Target X/Y**: 목표 지점 좌표를 입력합니다.
- **Obstacle 1/2/3 X/Y**: 장애물 좌표를 입력합니다.
- **Random**: 해당 항목의 좌표를 랜덤으로 생성합니다.
- **Random All**: 목표와 모든 장애물 좌표를 랜덤으로 생성합니다.
- **Apply**: 입력된 좌표를 Genesis 시뮬레이션에 적용합니다.

### Simulation Control

- **Start**: 로봇 정책 실행을 시작합니다.
- **Pause**: 시뮬레이션 진행을 일시정지합니다.
- **Reset**: 로봇과 환경 상태를 초기화합니다. 마지막으로 Apply한 Target / Obstacle 좌표가 있으면 해당 배치를 유지한 채 초기화됩니다.
- **Info**: 현재 로봇 위치, 목표 위치, 장애물 위치, 목표까지의 거리 등을 출력합니다.
- **Fit Camera**: 로봇, 목표, 장애물이 한 화면에 보이도록 카메라 위치를 조정합니다.

---

## 8. Random 좌표 생성 범위

Random 버튼은 현재 로봇 위치를 기준으로 좌표를 생성합니다.

- Target Random: 로봇 기준 거리 1.5m ~ 8.0m
- Obstacle Random: 로봇 기준 거리 1.5m ~ 5.0m
- Target과 Obstacle, Obstacle끼리는 최소 0.8m 이상 떨어지도록 생성

주의: 좌표의 X/Y 각각이 위 범위로 제한되는 것이 아니라, 로봇으로부터의 거리 기준입니다.

---

## 9. 포함 파일 구조

```text
HumanoidVerse_submit_clean/
├─ assets/
├─ humanoidverse/
│  ├─ eval_web_control.py
│  ├─ eval_visualize.py
│  └─ eval_agent.py
├─ logs/
│  └─ H1_E2E/
│     └─ 최신 학습 결과 폴더/
│        └─ model_*.pt
├─ scripts/
│  ├─ setup_env.sh
│  └─ run_demo.sh
├─ LICENSE
├─ README.md
└─ pyproject.toml
```

---

## 10. 주요 실행 파일

### `humanoidverse/eval_web_control.py`

최종 데모 실행 파일입니다. Genesis Viewer와 로컬 웹 컨트롤 패널을 함께 실행합니다.

### `humanoidverse/eval_visualize.py`

웹 컨트롤 패널 없이 Genesis Viewer만 실행하는 백업용 시각화 파일입니다.

### `humanoidverse/eval_agent.py`

기존 평가 실행 파일입니다.

---

## 11. Checkpoint

본 제출물에는 학습된 checkpoint가 포함되어 있습니다.

데모 실행 시 `scripts/run_demo.sh`가 아래 경로에서 최신 checkpoint를 자동으로 찾습니다.

```bash
find logs/H1_E2E -name "model_*.pt" | sort -V | tail -1
```

따라서 별도의 학습 과정 없이 데모 실행이 가능합니다.

---

## 12. 문제 해결

### 1. 웹 페이지가 열리지 않는 경우

기본 주소는 다음과 같습니다.

```text
http://localhost:8080
```

기존 실행 프로세스가 남아 있으면 8080 포트가 충돌할 수 있습니다.

```bash
pkill -f eval_web_control.py
```

그 후 다시 실행합니다.

```bash
./scripts/run_demo.sh
```

### 2. checkpoint를 찾지 못하는 경우

다음 명령으로 checkpoint가 존재하는지 확인합니다.

```bash
find logs/H1_E2E -name "model_*.pt" | sort -V
```

`model_*.pt` 파일이 존재해야 데모를 실행할 수 있습니다.

### 3. Genesis Viewer가 뜨지 않는 경우

WSL2, NVIDIA Driver, OpenGL 환경 문제일 수 있습니다. 다음 명령으로 GPU가 인식되는지 확인합니다.

```bash
nvidia-smi
```

WSL에서 GUI 앱 실행이 가능한지도 확인합니다.

```bash
xeyes
```

### 4. `.sh` 파일 실행 권한 오류

압축 해제 후 실행 권한이 사라질 수 있습니다. 다음 명령을 다시 실행합니다.

```bash
chmod +x scripts/setup_env.sh scripts/run_demo.sh
```

### 5. Conda 환경 이름을 바꾸고 싶은 경우

기본 환경 이름은 `hgen`입니다. 다른 이름을 사용하려면 설치와 실행 모두 같은 이름을 지정해야 합니다.

```bash
ENV_NAME=hgen_submit_test ./scripts/setup_env.sh
ENV_NAME=hgen_submit_test ./scripts/run_demo.sh
```

---

## 13. 실행 요약

처음 실행 시:

```bash
cd HumanoidVerse_submit_clean
chmod +x scripts/setup_env.sh scripts/run_demo.sh
./scripts/setup_env.sh
./scripts/run_demo.sh
```

웹 패널 접속:

```text
http://localhost:8080
```

---

## 14. 제출물 설명

본 제출물은 학습된 정책 기반의 H1 10DoF 휴머노이드 목표 도달 및 장애물 회피 데모입니다.

`setup_env.sh`를 통해 실행 환경을 구성하고, `run_demo.sh`를 통해 Genesis Viewer와 웹 컨트롤 패널을 실행할 수 있습니다.