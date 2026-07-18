# 집 컴퓨터 세팅 가이드 (Claude에게)

이 파일을 읽고 아래 순서대로 진행해줘.

---

## 프로젝트 개요

**한양대 졸업프로젝트**: 복잡한 상호작용 환경에서 자연스러운 움직임 시뮬레이션
- Genesis + PPO 기반 휴머노이드 캐릭터가 장애물 회피하며 목표 지점 도달
- **정상일 담당**: 보상 함수 설계 (style imitation - CMU mocap 모방)
- **이정민 담당**: 환경 구축, 장애물, 시각화

**팀 레포**: https://github.com/dulgiDove/Humanoidverse.git
**브랜치**: master (정상일 작업), e2e-goal-reaching (이정민 작업)

---

## 노트북에서 이미 완료된 것

1. CMU mocap 전처리 → `humanoidverse/data/motions/cmu_walk_h1.npy` (shape: 12811, 10)
2. style imitation 보상 함수 구현 (`locomotion.py`의 `_reward_style_imitation()`)
3. eval_agent.py 단순화 (headless MP4 저장 방식)
4. 위 내용 GitHub master 브랜치에 push 완료
5. 노트북에서 StyleImitation 학습 완료 → 체크포인트가 OneDrive나 USB로 전달됨

---

## 집 컴퓨터 세팅 순서

### 1단계 - Miniconda 설치 (WSL Ubuntu에서)

```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
```
설치 완료 후 터미널 재시작 (또는 `source ~/.bashrc`)

### 2단계 - 레포 클론

```bash
git clone https://github.com/dulgiDove/Humanoidverse.git
cd Humanoidverse
```

### 3단계 - conda 환경 생성 및 패키지 설치

```bash
conda create -n humanoidverse python=3.10
conda activate humanoidverse
pip install torch
pip install genesis-world==0.2.1
pip install -e .
```

### 4단계 - CUDA 설정 (중요! 안 하면 CPU로 돌아감)

WSL에서 Genesis가 CUDA를 못 찾는 문제가 있음. 매번 학습 전에 아래를 실행해야 함:

```bash
export LD_LIBRARY_PATH=/usr/lib/wsl/lib:$LD_LIBRARY_PATH
```

영구 적용하려면:
```bash
echo 'export LD_LIBRARY_PATH=/usr/lib/wsl/lib:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc
```

### 5단계 - 노트북에서 받아야 할 파일

아래 파일들은 git에 올라가지 않음 (gitignore). 노트북에서 OneDrive/USB로 받아야 함:

| 파일 | 설명 |
|------|------|
| `humanoidverse/data/motions/cmu_walk_h1.npy` | CMU mocap 전처리 데이터 (필수) |
| `logs/StyleImitation/.../model_XXX.pt` | 노트북에서 학습한 체크포인트 |

파일 받은 후 같은 경로에 넣기:
- `cmu_walk_h1.npy` → `Humanoidverse/humanoidverse/data/motions/cmu_walk_h1.npy`
- 체크포인트 → `Humanoidverse/logs/StyleImitation/...` (폴더 구조 그대로)

### 6단계 - 학습 이어서 돌리기

처음부터 돌릴 경우:
```bash
export LD_LIBRARY_PATH=/usr/lib/wsl/lib:$LD_LIBRARY_PATH
conda activate humanoidverse
python humanoidverse/train_agent.py \
+simulator=genesis \
+exp=locomotion \
+domain_rand=NO_domain_rand \
+rewards=loco/reward_h1_locomotion \
+robot=h1/h1_10dof \
+terrain=terrain_locomotion_plane \
+obs=loco/leggedloco_obs_singlestep_withlinvel \
num_envs=512 \
project_name=StyleImitation \
experiment_name=H110dof_style \
headless=True
```

노트북 체크포인트에서 이어서 돌릴 경우 (`+checkpoint` 추가):
```bash
export LD_LIBRARY_PATH=/usr/lib/wsl/lib:$LD_LIBRARY_PATH
conda activate humanoidverse
python humanoidverse/train_agent.py \
+simulator=genesis \
+exp=locomotion \
+domain_rand=NO_domain_rand \
+rewards=loco/reward_h1_locomotion \
+robot=h1/h1_10dof \
+terrain=terrain_locomotion_plane \
+obs=loco/leggedloco_obs_singlestep_withlinvel \
num_envs=512 \
project_name=StyleImitation \
experiment_name=H110dof_style \
headless=True \
+checkpoint=logs/StyleImitation/.../model_XXX.pt
```
(경로는 실제 체크포인트 파일 경로로 바꿀 것)

### 7단계 - 학습 결과 시각화 (MP4 저장)

```bash
export LD_LIBRARY_PATH=/usr/lib/wsl/lib:$LD_LIBRARY_PATH
conda activate humanoidverse
python humanoidverse/eval_agent.py \
+checkpoint=logs/StyleImitation/.../model_XXX.pt
```

MP4 파일이 체크포인트 폴더 안에 `progress_ckpt_{번호}.mp4`로 저장됨.

---

## 참고사항

- `num_envs=512`는 RTX 4050 기준. 메모리 오류 나면 256으로 줄이기
- 학습 중간 확인은 tensorboard로: `tensorboard --logdir logs/`
- 체크포인트는 100 iteration마다 자동 저장됨 (`save_interval: 100`)
- 학습 시간 제한하려면 `timeout 3h python ...` 앞에 붙이기
- git push 설정: `git remote set-url origin https://jsi1022:토큰@github.com/dulgiDove/Humanoidverse.git`
  (토큰은 GitHub Settings → Developer settings → Personal access tokens에서 발급)
