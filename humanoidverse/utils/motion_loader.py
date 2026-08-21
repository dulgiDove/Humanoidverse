"""
[모션 전처리 스크립트 v2]
작성자: 정상일
목적: LAFAN1 데이터셋을 Unitree H1로 IK 리타게팅한 CSV 파일을 읽어서
      style imitation 학습용 클립 데이터(.npz)로 변환한다.

데이터 출처: https://huggingface.co/datasets/lvhaidong/LAFAN1_Retargeting_Dataset
 - IK + Interaction Mesh 기반으로 H1 관절 각도를 이미 계산해둔 데이터
 - 30FPS, 걷기 시퀀스만 사용 (walk1~4)

CSV 포맷 (26열):
  [0:7]   root_joint (x, y, z, qx, qy, qz, qw) ← 사용 안 함
  [7:26]  H1 19-DOF 관절 각도 (라디안)
    7:  left_hip_yaw        15: right_hip_yaw
    8:  left_hip_roll       16: right_hip_roll
    9:  left_hip_pitch      17: right_hip_pitch
    10: left_knee           18: right_knee
    11: left_ankle          19: right_ankle
    12: torso               20: torso (→ idx 10)
    ...
    (README의 H1 순서 그대로)

출력:
  humanoidverse/data/motions/cmu_walk_h1_full19_clips.npz
  clips: (num_clips, max_len, 19), lengths: (num_clips,) int64, 50Hz
"""

import os
import sys

# utils/math.py 가 stdlib math를 가리는 문제 방지
_this_dir = os.path.dirname(os.path.abspath(__file__))
if _this_dir in sys.path:
    sys.path.remove(_this_dir)

import numpy as np

# ── 경로 설정 ─────────────────────────────────────────────────────────────
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT    = os.path.join(SCRIPT_DIR, "..", "..")
LAFAN1_H1_DIR = "/mnt/c/Users/jayelle/lafan1_hf/h1"
OUT_PATH     = os.path.join(REPO_ROOT, "humanoidverse", "data", "motions",
                             "cmu_walk_h1_full19_clips.npz")

# ── 파라미터 ──────────────────────────────────────────────────────────────
SRC_FPS    = 30      # LAFAN1 CSV FPS
TARGET_FPS = 50      # 시뮬레이션 FPS
CLIP_LEN   = 200     # 클립 길이 (50Hz 기준 프레임 수, = 4초)
CLIP_STEP  = 100     # 클립 간격 (겹침 허용, = 2초)
MIN_CLIP   = 50      # 최소 클립 길이

# 사용할 동작 종류 (walk만)
WALK_PREFIXES = ['walk1', 'walk2', 'walk3', 'walk4']


def resample(data: np.ndarray, src_fps: float, tgt_fps: float) -> np.ndarray:
    """(T, D) 배열을 src_fps → tgt_fps 로 선형 보간."""
    T = data.shape[0]
    duration  = T / src_fps
    src_t = np.linspace(0, duration, T)
    tgt_t = np.linspace(0, duration, max(1, int(round(T * tgt_fps / src_fps))))
    out = np.zeros((len(tgt_t), data.shape[1]), dtype=np.float32)
    for d in range(data.shape[1]):
        out[:, d] = np.interp(tgt_t, src_t, data[:, d])
    return out


def load_csv(path: str) -> np.ndarray:
    """CSV 파일 → (T, 19) H1 관절 각도 배열 (라디안)."""
    raw = np.loadtxt(path, delimiter=',', dtype=np.float32)  # (T, 26)
    return raw[:, 7:]   # 앞 7열(루트) 제거 → (T, 19)


def split_clips(data: np.ndarray, clip_len: int, clip_step: int) -> list:
    """(T, 19) 시퀀스를 고정 길이 클립으로 분할."""
    clips = []
    T = len(data)
    start = 0
    while start + clip_len <= T:
        clips.append(data[start:start + clip_len].copy())
        start += clip_step
    # 남은 부분이 MIN_CLIP 이상이면 추가
    if start < T and (T - start) >= MIN_CLIP:
        clips.append(data[start:T].copy())
    return clips


def main():
    all_clips = []

    csv_files = sorted(os.listdir(LAFAN1_H1_DIR))
    walk_files = [f for f in csv_files
                  if f.endswith('.csv') and any(f.startswith(p) for p in WALK_PREFIXES)]

    print(f"걷기 CSV 파일 {len(walk_files)}개 처리 중...")

    for fname in walk_files:
        fpath = os.path.join(LAFAN1_H1_DIR, fname)
        try:
            joints = load_csv(fpath)              # (T, 19), 30Hz
            joints_50 = resample(joints, SRC_FPS, TARGET_FPS)  # → 50Hz

            clips = split_clips(joints_50, CLIP_LEN, CLIP_STEP)
            print(f"  {fname}: {len(joints)}f@30Hz → {len(joints_50)}f@50Hz → {len(clips)}클립")
            all_clips.extend(clips)

        except Exception as e:
            print(f"  ✗ {fname}: {e}")

    if not all_clips:
        print("오류: 처리된 클립이 없습니다.")
        return

    lengths  = [len(c) for c in all_clips]
    max_len  = max(lengths)
    padded   = np.zeros((len(all_clips), max_len, 19), dtype=np.float32)
    for i, c in enumerate(all_clips):
        padded[i, :len(c)] = c
    lengths_arr = np.array(lengths, dtype=np.int64)

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    np.savez(OUT_PATH, clips=padded, lengths=lengths_arr)

    print(f"\n총 {len(all_clips)}개 클립 저장 완료: {OUT_PATH}")
    print(f"shape: {padded.shape}  길이 범위: {min(lengths)}~{max(lengths)}프레임")

    # 관절별 통계
    all_frames = np.concatenate(all_clips, axis=0)
    labels = ['L_hip_yaw','L_hip_roll','L_hip_pitch','L_knee','L_ankle',
              'R_hip_yaw','R_hip_roll','R_hip_pitch','R_knee','R_ankle',
              'torso',
              'L_shoulder_pitch','L_shoulder_roll','L_shoulder_yaw','L_elbow',
              'R_shoulder_pitch','R_shoulder_roll','R_shoulder_yaw','R_elbow']
    print("\n관절별 범위 (라디안):")
    for i, label in enumerate(labels):
        mn, mx = all_frames[:, i].min(), all_frames[:, i].max()
        print(f"  {label:20s}: [{mn:+.3f}, {mx:+.3f}]")


if __name__ == '__main__':
    main()
