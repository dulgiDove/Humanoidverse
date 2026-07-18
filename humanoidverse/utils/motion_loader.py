"""
[모션 전처리 스크립트]
작성자: 정상일
목적: AMASS CMU mocap 데이터(.npz)를 읽어서 걷기 시퀀스만 추출하고,
      H1 로봇 하체 10 DOF 관절 각도 형식으로 변환 후 .npy로 저장한다.

사용법 (한 번만 실행하면 됨):
    python humanoidverse/utils/motion_loader.py

출력:
    humanoidverse/data/motions/cmu_walk_h1.npy
    shape: (총 프레임 수, 10) — float32
    10 DOF 순서 (h1.yaml 기준):
        0: left_hip_yaw
        1: left_hip_roll
        2: left_hip_pitch
        3: left_knee
        4: left_ankle
        5: right_hip_yaw
        6: right_hip_roll
        7: right_hip_pitch
        8: right_knee
        9: right_ankle

SMPL+H poses 배열 (156차원) 내 관절 인덱스:
    joint_i의 axis-angle = poses[:, i*3 : i*3+3]
    1: left_hip, 2: right_hip
    4: left_knee, 5: right_knee
    7: left_ankle, 8: right_ankle
"""

import sys
import os

# ── [중요] utils/math.py 가 표준 라이브러리 math 를 가리는 문제 방지 ──────────
# Python은 스크립트 실행 시 스크립트가 있는 폴더를 sys.path 맨 앞에 자동 추가한다.
# 이 폴더(humanoidverse/utils/)에 math.py 가 있어서 numpy 등이 import 에 실패한다.
# 아래 코드로 현재 폴더를 sys.path 에서 미리 제거해서 충돌을 방지한다.
_this_dir = os.path.dirname(os.path.abspath(__file__))
if _this_dir in sys.path:
    sys.path.remove(_this_dir)

import numpy as np
from scipy.spatial.transform import Rotation

# ────────────────────────────────────────────────
# 경로 설정
# ────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))   # humanoidverse/utils/
REPO_ROOT  = os.path.join(SCRIPT_DIR, "..", "..")         # 레포 루트
CMU_DIR    = os.path.join(REPO_ROOT, "humanoidverse", "data", "motions", "CMU")
OUT_PATH   = os.path.join(REPO_ROOT, "humanoidverse", "data", "motions", "cmu_walk_h1.npy")

# 시뮬레이션 주파수 (Hz) — locomotion.py의 dt=0.02 기준 50Hz
TARGET_FPS = 50

# 걷기 판별 기준
MIN_DURATION_SEC   = 2.0   # 최소 2초 이상
MIN_AVG_SPEED_MPS  = 0.3   # 평균 이동 속도 0.3 m/s 이상 (수평 거리 / 시간)

# CMU mocap 에서 걷기 동작으로 잘 알려진 subject 번호
# (숫자가 아닌 폴더명은 스킵)
KNOWN_WALKING_SUBJECTS = {
    '07', '08', '09', '35', '36', '37', '38', '39',
    '64', '70', '02', '16'
}


# ────────────────────────────────────────────────
# 헬퍼 함수
# ────────────────────────────────────────────────

def axis_angle_to_euler_xyz(aa_seq: np.ndarray) -> np.ndarray:
    """
    axis-angle 시퀀스를 Euler XYZ 각도로 변환.
    aa_seq: (T, 3)
    반환값: (T, 3) — [roll(x), pitch(y), yaw(z)] in radians
    """
    return Rotation.from_rotvec(aa_seq).as_euler('xyz')


def extract_lower_body_h1(poses: np.ndarray) -> np.ndarray:
    """
    SMPL+H poses (T, 156) → H1 하체 10 DOF (T, 10)

    SMPL → H1 관절 매핑:
        SMPL left_hip  (joint 1) euler_xyz → [yaw(y), roll(x), pitch(z)] → H1 [0,1,2]
        SMPL left_knee (joint 4) euler_xyz → pitch(z)                     → H1 [3]
        SMPL left_ankle(joint 7) euler_xyz → roll(x)                      → H1 [4]
        오른쪽도 동일하게 매핑                                             → H1 [5~9]

    부호 규약:
        - knee 굽힘 방향이 H1에서 양수(+)여야 하므로 부호 반전 적용
        - ankle 족저굴곡이 H1에서 음수(-)여야 하므로 부호 반전 적용
    """
    T = poses.shape[0]
    joints = np.zeros((T, 10), dtype=np.float32)

    # ── 왼쪽 하체 ──────────────────────────────────
    lhip   = axis_angle_to_euler_xyz(poses[:, 1*3:1*3+3])
    lknee  = axis_angle_to_euler_xyz(poses[:, 4*3:4*3+3])
    lankle = axis_angle_to_euler_xyz(poses[:, 7*3:7*3+3])

    joints[:, 0] =  lhip[:, 1]    # left_hip_yaw   ← y축 회전
    joints[:, 1] =  lhip[:, 0]    # left_hip_roll  ← x축 회전
    joints[:, 2] =  lhip[:, 2]    # left_hip_pitch ← z축 회전
    joints[:, 3] = -lknee[:, 2]   # left_knee      ← z축 회전 (부호 반전: 굽힘=+)
    joints[:, 4] = -lankle[:, 0]  # left_ankle     ← x축 회전 (부호 반전: 족저굴곡=-)

    # ── 오른쪽 하체 ────────────────────────────────
    rhip   = axis_angle_to_euler_xyz(poses[:, 2*3:2*3+3])
    rknee  = axis_angle_to_euler_xyz(poses[:, 5*3:5*3+3])
    rankle = axis_angle_to_euler_xyz(poses[:, 8*3:8*3+3])

    joints[:, 5] = -rhip[:, 1]    # right_hip_yaw  ← y축 회전 (좌우 대칭 부호 반전)
    joints[:, 6] = -rhip[:, 0]    # right_hip_roll ← x축 회전 (좌우 대칭 부호 반전)
    joints[:, 7] =  rhip[:, 2]    # right_hip_pitch
    joints[:, 8] = -rknee[:, 2]   # right_knee
    joints[:, 9] = -rankle[:, 0]  # right_ankle

    return joints


def resample(data: np.ndarray, orig_fps: float, target_fps: float) -> np.ndarray:
    """
    (T, D) 모션 데이터를 orig_fps → target_fps 로 리샘플링.
    선형 보간(np.interp) 사용.
    """
    T = data.shape[0]
    duration = T / orig_fps
    orig_t   = np.linspace(0, duration, T)
    target_t = np.linspace(0, duration, max(1, int(T * target_fps / orig_fps)))
    out = np.zeros((len(target_t), data.shape[1]), dtype=np.float32)
    for d in range(data.shape[1]):
        out[:, d] = np.interp(target_t, orig_t, data[:, d])
    return out


def is_walking(trans: np.ndarray, fps: float) -> bool:
    """
    trans (T, 3) 루트 이동 궤적으로 걷기 여부를 판별.
    - 충분한 길이인가 (MIN_DURATION_SEC)
    - 수평 이동 속도가 걷기 수준인가 (MIN_AVG_SPEED_MPS)
    """
    T = trans.shape[0]
    duration = T / fps
    if duration < MIN_DURATION_SEC:
        return False

    # 수평(xz 평면) 총 이동 거리
    horiz_displacement = np.sqrt(
        (trans[-1, 0] - trans[0, 0])**2 +
        (trans[-1, 2] - trans[0, 2])**2
    )
    avg_speed = horiz_displacement / duration
    return avg_speed >= MIN_AVG_SPEED_MPS


# ────────────────────────────────────────────────
# 메인 전처리 루틴
# ────────────────────────────────────────────────

def main():
    all_clips = []
    total_files = 0
    accepted = 0

    subject_dirs = sorted(os.listdir(CMU_DIR))

    for subj in subject_dirs:
        # 알려진 걷기 subject만 처리 (속도 절약)
        if subj not in KNOWN_WALKING_SUBJECTS:
            continue

        subj_path = os.path.join(CMU_DIR, subj)
        if not os.path.isdir(subj_path):
            continue

        for fname in sorted(os.listdir(subj_path)):
            if not fname.endswith('_poses.npz'):
                continue
            total_files += 1
            fpath = os.path.join(subj_path, fname)

            try:
                data = np.load(fpath)
                poses = data['poses']       # (T, 156)
                trans = data['trans']       # (T, 3)
                fps   = float(data['mocap_framerate'])

                # 걷기 시퀀스인지 판별
                if not is_walking(trans, fps):
                    continue

                # H1 하체 10 DOF 추출
                lower_body = extract_lower_body_h1(poses)  # (T, 10)

                # TARGET_FPS 로 리샘플링
                lower_body_resampled = resample(lower_body, fps, TARGET_FPS)

                all_clips.append(lower_body_resampled)
                accepted += 1
                print(f"  ✓ {subj}/{fname}  frames={lower_body_resampled.shape[0]}")

            except Exception as e:
                print(f"  ✗ {subj}/{fname}: {e}")

    print(f"\n전체 {total_files}개 파일 중 {accepted}개 걷기 시퀀스 채택")

    if len(all_clips) == 0:
        print("채택된 시퀀스가 없습니다. KNOWN_WALKING_SUBJECTS 또는 MIN_AVG_SPEED_MPS 기준을 조정하세요.")
        return

    # 시퀀스 이어붙이기
    ref_motion = np.concatenate(all_clips, axis=0)  # (총 프레임, 10)
    print(f"최종 reference motion: shape={ref_motion.shape}, dtype={ref_motion.dtype}")

    # 저장
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    np.save(OUT_PATH, ref_motion)
    print(f"저장 완료: {OUT_PATH}")

    # 간단한 통계 출력
    print("\n관절별 평균/표준편차 (라디안):")
    labels = ['L_yaw','L_roll','L_pitch','L_knee','L_ankle',
              'R_yaw','R_roll','R_pitch','R_knee','R_ankle']
    for i, label in enumerate(labels):
        print(f"  {label:10s}: mean={ref_motion[:,i].mean():.3f}, std={ref_motion[:,i].std():.3f}")


if __name__ == '__main__':
    main()
