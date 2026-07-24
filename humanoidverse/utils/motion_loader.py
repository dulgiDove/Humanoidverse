"""
[모션 전처리 스크립트]
작성자: 정상일
목적: AMASS CMU mocap 데이터(.npz)를 읽어서 걷기 시퀀스만 추출하고,
      H1 로봇 하체 10 DOF + 상체(토르소+양팔) 9 DOF, 총 19 DOF 관절 각도로
      변환 후 클립 단위(.npz)로 저장한다. (2026.07.21: 상체 추가)

사용법 (한 번만 실행하면 됨):
    python humanoidverse/utils/motion_loader.py

출력:
    humanoidverse/data/motions/cmu_walk_h1_full19_clips.npz
    clips: (num_clips, max_len, 19) float32, lengths: (num_clips,) int64
    19 DOF 순서 (h1.yaml dof_names 기준):
        0: left_hip_yaw       5: right_hip_yaw
        1: left_hip_roll      6: right_hip_roll
        2: left_hip_pitch     7: right_hip_pitch
        3: left_knee          8: right_knee
        4: left_ankle         9: right_ankle
        10: torso
        11: left_shoulder_pitch   15: right_shoulder_pitch
        12: left_shoulder_roll    16: right_shoulder_roll
        13: left_shoulder_yaw     17: right_shoulder_yaw
        14: left_elbow            18: right_elbow

SMPL+H poses 배열 (156차원) 내 관절 인덱스 (표준 SMPL body joint 순서):
    joint_i의 axis-angle = poses[:, i*3 : i*3+3]
    0: pelvis
    1: left_hip,    2: right_hip
    3: spine1,      6: spine2,      9: spine3
    4: left_knee,   5: right_knee
    7: left_ankle,  8: right_ankle
    13: left_collar, 14: right_collar
    16: left_shoulder, 17: right_shoulder
    18: left_elbow,    19: right_elbow

클립 분리: 서로 다른 클립을 이어붙이면 경계에서 관절이 순간이동하는 문제가
있었음 (2026.07.20 발견). 이번엔 처음부터 클립을 이어붙이지 않고
클립별로 분리된 상태로 저장한다.
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
# 원본 CMU 데이터(4.4GB)는 리포에 안 넣고 Downloads에 그대로 둠 (2026.07.21)
CMU_DIR    = "/mnt/c/Users/USER/Downloads/CMU"
OUT_PATH   = os.path.join(REPO_ROOT, "humanoidverse", "data", "motions", "cmu_walk_h1_full19_clips.npz")

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


def extract_upper_body_h1(poses: np.ndarray) -> np.ndarray:
    """
    SMPL+H poses (T, 156) → H1 상체 9 DOF (T, 9) [torso, L_shoulder x3, L_elbow, R_shoulder x3, R_elbow]

    H1 URDF 관절 회전축 (h1.urdf 확인):
        torso_joint:              z축
        {L,R}_shoulder_pitch:      y축
        {L,R}_shoulder_roll:       x축
        {L,R}_shoulder_yaw:        z축
        {L,R}_elbow:               y축

    SMPL → H1 매핑 (하체와 동일하게 axis-angle → euler_xyz 분해 후 매핑):
        torso  ← spine1+spine2+spine3 z축 합 (SMPL은 허리를 3관절로 나누지만 H1은 1관절이라 누적)
        shoulder_pitch/roll/yaw ← shoulder euler의 y/x/z 성분
        elbow  ← elbow euler의 y축 성분 (굽힘 방향 부호는 다리 렌더링 검증과 같은 방식으로 확인 필요)

    주의: 이 매핑은 1차 추정치이며, 실제 로봇에 입혀서 시각적으로 검증 후
    부호/성분을 조정했다 (check_mocap.py 계열 스크립트 참고).

    [2026.07.22 버그 수정] shoulder_yaw가 걷는 내내 거의 상수(-60~-70도 근처)로
    고정되어 있고 실제 걷기 스윙(pitch)에 이 큰 상수가 섞여서 팔이 정면-후면이
    아니라 대각선으로 움직이는 문제를 발견함 (학습된 정책 영상에서 육안으로 확인).
    SMPL 어깨 좌표계와 H1 어깨 좌표계 사이의 기준 프레임 차이(원본 프레임워크의
    smpl_pose_modifier: L_Shoulder=[0,0,-pi/2], R_Shoulder=[0,0,pi/2] 참고)가
    보정 없이 그대로 섞여 들어간 것으로 추정됨.
    Euler XYZ 분해에서 z(yaw) 성분에 상수를 더하고 빼는 것은 x(roll)/y(pitch)에
    영향을 주지 않음을 별도로 검증했으므로(Rotation.from_euler('xyz',[0,0,d]) * R
    형태로 왼쪽에 곱하면 yaw만 d만큼 정확히 이동), 클립별로 yaw의 평균(상수 오프셋)을
    빼서 걷기로 인한 실제 진동(oscillation)만 남긴다.
    """
    T = poses.shape[0]
    joints = np.zeros((T, 9), dtype=np.float32)

    # ── 토르소 (SMPL spine1+spine2+spine3의 z축 회전을 누적) ──────────────
    spine1 = axis_angle_to_euler_xyz(poses[:, 3*3:3*3+3])
    spine2 = axis_angle_to_euler_xyz(poses[:, 6*3:6*3+3])
    spine3 = axis_angle_to_euler_xyz(poses[:, 9*3:9*3+3])
    joints[:, 0] = spine1[:, 2] + spine2[:, 2] + spine3[:, 2]  # torso ← z축 누적

    # ── 왼쪽 팔 ────────────────────────────────────
    lshoulder = axis_angle_to_euler_xyz(poses[:, 16*3:16*3+3])
    lelbow    = axis_angle_to_euler_xyz(poses[:, 18*3:18*3+3])

    joints[:, 1] = lshoulder[:, 1]   # left_shoulder_pitch ← y축
    joints[:, 2] = lshoulder[:, 0]   # left_shoulder_roll  ← x축
    joints[:, 3] = lshoulder[:, 2] - lshoulder[:, 2].mean()  # left_shoulder_yaw ← z축, 클립별 상수 오프셋 제거
    joints[:, 4] = -lelbow[:, 1]     # left_elbow          ← y축 (부호는 검증 필요)

    # ── 오른쪽 팔 (좌우 대칭이라 roll/yaw 부호 반전) ──────────────
    rshoulder = axis_angle_to_euler_xyz(poses[:, 17*3:17*3+3])
    relbow    = axis_angle_to_euler_xyz(poses[:, 19*3:19*3+3])

    joints[:, 5] = rshoulder[:, 1]    # right_shoulder_pitch ← y축
    joints[:, 6] = -rshoulder[:, 0]   # right_shoulder_roll  ← x축 (좌우 대칭 부호 반전)
    _ryaw_centered = rshoulder[:, 2] - rshoulder[:, 2].mean()  # 클립별 상수 오프셋 제거 후 부호 반전
    joints[:, 7] = -_ryaw_centered    # right_shoulder_yaw   ← z축 (좌우 대칭 부호 반전)
    joints[:, 8] = -relbow[:, 1]      # right_elbow          ← y축

    return joints


def extract_full_body_h1(poses: np.ndarray) -> np.ndarray:
    """하체 10 DOF + 상체 9 DOF → 19 DOF (h1.yaml dof_names 순서와 동일)"""
    lower = extract_lower_body_h1(poses)   # (T, 10)
    upper = extract_upper_body_h1(poses)   # (T, 9)
    # h1.yaml dof_names 순서: [하체 10개] + [torso, L팔 4개, R팔 4개]
    return np.concatenate([lower, upper], axis=1)  # (T, 19)


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

                # H1 전신(하체10+상체9) 19 DOF 추출
                full_body = extract_full_body_h1(poses)  # (T, 19)

                # TARGET_FPS 로 리샘플링
                full_body_resampled = resample(full_body, fps, TARGET_FPS)

                # 클립 자체 내부에도 노이즈성 급점프가 있을 수 있어 한 번 더 체크
                diffs = np.linalg.norm(np.diff(full_body_resampled, axis=0), axis=1)
                if len(diffs) > 0 and diffs.max() > 1.0:
                    print(f"  ⚠ {subj}/{fname}: 클립 내부 불연속 감지(max_diff={diffs.max():.2f}), 스킵")
                    continue

                all_clips.append(full_body_resampled)
                accepted += 1
                print(f"  ✓ {subj}/{fname}  frames={full_body_resampled.shape[0]}")

            except Exception as e:
                print(f"  ✗ {subj}/{fname}: {e}")

    print(f"\n전체 {total_files}개 파일 중 {accepted}개 걷기 시퀀스 채택")

    if len(all_clips) == 0:
        print("채택된 시퀀스가 없습니다. KNOWN_WALKING_SUBJECTS 또는 MIN_AVG_SPEED_MPS 기준을 조정하세요.")
        return

    # ── 클립을 이어붙이지 않고 각각 분리된 상태로 저장 (2026.07.20 교훈 반영) ──
    # 서로 다른 클립을 이어붙이면 경계에서 관절이 순간이동하는 문제가 있었음
    MIN_CLIP_LEN = 50  # 최소 1초(50Hz) 이상만 채택
    clips = [c for c in all_clips if len(c) >= MIN_CLIP_LEN]
    lengths = [len(c) for c in clips]
    print(f"클립 개수: {len(clips)} (최소 길이 필터 후), 길이 범위: {min(lengths)}~{max(lengths)}프레임")

    max_len = max(lengths)
    padded = np.zeros((len(clips), max_len, 19), dtype=np.float32)
    for i, c in enumerate(clips):
        padded[i, :len(c)] = c
    lengths_arr = np.array(lengths, dtype=np.int64)

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    np.savez(OUT_PATH, clips=padded, lengths=lengths_arr)
    print(f"저장 완료: {OUT_PATH}  (clips shape={padded.shape})")

    # 간단한 통계 출력 (패딩 제외하고 실제 프레임만)
    all_frames = np.concatenate(clips, axis=0)
    print("\n관절별 평균/표준편차 (라디안):")
    labels = ['L_hip_yaw','L_hip_roll','L_hip_pitch','L_knee','L_ankle',
              'R_hip_yaw','R_hip_roll','R_hip_pitch','R_knee','R_ankle',
              'torso',
              'L_shoulder_pitch','L_shoulder_roll','L_shoulder_yaw','L_elbow',
              'R_shoulder_pitch','R_shoulder_roll','R_shoulder_yaw','R_elbow']
    for i, label in enumerate(labels):
        print(f"  {label:18s}: mean={all_frames[:,i].mean():.3f}, std={all_frames[:,i].std():.3f}, "
              f"range=[{all_frames[:,i].min():.3f}, {all_frames[:,i].max():.3f}]")


if __name__ == '__main__':
    main()
