"""
IK 파이프라인 Round-trip 검증
-------------------------------
LAFAN1 CSV 각도 → H1 FK → 3D 관절 위치 → 우리 IK → 복원 각도
복원 각도가 원래 CSV 각도와 일치하면 IK 수식이 정확하다는 증명

H1 관절 체인 (왼쪽 다리):
  pelvis
  └─ (0, +0.0875, -0.1742)  Rz(hip_yaw)
     └─ (+0.039468, 0, 0)   Rx(hip_roll)
        └─ (0, +0.11536, 0)  Ry(hip_pitch)    ← p_hip (IK 기준점)
           └─ (0, 0, -0.4)  Ry(knee)           ← p_knee
              └─ (0, 0, -0.4)                  ← p_ankle
"""

import os
import sys
_d = os.path.dirname(os.path.abspath(__file__))
if _d in sys.path:
    sys.path.remove(_d)

import numpy as np

# ─── H1 URDF 관절 오프셋 ──────────────────────────────────────────
#  (pelvis 로컬 프레임 기준, 단위: 미터)
LEFT_HIP_YAW_OFFSET   = np.array([ 0.0,      +0.0875,  -0.1742])
LEFT_HIP_ROLL_OFFSET  = np.array([+0.039468,  0.0,       0.0   ])
LEFT_HIP_PITCH_OFFSET = np.array([ 0.0,      +0.11536,   0.0   ])
THIGH_VEC             = np.array([ 0.0,       0.0,      -0.4   ])
SHIN_VEC              = np.array([ 0.0,       0.0,      -0.4   ])

RIGHT_HIP_YAW_OFFSET   = np.array([ 0.0,     -0.0875,  -0.1742])
RIGHT_HIP_ROLL_OFFSET  = np.array([+0.039468,  0.0,      0.0   ])
RIGHT_HIP_PITCH_OFFSET = np.array([ 0.0,     -0.11536,   0.0   ])


def Rz(a): s, c = np.sin(a), np.cos(a); return np.array([[c,-s,0],[s,c,0],[0,0,1]])
def Rx(a): s, c = np.sin(a), np.cos(a); return np.array([[1,0,0],[0,c,-s],[0,s,c]])
def Ry(a): s, c = np.sin(a), np.cos(a); return np.array([[c,0,s],[0,1,0],[-s,0,c]])


def h1_leg_fk(yaw, roll, pitch, knee, is_left=True):
    """
    H1 다리 FK: 관절 각도 → 3D 위치 (pelvis 로컬 프레임)

    Returns:
        p_hip:   hip_pitch 관절 위치 (IK에서 고관절 기준점)
        p_knee:  무릎 관절 위치
        p_ankle: 발목 관절 위치
    """
    if is_left:
        o_yaw   = LEFT_HIP_YAW_OFFSET
        o_roll  = LEFT_HIP_ROLL_OFFSET
        o_pitch = LEFT_HIP_PITCH_OFFSET
    else:
        o_yaw   = RIGHT_HIP_YAW_OFFSET
        o_roll  = RIGHT_HIP_ROLL_OFFSET
        o_pitch = RIGHT_HIP_PITCH_OFFSET

    # 누적 변환 (위치 + 회전)
    # T: pelvis → hip_yaw_joint 평행이동 + yaw 회전
    R0 = Rz(yaw)
    p0 = o_yaw

    # T: → hip_roll_joint
    R1 = R0 @ Rx(roll)
    p1 = p0 + R0 @ o_roll

    # T: → hip_pitch_joint  ← 이게 IK의 p_hip
    R2 = R1 @ Ry(pitch)
    p_hip = p1 + R1 @ o_pitch

    # T: → knee_joint  (허벅지 = thigh_vec)
    R3 = R2 @ Ry(knee)
    p_knee = p_hip + R2 @ THIGH_VEC

    # T: → ankle_joint  (정강이 = shin_vec)
    p_ankle = p_knee + R3 @ SHIN_VEC

    return p_hip, p_knee, p_ankle, R2   # R2 = hip_pitch까지의 누적 회전


def ik_from_positions(p_hip, p_knee, p_ankle, R_pelvis):
    """
    해석적 IK: ik_retarget.py 의 batch_leg_ik 와 동일한 수식 (단일 프레임)
    """
    H1_THIGH = H1_SHIN = 0.4

    thigh_dir = (p_knee - p_hip); thigh_dir /= max(np.linalg.norm(thigh_dir), 1e-8)
    shin_dir  = (p_ankle - p_knee); shin_dir  /= max(np.linalg.norm(shin_dir), 1e-8)

    p_knee_h1  = p_hip + thigh_dir * H1_THIGH
    p_ankle_h1 = p_knee_h1 + shin_dir * H1_SHIN

    # 무릎 (law of cosines)
    d = np.linalg.norm(p_ankle_h1 - p_hip)
    d = np.clip(d, 1e-6, H1_THIGH + H1_SHIN - 1e-6)
    cos_val = (H1_THIGH**2 + H1_SHIN**2 - d**2) / (2 * H1_THIGH * H1_SHIN)
    knee = np.pi - np.arccos(np.clip(cos_val, -1, 1))

    # 고관절 pitch/roll
    # thigh_local[0] = -sin(pitch)  (Ry 정의 기준)
    thigh_local = R_pelvis.T @ thigh_dir
    hip_pitch = -np.arcsin(np.clip(thigh_local[0], -1, 1))
    cos_pitch = np.cos(hip_pitch)
    sin_roll  = thigh_local[1] / max(abs(cos_pitch), 0.05)
    hip_roll  = np.arcsin(np.clip(sin_roll, -1, 1))

    # 발목: hip_pitch + knee + ankle = 0 (발바닥 수평 유지)
    ankle = -(hip_pitch + knee)

    return 0.0, hip_roll, hip_pitch, knee, ankle   # (yaw=0, roll, pitch, knee, ankle)


def main():
    csv_path = "/mnt/c/Users/jayelle/lafan1_hf/h1/walk1_subject1.csv"
    data = np.loadtxt(csv_path, delimiter=',', dtype=np.float32)
    # [0:7] root_joint, [7:26] H1 19-DOF
    angles = data[:, 7:]   # (T, 19)

    # LAFAN1 CSV H1 관절 순서 (motion_loader.py 기준):
    # 0: L_hip_yaw   1: L_hip_roll   2: L_hip_pitch
    # 3: L_knee      4: L_ankle
    # 5: R_hip_yaw   6: R_hip_roll   7: R_hip_pitch
    # 8: R_knee      9: R_ankle
    # 10: torso
    # 11-14: L_shoulder x3 + L_elbow
    # 15-18: R_shoulder x3 + R_elbow

    T = len(angles)
    print(f"CSV: {T} 프레임 (walk1_subject1)")
    print()

    errors_knee   = []
    errors_pitch  = []
    errors_roll   = []
    errors_ankle  = []

    for t in range(T):
        a = angles[t]
        yaw_l   = a[0]; roll_l  = a[1]; pitch_l = a[2]
        knee_l  = a[3]; ankle_l = a[4]

        # ─── 1. H1 FK: 각도 → 3D 위치 ───────────────────────────
        p_hip, p_knee, p_ankle, R_hip = h1_leg_fk(
            yaw_l, roll_l, pitch_l, knee_l, is_left=True)

        # R_pelvis = 단위 행렬 (pelvis 로컬 프레임 기준이므로)
        R_pelvis = np.eye(3)

        # ─── 2. IK: 3D 위치 → 복원 각도 ─────────────────────────
        yaw_r, roll_r, pitch_r, knee_r, ankle_r = ik_from_positions(
            p_hip, p_knee, p_ankle, R_pelvis)

        # ─── 3. 오차 계산 ────────────────────────────────────────
        errors_knee.append(abs(knee_r - knee_l))
        errors_pitch.append(abs(pitch_r - pitch_l))
        errors_roll.append(abs(roll_r - roll_l))
        errors_ankle.append(abs(ankle_r - ankle_l))

    print("=" * 55)
    print(f"{'관절':<15} {'평균오차(rad)':>14} {'최대오차(rad)':>14}")
    print("=" * 55)
    for name, errs in [
        ('L_knee',      errors_knee),
        ('L_hip_pitch', errors_pitch),
        ('L_hip_roll',  errors_roll),
        ('L_ankle',     errors_ankle),
    ]:
        print(f"  {name:<13}  {np.mean(errs):>12.5f}   {np.max(errs):>12.5f}")
    print("=" * 55)

    print()
    print("[해석]")
    if np.mean(errors_knee) < 0.001:
        print("  무릎 ✓  : law of cosines 수식 완벽히 일치")
    else:
        print(f"  무릎 오차: {np.mean(errors_knee):.4f} rad")

    if np.mean(errors_pitch) < 0.01:
        print("  hip_pitch ✓: 수식 정확")
    else:
        print(f"  hip_pitch 오차: {np.mean(errors_pitch):.4f} rad (약간의 근사 허용 범위)")

    if np.mean(errors_roll) < 0.02:
        print("  hip_roll ✓: 수식 정확")
    else:
        print(f"  hip_roll 오차: {np.mean(errors_roll):.4f} rad")

    print()
    print("[첫 5 프레임 비교]")
    print(f"  {'프레임':>5}  {'CSV_knee':>9}  {'IK_knee':>9}  {'CSV_pitch':>10}  {'IK_pitch':>10}")
    for t in range(min(5, T)):
        a = angles[t]
        p_h, p_k, p_a, _ = h1_leg_fk(a[0], a[1], a[2], a[3])
        _, roll_r, pitch_r, knee_r, _ = ik_from_positions(p_h, p_k, p_a, np.eye(3))
        print(f"  {t:>5}  {a[3]:>9.4f}  {knee_r:>9.4f}  {a[2]:>10.4f}  {pitch_r:>10.4f}")


if __name__ == '__main__':
    main()
