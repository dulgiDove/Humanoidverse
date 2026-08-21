"""
[LAFAN1 BVH → H1 IK 파이프라인]
작성: 정상일

흐름:
  1. BVH 파서: HIERARCHY → 관절 트리, MOTION → 프레임별 오일러 각도
  2. FK: 오일러 → 회전 행렬 → 월드 3D 관절 위치 (cm)
  3. 좌표 변환: LAFAN1(Y-up, cm) → H1(Z-up, m)
  4. 해석적 IK: 3D 위치 → H1 19-DOF 관절 각도
  5. 리샘플링 + 클리핑 → .npz 저장

BVH 좌표계 (LAFAN1 기준):
  Y = 위 (up)
  Z = 왼쪽 (+) / 오른쪽 (-)
  X = 앞 (forward)   ← FK 후 실측으로 확인

H1 좌표계 (URDF 기준):
  X = 앞 (forward)
  Y = 왼쪽 (left)
  Z = 위 (up)

변환: H1_X = BVH_X, H1_Y = BVH_Z, H1_Z = BVH_Y
"""

import os, sys, glob
_d = os.path.dirname(os.path.abspath(__file__))
if _d in sys.path:
    sys.path.remove(_d)

import numpy as np
from scipy.spatial.transform import Rotation

# ─── 경로 ────────────────────────────────────────────────────────
REPO_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
BVH_DIR   = "/mnt/c/Users/jayelle/lafan1_bvh"
OUT_PATH  = os.path.join(REPO_ROOT, "humanoidverse", "data", "motions",
                         "lafan1_walk_h1_ik.npz")

WALK_PREFIXES = ('walk1', 'walk2', 'walk3', 'walk4')
TARGET_FPS = 50
CLIP_LEN   = 200
CLIP_STEP  = 100
MIN_CLIP   = 50

# ─── H1 파라미터 (URDF에서 추출) ─────────────────────────────────
H1_THIGH = 0.400
H1_SHIN  = 0.400
H1_UA    = 0.198

LIMITS = np.array([
    [-0.43,  0.43],  # 0  L_hip_yaw
    [-0.43,  0.43],  # 1  L_hip_roll
    [-1.57,  1.57],  # 2  L_hip_pitch
    [-0.26,  2.05],  # 3  L_knee
    [-0.87,  0.52],  # 4  L_ankle
    [-0.43,  0.43],  # 5  R_hip_yaw
    [-0.43,  0.43],  # 6  R_hip_roll
    [-1.57,  1.57],  # 7  R_hip_pitch
    [-0.26,  2.05],  # 8  R_knee
    [-0.87,  0.52],  # 9  R_ankle
    [-2.35,  2.35],  # 10 torso
    [-2.87,  2.87],  # 11 L_sh_pitch
    [-0.34,  3.11],  # 12 L_sh_roll
    [-1.30,  4.45],  # 13 L_sh_yaw
    [-1.25,  2.61],  # 14 L_elbow
    [-2.87,  2.87],  # 15 R_sh_pitch
    [-3.11,  0.34],  # 16 R_sh_roll
    [-4.45,  1.30],  # 17 R_sh_yaw
    [-1.25,  2.61],  # 18 R_elbow
], dtype=np.float32)


# ════════════════════════════════════════════════════════════════
# Step 1: BVH 파서
# ════════════════════════════════════════════════════════════════

def parse_bvh(filepath: str):
    """
    BVH 파일 파싱.

    BVH 포맷:
      HIERARCHY 섹션: 관절 이름, 부모-자식 관계, T-포즈 오프셋, 채널 순서
      MOTION 섹션:    프레임 수, 프레임 시간, 프레임별 채널 값 (도 단위)

    Returns:
        joints: list of dict
            {name, parent_idx, offset(3,), channels(list of str)}
        frames: np.ndarray (T, total_channels)  오일러 각도 (도) + root 위치 (cm)
        fps:    float
    """
    with open(filepath, 'r') as f:
        lines = f.readlines()

    joints = []          # 파싱된 관절 목록
    joint_stack = []     # 현재 열린 관절 스택 (계층 추적용)
    channel_count = 0    # 전체 채널 수 누적
    in_motion = False
    fps = 30.0
    frame_lines = []

    # BVH 파서 핵심 문제:
    # 'End Site { ... }' 블록의 }가 관절 스택을 잘못 pop하는 버그 방지
    # → end_site_depth:  0 = 밖, -1 = '{' 대기 중, >0 = 블록 안 깊이
    end_site_depth = 0
    pending_joint_idx = -1   # ROOT/JOINT 만나면 여기 저장, { 만나면 stack에 push

    i = 0
    while i < len(lines):
        line = lines[i].strip()
        tokens = line.split()

        if not tokens:
            i += 1
            continue

        if tokens[0] in ('ROOT', 'JOINT'):
            name = tokens[1]
            parent_idx = joint_stack[-1] if joint_stack else -1
            joints.append({
                'name':       name,
                'parent_idx': parent_idx,
                'offset':     np.zeros(3),
                'channels':   [],
                'chan_start':  channel_count,
            })
            pending_joint_idx = len(joints) - 1

        elif tokens[0] == 'End':  # 'End Site'
            end_site_depth = -1          # 다음 '{' 을 End Site 오프닝으로 처리

        elif tokens[0] == '{':
            if end_site_depth == -1:
                end_site_depth = 1       # End Site 블록 진입
            elif end_site_depth > 0:
                end_site_depth += 1      # End Site 내부 중첩 (드묾)
            elif pending_joint_idx >= 0:
                joint_stack.append(pending_joint_idx)  # 관절 블록 시작
                pending_joint_idx = -1

        elif tokens[0] == '}':
            if end_site_depth > 0:
                end_site_depth -= 1      # End Site 블록 닫힘 (0이 되면 완전히 빠져나옴)
            elif joint_stack:
                joint_stack.pop()        # 관절 블록 닫힘

        elif tokens[0] == 'OFFSET' and joint_stack and end_site_depth == 0:
            joints[joint_stack[-1]]['offset'] = np.array(
                [float(tokens[1]), float(tokens[2]), float(tokens[3])]
            )

        elif tokens[0] == 'CHANNELS' and joint_stack:
            n_chan = int(tokens[1])
            ch_names = tokens[2:2 + n_chan]
            joints[joint_stack[-1]]['channels'] = ch_names
            channel_count += n_chan

        elif tokens[0] == 'MOTION':
            in_motion = True

        elif in_motion and tokens[0] == 'Frame' and len(tokens) > 1 and tokens[1] == 'Time:':
            fps = 1.0 / float(tokens[2])

        elif in_motion and len(tokens) > 1:
            try:
                frame_lines.append([float(v) for v in tokens])
            except ValueError:
                pass

        i += 1

    frames = np.array(frame_lines, dtype=np.float32)  # (T, total_channels)
    return joints, frames, fps


# ════════════════════════════════════════════════════════════════
# Step 2: BVH FK
# ════════════════════════════════════════════════════════════════

def euler_to_rotmat(angles_deg: np.ndarray, order: str) -> np.ndarray:
    """
    오일러 각도 (도) → 3×3 회전 행렬
    order: 채널 순서 예) 'ZYX', 'XYZ'
    scipy는 intrinsic, BVH는 extrinsic → 순서 반전해서 intrinsic으로 변환
    """
    # BVH 오일러는 extrinsic (global axis) 회전
    # scipy from_euler는 기본이 intrinsic
    # extrinsic ZYX = intrinsic XYZ (반전)
    return Rotation.from_euler(order[::-1].lower(),
                               angles_deg[::-1],
                               degrees=True).as_matrix()


def bvh_fk(joints: list, frames: np.ndarray) -> np.ndarray:
    """
    BVH FK: 프레임별 오일러 각도 → 월드 3D 관절 위치 (cm, BVH 좌표계)

    각 관절의 변환:
      T_world[joint] = T_world[parent] @ T_local[joint]
      T_local = translate(offset) @ rotate(channels)

    Returns:
        positions: (T, N_joints, 3)  월드 좌표 (cm)
    """
    T = len(frames)
    N = len(joints)
    positions = np.zeros((T, N, 3), dtype=np.float32)

    # 월드 변환 행렬 (T, N, 4, 4)
    G = np.tile(np.eye(4, dtype=np.float32), (T, N, 1, 1))

    for ji, j in enumerate(joints):
        offset   = j['offset']           # (3,) T-포즈 오프셋 (cm)
        channels = j['channels']         # 채널 이름 목록
        c_start  = j['chan_start']        # 이 관절의 채널 시작 인덱스

        # ── 로컬 회전 행렬 계산 ──
        rot_channels = [c for c in channels if 'rotation' in c.lower()]
        pos_channels = [c for c in channels if 'position' in c.lower()]
        rot_order    = ''.join(c[0] for c in rot_channels)  # e.g. 'ZYX'

        # 채널 값 추출
        chan_vals = frames[:, c_start:c_start + len(channels)]  # (T, n_chan)

        # 위치 채널 (root만 있음)
        root_pos = np.zeros((T, 3), dtype=np.float32)
        for k, c in enumerate(channels):
            if 'Xposition' in c: root_pos[:, 0] = chan_vals[:, k]
            elif 'Yposition' in c: root_pos[:, 1] = chan_vals[:, k]
            elif 'Zposition' in c: root_pos[:, 2] = chan_vals[:, k]

        # 회전 채널 인덱스 추출
        rot_idx = [k for k, c in enumerate(channels) if 'rotation' in c.lower()]
        angles  = chan_vals[:, rot_idx]  # (T, 3) 도 단위

        # 벡터화: 모든 프레임의 회전 행렬 한번에 계산
        # scipy 배치 처리
        rotmats = Rotation.from_euler(
            rot_order[::-1].lower(),
            angles[:, ::-1],
            degrees=True
        ).as_matrix().astype(np.float32)  # (T, 3, 3)

        # 로컬 변환 행렬 구성
        local = np.tile(np.eye(4, dtype=np.float32), (T, 1, 1))
        local[:, :3, :3] = rotmats
        local[:, :3,  3] = offset  # T-포즈 오프셋

        # root의 경우 위치도 추가
        if pos_channels:
            local[:, :3, 3] = root_pos  # 오프셋 대신 실제 위치 사용
            # 사실 root는 오프셋 무시하고 채널 위치를 씀

        # 월드 변환 = 부모 월드 × 로컬
        parent_idx = j['parent_idx']
        if parent_idx == -1:
            G[:, ji] = local
        else:
            G[:, ji] = G[:, parent_idx] @ local

        positions[:, ji] = G[:, ji, :3, 3]

    return positions  # (T, N, 3) cm, BVH 좌표계


# ════════════════════════════════════════════════════════════════
# Step 3: 좌표 변환 + IK (ik_retarget.py 와 동일한 수식)
# ════════════════════════════════════════════════════════════════

# LAFAN1 BVH → H1 좌표 변환
# BVH: X=forward, Y=up, Z=left
# H1:  X=forward, Y=left, Z=up
# 변환: H1_X = BVH_X, H1_Y = BVH_Z, H1_Z = BVH_Y
BVH2H1 = np.array([
    [1, 0, 0],   # H1_X = BVH_X
    [0, 0, 1],   # H1_Y = BVH_Z
    [0, 1, 0],   # H1_Z = BVH_Y
], dtype=np.float64)


def batch_leg_ik(p_hip, p_knee, p_ankle, R_pelvis, is_left):
    """해석적 IK - ik_retarget.py와 동일한 수식 (검증 완료)"""
    T = len(p_hip)

    thigh_vec = p_knee - p_hip
    shin_vec  = p_ankle - p_knee
    thigh_len = np.linalg.norm(thigh_vec, axis=1, keepdims=True).clip(1e-6)
    shin_len  = np.linalg.norm(shin_vec,  axis=1, keepdims=True).clip(1e-6)
    thigh_dir = thigh_vec / thigh_len
    shin_dir  = shin_vec  / shin_len

    p_knee_h1  = p_hip   + thigh_dir * H1_THIGH
    p_ankle_h1 = p_knee_h1 + shin_dir * H1_SHIN

    # 무릎: law of cosines
    d = np.linalg.norm(p_ankle_h1 - p_hip, axis=1).clip(
        abs(H1_THIGH - H1_SHIN) + 1e-6, H1_THIGH + H1_SHIN - 1e-6)
    cos_val = (H1_THIGH**2 + H1_SHIN**2 - d**2) / (2 * H1_THIGH * H1_SHIN)
    knee = np.pi - np.arccos(np.clip(cos_val, -1.0, 1.0))

    # 고관절: 허벅지 방향 분해
    # thigh_local[0] = -sin(pitch)  →  pitch = -arcsin(thigh_local[0])
    thigh_local = np.einsum('tij,tj->ti', R_pelvis.transpose(0, 2, 1), thigh_dir)
    hip_pitch = -np.arcsin(np.clip(thigh_local[:, 0], -1.0, 1.0))
    cos_pitch = np.cos(hip_pitch)
    sin_roll  = np.where(np.abs(cos_pitch) > 0.05,
                         thigh_local[:, 1] / np.maximum(np.abs(cos_pitch), 0.05),
                         thigh_local[:, 1])
    hip_roll = np.arcsin(np.clip(sin_roll, -1.0, 1.0))
    hip_yaw  = np.zeros(T, dtype=np.float32)

    # 발목: 발바닥 수평 유지
    ankle = -(hip_pitch + knee)

    return np.stack([hip_yaw, hip_roll, hip_pitch, knee, ankle], axis=1).astype(np.float32)


def batch_arm_ik(p_shoulder, p_elbow, p_wrist, R_torso, is_left):
    """해석적 IK - ik_retarget.py와 동일한 수식"""
    T = len(p_shoulder)

    ua_vec = p_elbow - p_shoulder
    fa_vec = p_wrist - p_elbow
    ua_len = np.linalg.norm(ua_vec, axis=1, keepdims=True).clip(1e-6)
    fa_len = np.linalg.norm(fa_vec, axis=1, keepdims=True).clip(1e-6)
    ua_dir = ua_vec / ua_len
    fa_dir = fa_vec / fa_len

    cos_elbow = np.clip(np.einsum('ti,ti->t', ua_dir, fa_dir), -1.0, 1.0)
    elbow = np.arccos(cos_elbow)

    ua_local = np.einsum('tij,tj->ti', R_torso.transpose(0, 2, 1), ua_dir)
    lat = ua_local[:, 1] if is_left else -ua_local[:, 1]
    fwd = ua_local[:, 0]
    up  = ua_local[:, 2]
    lat_safe = np.maximum(lat, 0.05)

    sh_pitch    = np.arctan2(fwd, lat_safe)
    sh_roll_raw = np.arctan2(up,  lat_safe)
    sh_roll     = sh_roll_raw if is_left else -sh_roll_raw
    sh_yaw      = np.zeros(T, dtype=np.float32)

    return np.stack([sh_pitch, sh_roll, sh_yaw, elbow], axis=1).astype(np.float32)


# ════════════════════════════════════════════════════════════════
# Step 4: 파일 처리
# ════════════════════════════════════════════════════════════════

def get_joint_idx(joints, name):
    """관절 이름으로 인덱스 반환"""
    for i, j in enumerate(joints):
        if j['name'] == name:
            return i
    raise ValueError(f"관절 '{name}' 없음. 목록: {[j['name'] for j in joints]}")


def process_bvh_file(filepath: str):
    """
    BVH 파일 하나 → H1 19-DOF 관절 각도 (T, 19), 50Hz
    """
    # ── 1. 파싱 ──
    joints, frames, src_fps = parse_bvh(filepath)
    T = len(frames)

    # ── 2. FK → 월드 3D 위치 (cm, BVH 좌표계) ──
    pos_bvh = bvh_fk(joints, frames)  # (T, N, 3)

    # ── 3. cm → m 변환 + BVH → H1 좌표계 ──
    pos_m = pos_bvh / 100.0  # cm → m
    pos_h1 = (BVH2H1 @ pos_m.reshape(-1, 3).T).T.reshape(T, len(joints), 3)

    # 관절 인덱스
    idx = {name: get_joint_idx(joints, name) for name in [
        'Hips',
        'LeftUpLeg', 'LeftLeg', 'LeftFoot',
        'RightUpLeg', 'RightLeg', 'RightFoot',
        'Spine',
        'LeftArm', 'LeftForeArm', 'LeftHand',
        'RightArm', 'RightForeArm', 'RightHand',
    ]}

    # ── 4. 골반 회전 행렬 ──
    # LAFAN1 BVH root는 ~90° 바인드포즈 오프셋이 내포되어 있어
    # 그대로 R_pelvis로 쓰면 hip_pitch가 크게 틀림.
    # IK에서는 '기립 자세 기준' 상대 각도가 필요하므로 단위 행렬 사용.
    R_pelvis = np.tile(np.eye(3, dtype=np.float32), (T, 1, 1))
    R_torso  = R_pelvis

    # ── 5. IK ──
    output = np.zeros((T, 19), dtype=np.float32)

    # 왼쪽 다리
    output[:, 0:5] = batch_leg_ik(
        pos_h1[:, idx['LeftUpLeg']],
        pos_h1[:, idx['LeftLeg']],
        pos_h1[:, idx['LeftFoot']],
        R_pelvis, is_left=True
    )

    # 오른쪽 다리
    output[:, 5:10] = batch_leg_ik(
        pos_h1[:, idx['RightUpLeg']],
        pos_h1[:, idx['RightLeg']],
        pos_h1[:, idx['RightFoot']],
        R_pelvis, is_left=False
    )

    # torso: 골반 대비 yaw 변화량 → 0으로 단순화 (걷기에서 작음)
    output[:, 10] = 0.0

    # 왼팔
    output[:, 11:15] = batch_arm_ik(
        pos_h1[:, idx['LeftArm']],
        pos_h1[:, idx['LeftForeArm']],
        pos_h1[:, idx['LeftHand']],
        R_torso, is_left=True
    )

    # 오른팔
    output[:, 15:19] = batch_arm_ik(
        pos_h1[:, idx['RightArm']],
        pos_h1[:, idx['RightForeArm']],
        pos_h1[:, idx['RightHand']],
        R_torso, is_left=False
    )

    # 관절 범위 클램핑
    for j in range(19):
        output[:, j] = np.clip(output[:, j], LIMITS[j, 0], LIMITS[j, 1])

    # 50Hz 리샘플링
    output = resample(output, src_fps, TARGET_FPS)
    return output


# ─── 보조 함수 ────────────────────────────────────────────────────

def resample(data, src_fps, tgt_fps):
    T = data.shape[0]
    src_t = np.linspace(0, T / src_fps, T)
    tgt_T = max(1, int(round(T * tgt_fps / src_fps)))
    tgt_t = np.linspace(0, T / src_fps, tgt_T)
    out = np.zeros((tgt_T, data.shape[1]), dtype=np.float32)
    for d in range(data.shape[1]):
        out[:, d] = np.interp(tgt_t, src_t, data[:, d])
    return out


def split_clips(data):
    clips, T, start = [], len(data), 0
    while start + CLIP_LEN <= T:
        clips.append(data[start:start + CLIP_LEN].copy())
        start += CLIP_STEP
    if start < T and (T - start) >= MIN_CLIP:
        clips.append(data[start:T].copy())
    return clips


# ════════════════════════════════════════════════════════════════
# 메인
# ════════════════════════════════════════════════════════════════

def main():
    bvh_files = sorted([
        f for f in glob.glob(os.path.join(BVH_DIR, '*.bvh'))
        if os.path.basename(f).startswith(WALK_PREFIXES)
    ])
    print(f"walk BVH 파일 {len(bvh_files)}개 처리 중...")

    all_clips = []
    for fpath in bvh_files:
        fname = os.path.basename(fpath)
        try:
            data   = process_bvh_file(fpath)
            clips  = split_clips(data)
            print(f"  {fname}: {len(data)}f@50Hz → {len(clips)}클립")
            all_clips.extend(clips)
        except Exception as e:
            print(f"  ✗ {fname}: {e}")

    if not all_clips:
        print("오류: 처리된 클립 없음")
        return

    lengths = [len(c) for c in all_clips]
    max_len = max(lengths)
    padded  = np.zeros((len(all_clips), max_len, 19), dtype=np.float32)
    for i, c in enumerate(all_clips):
        padded[i, :len(c)] = c

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    np.savez(OUT_PATH, clips=padded, lengths=np.array(lengths, dtype=np.int64))

    print(f"\n총 {len(all_clips)}클립 저장: {OUT_PATH}")
    print(f"shape: {padded.shape}")

    # 관절별 통계 출력
    all_frames = np.concatenate(all_clips)
    labels = [
        'L_hip_yaw  ', 'L_hip_roll ', 'L_hip_pitch', 'L_knee     ', 'L_ankle    ',
        'R_hip_yaw  ', 'R_hip_roll ', 'R_hip_pitch', 'R_knee     ', 'R_ankle    ',
        'torso      ',
        'Lsh_pitch  ', 'Lsh_roll   ', 'Lsh_yaw    ', 'L_elbow    ',
        'Rsh_pitch  ', 'Rsh_roll   ', 'Rsh_yaw    ', 'R_elbow    ',
    ]
    print("\n[관절별 범위] (라디안)")
    print(f"  {'관절':13s}  {'min':>7s}  {'max':>7s}  {'mean':>7s}")
    for i, label in enumerate(labels):
        mn, mx, avg = all_frames[:, i].min(), all_frames[:, i].max(), all_frames[:, i].mean()
        print(f"  {label}: [{mn:+.3f}, {mx:+.3f}]  avg={avg:+.3f}")

    print(f"\n참조: LAFAN1 CSV 무릎 범위 = [+0.063, +2.050] rad")
    print(f"      이번 BVH IK 무릎 범위 = [{all_frames[:,3].min():+.3f}, {all_frames[:,3].max():+.3f}] rad")


if __name__ == '__main__':
    main()
