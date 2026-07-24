# 전신(19-DOF) 확장 + Style Imitation 안정화 작업 내용

기존 하체 10-DOF + style imitation 보상을 이정민의 e2e-goal-reaching 브랜치(목표
도달 + 커리큘럼 + 장애물 회피) 위에 올려서 전신 19-DOF(상체 토르소+양팔 추가)로
확장하는 과정에서 발견/수정한 것들을 정리했다.

## 1. `humanoidverse/config/algo/ppo.yaml` — `entropy_coef: 0.01 -> 0.005`

10-DOF에서 19-DOF로 로봇을 바꾸자 학습이 발산하는 문제가 있었다. 원인은
`humanoidverse/agents/modules/ppo_modules.py`의 entropy 계산:

```python
@property
def entropy(self):
    return self.distribution.entropy().sum(dim=-1)   # 관절 수만큼 sum
```

관절 수가 늘면 이 값 자체가 커지고, `actor_loss = surrogate_loss - entropy_coef * entropy_loss`
에서 entropy 항이 상대적으로 더 강하게 작용해서 액션 노이즈(`Mean action noise std`)가
계속 커지기만 하고 수렴하지 않았다. `entropy_coef`를 관절 수 비율(10→19, 약 1.9배)에
맞춰 낮추자 정상적으로 수렴함.

## 2. `humanoidverse/config/rewards/loco/reward_h1_locomotion.yaml`

### `style_imitation_sigma: 0.5 -> 1.5`

`_reward_style_imitation`은 `exp(-오차제곱합 / sigma)` 형태라, 목표 거리가 멀어지고
정책이 빠르게/공격적으로 움직이기 시작하면 모캡과의 오차가 커져서 보상이 완전히
0으로 죽는 현상이 있었다(8m 근처에서 몇 시간째 `rew_style_imitation: 0.0000`).
sigma를 넓혀서 오차가 좀 크더라도 신호가 살아있게 함.

### `upperbody_joint_angle_freeze: -0.1` 추가

원본 프레임워크에는 있었지만(`locomotion.py`의 `_reward_upperbody_joint_angle_freeze`)
yaml에 등록이 안 되어 있어서 계산 자체가 안 되고 있었다. style_imitation 신호가
죽었을 때 상체(특히 팔)를 붙잡아줄 다른 보상이 하나도 없어서, 한쪽 팔이 위로
뻗은 채 고정되는 등 이상한 자세로 수렴하는 문제가 있었다. 이 보상을 추가해서
"모캡과 똑같이"는 아니어도 "기본 자세에서 크게 벗어나지 마라"는 최소한의
안전장치를 걸었다.

## 3. `humanoidverse/utils/motion_loader.py`

CMU mocap → H1 관절각 변환 스크립트(원본 4.4GB CMU 데이터는 레포에 없음,
`/mnt/c/Users/USER/Downloads/CMU`에서 전처리해서 `cmu_walk_h1_full19_clips.npz`만
생성). 상체 확장 시 새로 작성한 `extract_upper_body_h1`에서 어깨 yaw 버그 발견:

- 학습된 정책 영상을 보니 팔이 걷는 방향 기준 대각선(약 45도)으로 움직였음
- 실제 클립 데이터를 까보니 `L_shoulder_yaw`/`R_shoulder_yaw`가 걷는 내내 거의
  상수(-60~-70도 근처, 표준편차 0.03~0.05)로 고정되어 있었음 — 걷기로 인한
  진동(pitch)에 이 상수가 그대로 섞여서 대각선 스윙처럼 보인 것
- 원인 추정: SMPL 어깨 좌표계와 H1 어깨 좌표계 사이 기준 프레임 차이(원본
  프레임워크의 `smpl_pose_modifier: L_Shoulder=[0,0,-pi/2]` 참고) 미보정
- 수정: 클립별로 yaw 값의 평균(상수 오프셋)을 빼서 진동만 남김. Euler XYZ
  분해에서 z(yaw) 성분에 상수를 더하고 빼는 건 x(roll)/y(pitch)에 영향을
  주지 않음을 별도 검증함(`Rotation.from_euler('xyz',[0,0,d]) * R`로 확인)

원본 CMU raw 데이터를 이미 삭제한 상태라, "데이터 선별 자체가 걷기치고는 관절
가동범위(무릎 등)가 작다"는 별도 이슈는 발견만 하고 미해결 상태로 남겨둠 —
필요하면 원본을 재다운로드해서 재검증해야 함.

## 4. `humanoidverse/envs/locomotion/locomotion.py`

- `__init__`에서 `config.robot.motion.get(...)` 호출 시 h1(19-DOF) 로봇 config에
  `motion` 블록 자체가 없어서 나던 크래시 수정 (`h1.yaml`에 `motion: {}` 추가)
- `_reward_style_imitation`: `current_pose = self.simulator.dof_pos[:, :10]`로
  하드코딩되어 있던 걸 `ref_dof = ref_pose.shape[-1]` 기준으로 동적으로 비교하도록
  수정. 19-DOF 로봇에서 이 슬라이싱을 안 고쳤으면 상체는 아예 비교 대상에서
  빠져서 style 보상이 상체에 전혀 영향을 못 줄 뻔했음

## 검증 이력

- 1차: entropy_coef만 고치고 처음부터 재학습 → 8m 커리큘럼 완주, 하지만
  style_imitation이 8m 근처에서 다시 0으로 죽음, 영상에서 팔 고정/상체 틀어짐 확인
- 2차: sigma+freeze 보상 추가 후 체크포인트에서 재개 → style_imitation이
  8m에서도 0.9 이상 유지, 영상에서 팔 고정 문제는 해결됐으나 대각선 스윙 발견
- 3차: yaw 버그 수정 후 체크포인트에서 재개 → 24시간 완주, style_imitation
  0.95 유지, 영상에서 대각선 스윙도 개선됨
- 4차(진행중): 위 모든 수정을 반영한 상태로 처음부터(체크포인트 없이) 30시간
  재현성 검증 학습 중
