"""
HumanoidVerse + Genesis 설치 확인 테스트 스크립트
G1-12DoF 기준

실행 방법:
  conda activate hgen
  cd HumanoidVerse
  python test_installation.py
"""

import sys
import importlib

# ──────────────────────────────────────────────
# 색상 출력 헬퍼
# ──────────────────────────────────────────────
GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
CYAN   = "\033[96m"
RESET  = "\033[0m"

def ok(msg):   print(f"  {GREEN}[PASS]{RESET} {msg}")
def fail(msg): print(f"  {RED}[FAIL]{RESET} {msg}")
def info(msg): print(f"  {CYAN}[INFO]{RESET} {msg}")
def warn(msg): print(f"  {YELLOW}[WARN]{RESET} {msg}")

results = {"pass": 0, "fail": 0}

def check(name, fn):
    try:
        result = fn()
        ok(f"{name}" + (f" — {result}" if result else ""))
        results["pass"] += 1
        return True
    except Exception as e:
        fail(f"{name} → {e}")
        results["fail"] += 1
        return False


# ══════════════════════════════════════════════
# 1. Python 버전 확인
# ══════════════════════════════════════════════
print(f"\n{CYAN}{'='*50}{RESET}")
print(f"{CYAN}  Step 1. Python 버전 확인{RESET}")
print(f"{CYAN}{'='*50}{RESET}")

def check_python():
    v = sys.version_info
    assert v.major == 3 and v.minor == 10, \
        f"Python 3.10 필요, 현재 {v.major}.{v.minor}"
    return f"{v.major}.{v.minor}.{v.micro}"

check("Python 3.10", check_python)


# ══════════════════════════════════════════════
# 2. 핵심 패키지 import 확인
# ══════════════════════════════════════════════
print(f"\n{CYAN}{'='*50}{RESET}")
print(f"{CYAN}  Step 2. 핵심 패키지 확인{RESET}")
print(f"{CYAN}{'='*50}{RESET}")

packages = [
    ("numpy",       "numpy",       "__version__"),
    ("torch",       "torch",       "__version__"),
    ("genesis",     "genesis",     None),
    ("hydra",       "hydra",       "__version__"),
    ("omegaconf",   "omegaconf",   "__version__"),
    ("tensorboard", "tensorboard", "__version__"),
]

for display, mod_name, attr in packages:
    def _check(m=mod_name, a=attr):
        mod = importlib.import_module(m)
        return getattr(mod, a) if a else "installed"
    check(display, _check)

# rsl_rl 별도 확인 (패키지명이 다름)
def check_rslrl():
    import rsl_rl
    return getattr(rsl_rl, "__version__", "installed")
check("rsl-rl", check_rslrl)


# ══════════════════════════════════════════════
# 3. PyTorch GPU 확인
# ══════════════════════════════════════════════
print(f"\n{CYAN}{'='*50}{RESET}")
print(f"{CYAN}  Step 3. GPU / CUDA 확인{RESET}")
print(f"{CYAN}{'='*50}{RESET}")

def check_cuda():
    import torch
    assert torch.cuda.is_available(), "CUDA 사용 불가 — GPU 드라이버 또는 CUDA 확인 필요"
    name = torch.cuda.get_device_name(0)
    mem  = torch.cuda.get_device_properties(0).total_memory / 1024**3
    return f"{name}  ({mem:.1f} GB)"

def check_cuda_version():
    import torch
    return torch.version.cuda

def check_tensor_gpu():
    import torch
    t = torch.zeros(3, 3).cuda()
    return f"shape={list(t.shape)}, device={t.device}"

check("CUDA available", check_cuda)
check("CUDA version",   check_cuda_version)
check("GPU tensor 생성", check_tensor_gpu)


# ══════════════════════════════════════════════
# 4. Genesis 초기화 확인
# ══════════════════════════════════════════════
print(f"\n{CYAN}{'='*50}{RESET}")
print(f"{CYAN}  Step 4. Genesis 초기화 확인{RESET}")
print(f"{CYAN}{'='*50}{RESET}")

def check_genesis_init():
    import genesis as gs
    gs.init(backend=gs.cuda, logging_level="warning")
    return "Genesis initialized (cuda backend)"

def check_genesis_scene():
    import genesis as gs
    scene = gs.Scene(show_viewer=False)
    scene.add_entity(gs.morphs.Plane())
    scene.build(n_envs=1)
    return "Scene 생성 성공 (n_envs=1)"

check("Genesis init",  check_genesis_init)
check("Genesis scene", check_genesis_scene)


# ══════════════════════════════════════════════
# 5. HumanoidVerse 패키지 확인
# ══════════════════════════════════════════════
print(f"\n{CYAN}{'='*50}{RESET}")
print(f"{CYAN}  Step 5. HumanoidVerse 확인{RESET}")
print(f"{CYAN}{'='*50}{RESET}")

def check_humanoidverse():
    import humanoidverse
    return "installed"

def check_robot_config():
    import os
    # HumanoidVerse 루트 기준 config 경로
    config_path = "humanoidverse/config/robot/g1/g1_12dof.yaml"
    assert os.path.exists(config_path), \
        f"G1-12DoF config 파일 없음: {config_path}"
    return config_path

def check_reward_config():
    import os
    reward_path = "humanoidverse/config/rewards/loco/reward_h1_locomotion.yaml"
    assert os.path.exists(reward_path), \
        f"reward config 없음: {reward_path}"
    return reward_path

check("humanoidverse import",   check_humanoidverse)
check("G1-12DoF robot config",  check_robot_config)
check("locomotion reward config", check_reward_config)


# ══════════════════════════════════════════════
# 6. G1 URDF/MJCF 로딩 확인
# ══════════════════════════════════════════════
print(f"\n{CYAN}{'='*50}{RESET}")
print(f"{CYAN}  Step 6. G1-12DoF 로봇 로딩 확인{RESET}")
print(f"{CYAN}{'='*50}{RESET}")

def check_g1_loading():
    import genesis as gs
    import glob, os

    # URDF 또는 MJCF 경로 탐색
    patterns = [
        "humanoidverse/assets/robots/g1/*.urdf",
        "humanoidverse/assets/robots/g1/*.xml",
        "humanoidverse/data/robots/g1/*.urdf",
    ]
    found = []
    for p in patterns:
        found.extend(glob.glob(p))

    assert found, "G1 URDF/MJCF 파일을 찾을 수 없음 (assets 경로 확인)"
    return f"발견: {os.path.basename(found[0])}"

check("G1 asset 파일 확인", check_g1_loading)


# ══════════════════════════════════════════════
# 최종 결과
# ══════════════════════════════════════════════
total = results["pass"] + results["fail"]
print(f"\n{CYAN}{'='*50}{RESET}")
print(f"  결과: {GREEN}{results['pass']} passed{RESET} / "
      f"{RED}{results['fail']} failed{RESET} / {total} total")
print(f"{CYAN}{'='*50}{RESET}\n")

if results["fail"] == 0:
    print(f"{GREEN}모든 테스트 통과! 학습을 시작할 수 있습니다.{RESET}")
    print(f"\n다음 명령으로 학습을 시작하세요:")
    print(f"""
  python humanoidverse/train_agent.py \\
    +simulator=genesis \\
    +exp=locomotion \\
    +domain_rand=NO_domain_rand \\
    +rewards=loco/reward_h1_locomotion \\
    +robot=g1/g1_12dof \\
    +terrain=terrain_locomotion_plane \\
    +obs=loco/leggedloco_obs_singlestep_withlinvel \\
    num_envs=1024 \\
    project_name=G1_Locomotion \\
    experiment_name=G1_12dof_stage1 \\
    headless=True
""")
else:
    print(f"{RED}일부 테스트 실패. 위의 [FAIL] 항목을 먼저 해결하세요.{RESET}")
    print(f"{YELLOW}에러 메시지를 복사해서 문의하면 빠르게 해결할 수 있습니다.{RESET}\n")

sys.exit(0 if results["fail"] == 0 else 1)