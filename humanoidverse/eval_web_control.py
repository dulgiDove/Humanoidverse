import os
import sys
from pathlib import Path

import hydra
from hydra.utils import instantiate
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf
from humanoidverse.utils.logging import HydraLoggerBridge
import logging
from utils.config_utils import *  # noqa: E402, F403

# add argparse arguments

from humanoidverse.utils.config_utils import *  # noqa: E402, F403
from loguru import logger

import threading
from pynput import keyboard

def on_press(key, env):
    try:
        if key.char == 'n':
            env.next_task()
            logger.info("Moved to the next task.")
        # Force Control
        if hasattr(key, 'char'):
            if key.char == '1':
                env.apply_force_tensor[:, env.left_hand_link_index, 2] += 1.0
                logger.info(f"Left hand force: {env.apply_force_tensor[:, env.left_hand_link_index, :]}")
            elif key.char == '2':
                env.apply_force_tensor[:, env.left_hand_link_index, 2] -= 1.0
                logger.info(f"Left hand force: {env.apply_force_tensor[:, env.left_hand_link_index, :]}")
            elif key.char == '3':
                env.apply_force_tensor[:, env.right_hand_link_index, 2] += 1.0
                logger.info(f"Right hand force: {env.apply_force_tensor[:, env.right_hand_link_index, :]}")
            elif key.char == '4':
                env.apply_force_tensor[:, env.right_hand_link_index, 2] -= 1.0
                logger.info(f"Right hand force: {env.apply_force_tensor[:, env.right_hand_link_index, :]}")
    except AttributeError:
        pass

def listen_for_keypress(env):
    with keyboard.Listener(on_press=lambda key: on_press(key, env)) as listener:
        listener.join()


# ===================== Web Control Panel =====================
import json
import math
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse

import torch

WEB_CONTROL_PANEL_HTML = r"""
<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>Humanoid Navigation Challenge</title>
  <style>
    body {
      font-family: Arial, sans-serif;
      margin: 0;
      background: #f4f4f4;
      color: #222;
    }
    .topbar {
      background: #222;
      color: white;
      padding: 12px 18px;
      font-size: 14px;
      position: sticky;
      top: 0;
      z-index: 10;
    }
    .status {
      font-weight: bold;
      margin-right: 24px;
    }
    .checkpoint {
      font-family: monospace;
      font-size: 12px;
      color: #d8d8d8;
    }
    .container {
      padding: 20px;
      max-width: 760px;
      margin: auto;
    }
    h1 {
      font-size: 24px;
      margin: 0 0 16px 0;
    }
    .card {
      background: white;
      padding: 16px;
      margin-bottom: 14px;
      border-radius: 10px;
      box-shadow: 0 2px 8px rgba(0,0,0,0.08);
    }
    .row {
      display: grid;
      grid-template-columns: 120px 1fr 1fr 100px;
      gap: 10px;
      align-items: center;
      margin: 8px 0;
    }
    input {
      padding: 8px;
      font-size: 14px;
    }
    button {
      padding: 9px 12px;
      font-size: 14px;
      cursor: pointer;
      border: 0;
      border-radius: 6px;
      background: #333;
      color: white;
    }
    button:hover {
      background: #555;
    }
    .buttons {
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
      margin-top: 12px;
    }
    .primary {
      background: #0b63ce;
    }
    .danger {
      background: #b3261e;
    }
    .soft {
      background: #666;
    }
    pre {
      white-space: pre-wrap;
      background: #111;
      color: #eee;
      padding: 12px;
      border-radius: 8px;
      overflow-x: auto;
      min-height: 120px;
    }
    .domain-info {
      background: #eef5ff;
      border-left: 4px solid #0b63ce;
      padding: 10px 12px;
      border-radius: 6px;
      font-size: 13px;
      line-height: 1.5;
      margin-bottom: 14px;
    }
    .domain-info b {
      color: #0b3f87;
    }
    .hint {
      color: #555;
      font-size: 13px;
      margin-top: 8px;
    }
  </style>
</head>
<body>
  <div class="topbar">
    <span class="status">Status: <span id="status">UNKNOWN</span></span>
    <span class="checkpoint">Checkpoint: <span id="checkpoint">-</span></span>
  </div>

  <div class="container">
    <h1>Humanoid Navigation Challenge</h1>

    <div class="card">
      <h3>Coordinate Setup</h3>

      <div class="domain-info">
        <b>Random Domain</b><br>
        Target Random: 현재 로봇 위치 기준 거리 <b>1.5m ~ 8.0m</b> 범위에서 생성됩니다.<br>
        Obstacle Random: 현재 로봇 위치 기준 거리 <b>1.5m ~ 5.0m</b> 범위에서 생성됩니다.<br>
        좌표의 X/Y 각각이 위 범위에 제한되는 것이 아니라, <b>로봇과의 거리(radius)</b> 기준입니다.<br>
        Target과 Obstacle, Obstacle끼리는 최소 <b>0.8m</b> 이상 떨어지도록 생성합니다.
      </div>

      <div class="row">
        <b>Target</b>
        <input id="targetX" type="number" step="0.1" placeholder="X">
        <input id="targetY" type="number" step="0.1" placeholder="Y">
        <button onclick="randomOne('target')">Random</button>
      </div>

      <div class="row">
        <b>Obstacle 1</b>
        <input id="obs0X" type="number" step="0.1" placeholder="X">
        <input id="obs0Y" type="number" step="0.1" placeholder="Y">
        <button onclick="randomOne('obs0')">Random</button>
      </div>

      <div class="row">
        <b>Obstacle 2</b>
        <input id="obs1X" type="number" step="0.1" placeholder="X">
        <input id="obs1Y" type="number" step="0.1" placeholder="Y">
        <button onclick="randomOne('obs1')">Random</button>
      </div>

      <div class="row">
        <b>Obstacle 3</b>
        <input id="obs2X" type="number" step="0.1" placeholder="X">
        <input id="obs2Y" type="number" step="0.1" placeholder="Y">
        <button onclick="randomOne('obs2')">Random</button>
      </div>

      <div class="buttons">
        <button class="soft" onclick="randomAll()">Random All</button>
        <button class="primary" onclick="applyCoords()">Apply</button>
      </div>

      <div class="hint">
        Random은 입력칸만 바꿉니다. 실제 Genesis 환경에는 Apply를 눌러야 반영됩니다.
      </div>
    </div>

    <div class="card">
      <h3>Simulation Control</h3>
      <div class="buttons">
        <button class="primary" onclick="startSim()">Start</button>
        <button onclick="pauseSim()">Pause</button>
        <button class="danger" onclick="resetSim()">Reset</button>
        <button class="soft" onclick="showInfo()">Info</button>
        <button class="soft" onclick="fitCamera()">Fit Camera</button>
      </div>
    </div>

    <div class="card">
      <h3>Info Output</h3>
      <pre id="info">Press Info button.</pre>
    </div>
  </div>

<script>
let currentRobot = [0.0, 0.0];
function fmt(v) {
  if (v === null || v === undefined) return "-";
  return Number(v).toFixed(3);
}

async function api(path, body=null) {
  const opt = body === null
    ? { method: "GET" }
    : {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body)
      };

  const res = await fetch(path, opt);
  return await res.json();
}

function setXY(prefix, x, y) {
  document.getElementById(prefix + "X").value = Number(x).toFixed(2);
  document.getElementById(prefix + "Y").value = Number(y).toFixed(2);
}

function getXY(prefix) {
  return [
    Number(document.getElementById(prefix + "X").value),
    Number(document.getElementById(prefix + "Y").value)
  ];
}

function dist(a, b) {
  const dx = a[0] - b[0];
  const dy = a[1] - b[1];
  return Math.sqrt(dx * dx + dy * dy);
}

function randRange(minV, maxV) {
  return minV + Math.random() * (maxV - minV);
}

function randomPolarAround(center, minR, maxR) {
  const r = randRange(minR, maxR);
  const t = -Math.PI + Math.random() * Math.PI * 2.0;
  return [
    center[0] + r * Math.cos(t),
    center[1] + r * Math.sin(t)
  ];
}

function readXYOrNull(prefix) {
  const x = Number(document.getElementById(prefix + "X").value);
  const y = Number(document.getElementById(prefix + "Y").value);
  if (Number.isNaN(x) || Number.isNaN(y)) return null;
  return [x, y];
}

function getExistingObstaclesExcept(exceptIndex) {
  const result = [];
  for (let i = 0; i < 3; i++) {
    if (i === exceptIndex) continue;
    const xy = readXYOrNull("obs" + i);
    if (xy !== null) result.push(xy);
  }
  return result;
}

function randomTargetCandidate() {
  // 학습 코드 기준: robot 기준 min_goal_dist ~ max_goal_dist.
  // 커리큘럼 전체 범위에 맞춰 1.5m ~ 8.0m 사용.
  return randomPolarAround(currentRobot, 1.5, 8.0);
}

function randomObstacleCandidate() {
  // 학습 코드 기준: obstacle은 robot 기준 1.5m ~ 5.0m.
  return randomPolarAround(currentRobot, 1.5, 5.0);
}

function randomOne(name) {
  const minGap = 0.8;

  if (name === "target") {
    const obstacles = getExistingObstaclesExcept(null);

    for (let tries = 0; tries < 100; tries++) {
      const xy = randomTargetCandidate();

      let ok = true;
      for (const obs of obstacles) {
        if (dist(xy, obs) < minGap) {
          ok = false;
          break;
        }
      }

      if (ok) {
        setXY("target", xy[0], xy[1]);
        return;
      }
    }

    const fallback = randomTargetCandidate();
    setXY("target", fallback[0], fallback[1]);
    return;
  }

  if (name.startsWith("obs")) {
    const idx = Number(name.replace("obs", ""));
    const target = readXYOrNull("target") || [currentRobot[0] + 3.0, currentRobot[1]];
    const otherObstacles = getExistingObstaclesExcept(idx);

    for (let tries = 0; tries < 100; tries++) {
      const xy = randomObstacleCandidate();

      let ok = true;

      if (dist(xy, currentRobot) < 0.5) ok = false;
      if (dist(xy, target) < minGap) ok = false;

      for (const obs of otherObstacles) {
        if (dist(xy, obs) < minGap) {
          ok = false;
          break;
        }
      }

      if (ok) {
        setXY(name, xy[0], xy[1]);
        return;
      }
    }

    const fallback = randomObstacleCandidate();
    setXY(name, fallback[0], fallback[1]);
  }
}

function randomAll() {
  randomOne("target");
  randomOne("obs0");
  randomOne("obs1");
  randomOne("obs2");
}

function collectPayload() {
  return {
    target: getXY("target"),
    obstacles: [
      getXY("obs0"),
      getXY("obs1"),
      getXY("obs2")
    ]
  };
}

async function applyCoords() {
  const payload = collectPayload();
  const data = await api("/api/apply", payload);
  document.getElementById("info").textContent = data.message;
  await refreshTop(false);
}

async function startSim() {
  const data = await api("/api/start", {});
  document.getElementById("info").textContent = data.message;
  await refreshTop(false);
}

async function pauseSim() {
  const data = await api("/api/pause", {});
  document.getElementById("info").textContent = data.message;
  await refreshTop(false);
}

async function resetSim() {
  const data = await api("/api/reset", {});
  document.getElementById("info").textContent = data.message;
  await refreshTop(false);
}

async function fitCamera() {
  const data = await api("/api/fit_camera", {});
  document.getElementById("info").textContent = data.message;
}

async function loadMeta() {
  const data = await api("/api/meta");
  document.getElementById("checkpoint").textContent = data.checkpoint || "-";
}

async function showInfo() {
  const data = await api("/api/info");

  const lines = [];
  lines.push(`Robot Position      : (${fmt(data.robot[0])}, ${fmt(data.robot[1])})`);
  lines.push(`Target Position     : (${fmt(data.target[0])}, ${fmt(data.target[1])})`);
  lines.push(`Obstacle 1          : (${fmt(data.obstacles[0][0])}, ${fmt(data.obstacles[0][1])})`);
  lines.push(`Obstacle 2          : (${fmt(data.obstacles[1][0])}, ${fmt(data.obstacles[1][1])})`);
  lines.push(`Obstacle 3          : (${fmt(data.obstacles[2][0])}, ${fmt(data.obstacles[2][1])})`);
  lines.push(`Distance to Target  : ${fmt(data.distance)} m`);

  document.getElementById("info").textContent = lines.join("\n");
}

async function refreshTop(fillInputs=false) {
  const data = await api("/api/info");

  document.getElementById("status").textContent = data.status;
  currentRobot = data.robot || [0.0, 0.0];

  if (fillInputs) {
    setXY("target", data.target[0], data.target[1]);
    setXY("obs0", data.obstacles[0][0], data.obstacles[0][1]);
    setXY("obs1", data.obstacles[1][0], data.obstacles[1][1]);
    setXY("obs2", data.obstacles[2][0], data.obstacles[2][1]);
  }
}

window.onload = async () => {
  await loadMeta();
  await refreshTop(true);
  setInterval(() => refreshTop(false), 800);
};
</script>
</body>
</html>
"""

_WEB_LOCK = threading.Lock()
_WEB_STATE = {
    "running": False,
    "checkpoint": "",
    "apply_payload": None,
    "reset_requested": False,
    "fit_camera_requested": False,
    "last_payload": None,
    "last_scene_signature": None,
    "snapshot": {
        "status": "PAUSED",
        "robot": [0.0, 0.0],
        "target": [0.0, 0.0],
        "obstacles": [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
        "distance": 0.0,
        },
}

_SNAPSHOT_INTERVAL_SEC = 0.10
_LAST_SNAPSHOT_UPDATE = 0.0


def maybe_update_web_snapshot(env, force: bool = False):
    """웹 페이지가 읽는 snapshot을 너무 자주 갱신하지 않도록 제한한다."""
    global _LAST_SNAPSHOT_UPDATE

    now = time.time()
    if not force and (now - _LAST_SNAPSHOT_UPDATE) < _SNAPSHOT_INTERVAL_SEC:
        return

    update_web_snapshot(env)
    _LAST_SNAPSHOT_UPDATE = now


def web_set_checkpoint(path: str):
    with _WEB_LOCK:
        _WEB_STATE["checkpoint"] = path


def web_is_running() -> bool:
    with _WEB_LOCK:
        return bool(_WEB_STATE["running"])



def _safe_float(v, default=0.0):
    try:
        x = float(v)
        if not math.isfinite(x):
            return default
        return x
    except Exception:
        return default


def _tensor_xy_to_list(t):
    try:
        return [
            float(t[0].detach().cpu().item()),
            float(t[1].detach().cpu().item()),
        ]
    except Exception:
        return [0.0, 0.0]


def _get_robot_xy_tensor(env):
    try:
        return env.simulator.robot_root_states[0, :2]
    except Exception:
        return torch.zeros(2, device=env.device)


def _get_robot_xy(env):
    return _tensor_xy_to_list(_get_robot_xy_tensor(env))


def _get_target_xy(env):
    try:
        return _tensor_xy_to_list(env.target_pos[0])
    except Exception:
        return [0.0, 0.0]


def _get_obstacles_xy(env):
    result = []
    try:
        for i in range(3):
            result.append(_tensor_xy_to_list(env.obstacle_pos[0, i]))
    except Exception:
        result = [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]
    return result


def _update_markers(env):
    try:
        tx = float(env.target_pos[0, 0].detach().cpu().item())
        ty = float(env.target_pos[0, 1].detach().cpu().item())

        if hasattr(env.simulator, "target_marker"):
            env.simulator.target_marker.set_pos([[tx, ty, 0.3]])

        if hasattr(env.simulator, "obstacle_markers"):
            for i in range(3):
                ox = float(env.obstacle_pos[0, i, 0].detach().cpu().item())
                oy = float(env.obstacle_pos[0, i, 1].detach().cpu().item())
                env.simulator.obstacle_markers[i].set_pos([[ox, oy, 0.76]])
    except Exception as e:
        print("[web control] marker update skipped:", e)


def _refresh_prev_dist(env):
    try:
        robot_xy = _get_robot_xy_tensor(env)
        env.prev_dist_to_target[0] = torch.norm(env.target_pos[0] - robot_xy)
    except Exception as e:
        print("[web control] prev_dist update skipped:", e)


def _apply_payload_to_env(env, payload):
    target = payload.get("target", None)
    obstacles = payload.get("obstacles", None)

    with torch.no_grad():
        if target is not None and len(target) >= 2:
            env.target_pos[0, 0] = _safe_float(target[0])
            env.target_pos[0, 1] = _safe_float(target[1])

        if obstacles is not None:
            for i in range(min(3, len(obstacles))):
                if obstacles[i] is None or len(obstacles[i]) < 2:
                    continue
                env.obstacle_pos[0, i, 0] = _safe_float(obstacles[i][0])
                env.obstacle_pos[0, i, 1] = _safe_float(obstacles[i][1])

    _refresh_prev_dist(env)
    _update_markers(env)




def _scene_signature(env):
    """Camera auto-fit trigger용. robot은 움직이므로 제외하고 target/obstacles만 비교한다."""
    try:
        target = _get_target_xy(env)
        obstacles = _get_obstacles_xy(env)

        values = []
        for p in [target] + obstacles:
            values.append(round(float(p[0]), 3))
            values.append(round(float(p[1]), 3))

        return tuple(values)
    except Exception:
        return None


def remember_current_scene_signature(env):
    sig = _scene_signature(env)
    with _WEB_LOCK:
        _WEB_STATE["last_scene_signature"] = sig


def maybe_fit_camera_on_scene_change(env):
    """목표/장애물 배치가 바뀌면 자동으로 카메라를 전체 조망 위치로 이동한다."""
    sig = _scene_signature(env)
    if sig is None:
        return

    should_fit = False

    with _WEB_LOCK:
        prev = _WEB_STATE.get("last_scene_signature", None)

        if prev is None:
            # 최초 장면도 바로 전체가 보이도록 카메라를 맞춘다.
            _WEB_STATE["last_scene_signature"] = sig
            should_fit = True
        elif sig != prev:
            _WEB_STATE["last_scene_signature"] = sig
            should_fit = True

    if should_fit:
        print("[web control] scene changed -> auto fit camera")
        fit_overview_camera(env)

def update_web_snapshot(env):
    maybe_fit_camera_on_scene_change(env)

    robot = _get_robot_xy(env)
    target = _get_target_xy(env)
    obstacles = _get_obstacles_xy(env)

    dx = target[0] - robot[0]
    dy = target[1] - robot[1]
    distance = math.sqrt(dx * dx + dy * dy)

    with _WEB_LOCK:
        status = "RUNNING" if _WEB_STATE["running"] else "PAUSED"

        _WEB_STATE["snapshot"] = {
            "status": status,
            "robot": robot,
            "target": target,
            "obstacles": obstacles,
            "distance": distance,
        }

def fit_overview_camera(env):
    """Robot / target / obstacles가 모두 보이도록 viewer camera를 넓게 배치한다."""
    try:
        robot = _get_robot_xy(env)
        target = _get_target_xy(env)
        obstacles = _get_obstacles_xy(env)

        points = [robot, target] + obstacles
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]

        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)

        cx = (min_x + max_x) * 0.5
        cy = (min_y + max_y) * 0.5

        width = max_x - min_x
        depth = max_y - min_y

        # 기존보다 더 넓게 잡는다.
        # 기존: span = max(width, depth, 6.0)
        # 변경: 최소 시야 범위를 키우고, 전체 span에 여유 배율을 추가한다.
        base_span = max(width, depth, 7.0)
        span = base_span * 1.25

        cam_pos = (
            cx - span * 0.80,
            cy - span * 1.00,
            max(8.0, span * 1.20),
        )

        cam_lookat = (
            cx,
            cy,
            0.55,
        )

        candidates = []

        sim = getattr(env, "simulator", None)
        if sim is not None:
            if hasattr(sim, "scene"):
                scene = sim.scene
                candidates.append(scene)

                if hasattr(scene, "viewer"):
                    candidates.append(scene.viewer)

            if hasattr(sim, "viewer"):
                candidates.append(sim.viewer)

        ok = False
        last_error = None

        for obj in candidates:
            if obj is None:
                continue

            # Genesis 버전/래퍼마다 함수 이름이 다를 수 있어서 여러 방식 시도
            for call in [
                lambda: obj.set_camera_pose(pos=cam_pos, lookat=cam_lookat),
                lambda: obj.set_camera_pose(camera_pos=cam_pos, camera_lookat=cam_lookat),
                lambda: obj.set_camera_pose(cam_pos, cam_lookat),
                lambda: obj.set_camera(pos=cam_pos, lookat=cam_lookat),
                lambda: obj.set_camera(camera_pos=cam_pos, camera_lookat=cam_lookat),
            ]:
                try:
                    call()
                    ok = True
                    break
                except Exception as e:
                    last_error = e

            if ok:
                break

            # 속성 직접 세팅 방식도 시도
            try:
                if hasattr(obj, "camera_pos"):
                    obj.camera_pos = cam_pos
                    ok = True
                if hasattr(obj, "camera_lookat"):
                    obj.camera_lookat = cam_lookat
                    ok = True
                if ok:
                    break
            except Exception as e:
                last_error = e

        msg = (
            f"Fit Camera: pos={tuple(round(v, 2) for v in cam_pos)}, "
            f"lookat={tuple(round(v, 2) for v in cam_lookat)}"
        )

        if ok:
            print("[web control]", msg)
            return True, msg

        msg = msg + f" | camera API not found. last_error={last_error}"
        print("[web control]", msg)
        return False, msg

    except Exception as e:
        msg = f"Fit Camera failed: {e}"
        print("[web control]", msg)
        return False, msg

def process_web_control(env):
    reset_requested = False
    fit_camera_requested = False
    apply_payload = None
    last_payload = None

    with _WEB_LOCK:
        reset_requested = _WEB_STATE["reset_requested"]
        _WEB_STATE["reset_requested"] = False

        fit_camera_requested = _WEB_STATE["fit_camera_requested"]
        _WEB_STATE["fit_camera_requested"] = False

        apply_payload = _WEB_STATE["apply_payload"]
        _WEB_STATE["apply_payload"] = None

        last_payload = _WEB_STATE["last_payload"]

    reset_result = None

    if reset_requested:
        print("[web control] reset requested")
        with torch.no_grad():
            reset_result = env.reset_all()

        with _WEB_LOCK:
            _WEB_STATE["running"] = False

        if last_payload is not None:
            _apply_payload_to_env(env, last_payload)

        fit_overview_camera(env)  # auto after reset
        remember_current_scene_signature(env)  # after reset

    if apply_payload is not None:
        print("[web control] apply coordinates")
        _apply_payload_to_env(env, apply_payload)

        with _WEB_LOCK:
            _WEB_STATE["running"] = False
            _WEB_STATE["last_payload"] = apply_payload

        fit_overview_camera(env)  # auto after apply
        remember_current_scene_signature(env)  # after apply

    if fit_camera_requested:
        fit_overview_camera(env)
        remember_current_scene_signature(env)  # after manual fit

    maybe_update_web_snapshot(
        env,
        force=(reset_requested or apply_payload is not None or fit_camera_requested),
    )
    return reset_result


class WebControlHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        return

    def _send_html(self, html: str, code: int = 200):
        body = html.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_json(self, data, code: int = 200):
        body = json.dumps(data).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_json(self):
        length = int(self.headers.get("Content-Length", "0") or 0)
        if length <= 0:
            return {}
        raw = self.rfile.read(length).decode("utf-8")
        if not raw:
            return {}
        return json.loads(raw)

    def do_GET(self):
        parsed = urlparse(self.path)

        if parsed.path == "/":
            self._send_html(WEB_CONTROL_PANEL_HTML)
            return

        if parsed.path == "/api/info":
            with _WEB_LOCK:
                data = dict(_WEB_STATE["snapshot"])
            self._send_json(data)
            return

        if parsed.path == "/api/meta":
            with _WEB_LOCK:
                data = {"checkpoint": _WEB_STATE["checkpoint"]}
            self._send_json(data)
            return

        self._send_json({"error": "not found"}, code=404)

    def do_POST(self):
        parsed = urlparse(self.path)

        try:
            payload = self._read_json()
        except Exception as e:
            self._send_json({"error": f"invalid json: {e}"}, code=400)
            return

        if parsed.path == "/api/start":
            with _WEB_LOCK:
                _WEB_STATE["running"] = True
            self._send_json({"ok": True, "message": "Start requested."})
            return

        if parsed.path == "/api/pause":
            with _WEB_LOCK:
                _WEB_STATE["running"] = False
            self._send_json({"ok": True, "message": "Paused."})
            return

        if parsed.path == "/api/reset":
            with _WEB_LOCK:
                _WEB_STATE["running"] = False
                _WEB_STATE["reset_requested"] = True
            self._send_json({"ok": True, "message": "Reset requested."})
            return

        if parsed.path == "/api/fit_camera":
            with _WEB_LOCK:
                _WEB_STATE["fit_camera_requested"] = True
            self._send_json({"ok": True, "message": "Fit Camera requested."})
            return

        if parsed.path == "/api/apply":
            with _WEB_LOCK:
                _WEB_STATE["running"] = False
                _WEB_STATE["apply_payload"] = payload
            self._send_json({"ok": True, "message": "Apply requested. Simulation paused."})
            return

        self._send_json({"error": "not found"}, code=404)


def start_web_control_server(port: int = 8080):
    server = ThreadingHTTPServer(("0.0.0.0", port), WebControlHandler)

    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    print("")
    print("===================================================")
    print(f"[web control] open: http://localhost:{port}")
    print("===================================================")
    print("")

    return server
# =================== End Web Control Panel ===================


@hydra.main(config_path="config", config_name="base_eval")
def main(override_config: OmegaConf):
    # logging to hydra log file
    hydra_log_path = os.path.join(HydraConfig.get().runtime.output_dir, "eval.log")
    logger.remove()
    logger.add(hydra_log_path, level="DEBUG")

    # Get log level from LOGURU_LEVEL environment variable or use INFO as default
    console_log_level = os.environ.get("LOGURU_LEVEL", "INFO").upper()
    logger.add(sys.stdout, level=console_log_level, colorize=True)

    logging.basicConfig(level=logging.DEBUG)
    logging.getLogger().addHandler(HydraLoggerBridge())

    os.chdir(hydra.utils.get_original_cwd())

    if override_config.checkpoint is not None:
        has_config = True
        checkpoint = Path(override_config.checkpoint)
        config_path = checkpoint.parent / "config.yaml"
        if not config_path.exists():
            config_path = checkpoint.parent.parent / "config.yaml"
            if not config_path.exists():
                has_config = False
                logger.error(f"Could not find config path: {config_path}")

        if has_config:
            logger.info(f"Loading training config file from {config_path}")
            with open(config_path) as file:
                train_config = OmegaConf.load(file)

            if train_config.eval_overrides is not None:
                train_config = OmegaConf.merge(
                    train_config, train_config.eval_overrides
                )

            config = OmegaConf.merge(train_config, override_config)
        else:
            config = override_config
    else:
        if override_config.eval_overrides is not None:
            config = override_config.copy()
            eval_overrides = OmegaConf.to_container(config.eval_overrides, resolve=True)
            for arg in sys.argv[1:]:
                if not arg.startswith("+"):
                    key = arg.split("=")[0]
                    if key in eval_overrides:
                        del eval_overrides[key]
            config.eval_overrides = OmegaConf.create(eval_overrides)
            config = OmegaConf.merge(config, eval_overrides)
        else:
            config = override_config
            
    simulator_type = config.simulator['_target_'].split('.')[-1]
    if simulator_type == 'IsaacSim':
        from omni.isaac.lab.app import AppLauncher
        import argparse
        parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
        parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
        parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
        parser.add_argument("--env_spacing", type=int, default=20, help="Distance between environments in simulator.")
        parser.add_argument("--output_dir", type=str, default="logs", help="Directory to store the training output.")
        AppLauncher.add_app_launcher_args(parser)

        # Parse known arguments to get argparse params
        args_cli, hydra_args = parser.parse_known_args()

        app_launcher = AppLauncher(args_cli)
        simulation_app = app_launcher.app
        print('args_cli', args_cli)
        print('hydra_args', hydra_args)
        sys.argv = [sys.argv[0]] + hydra_args
    if simulator_type == 'IsaacGym':
        import isaacgym
        
    from humanoidverse.agents.base_algo.base_algo import BaseAlgo  # noqa: E402
    from humanoidverse.utils.helpers import pre_process_config

    pre_process_config(config)

    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    eval_log_dir = Path(config.eval_log_dir)
    eval_log_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Saving eval logs to {eval_log_dir}")
    with open(eval_log_dir / "config.yaml", "w") as file:
        OmegaConf.save(config, file)

    ckpt_num = config.checkpoint.split('/')[-1].split('_')[-1].split('.')[0]
    config.env.config.save_rendering_dir = str(checkpoint.parent / "renderings" / f"ckpt_{ckpt_num}")
    config.env.config.ckpt_dir = str(checkpoint.parent) # commented out for now, might need it back to save motion
    env = instantiate(config.env, device=device)

    # 영상화 시 goal 거리 고정 (커리큘럼 무시하고 8m 목표)
    env.max_goal_dist = 8.0
    env.min_goal_dist = 8.0

    # Start a thread to listen for key press
    key_listener_thread = threading.Thread(target=listen_for_keypress, args=(env,))
    key_listener_thread.daemon = True
    key_listener_thread.start()

    algo: BaseAlgo = instantiate(config.algo, env=env, device=device, log_dir=None)
    algo.setup()
    algo.load(config.checkpoint)
    web_set_checkpoint(str(config.checkpoint))
    start_web_control_server(port=8080)
    algo._create_eval_callbacks()
    algo._pre_evaluate_policy()

    eval_policy = algo._get_inference_policy()
    obs_dict = env.reset_all()
    all_envs = torch.arange(env.num_envs, device=device)
    env._resample_target(all_envs)
    env._resample_obstacles(all_envs)

    init_actions = torch.zeros(env.num_envs, algo.num_act, device=device)
    actor_state = {
        "obs": obs_dict,
        "actions": init_actions,
        "done_indices": [],
        "stop": False
    }
    step = 0
    while True:
        _web_reset_obs = process_web_control(env)
        if _web_reset_obs is not None:
            try:
                if isinstance(_web_reset_obs, tuple):
                    actor_state['obs'] = _web_reset_obs[0]
                else:
                    actor_state['obs'] = _web_reset_obs
            except Exception as e:
                print('[web control] reset obs update skipped:', e)
        if not web_is_running():
            time.sleep(0.03)
            continue
        actor_state["step"] = step
        actions = eval_policy(actor_state["obs"]['actor_obs'])
        actor_state["actions"] = actions
        actor_state = algo.env_step(actor_state)
        
        step += 1

        if step >= 3000:
            break

if __name__ == "__main__":
    main()