import asyncio
import json
import threading
import websockets


class RobotWebSocketServer:

    def __init__(self, host="localhost", port=8765):
        self.host = host
        self.port = port
        self._new_target = None
        self._target_updated = False
        self._new_obstacles = [None, None, None]
        self._obstacle_updated = [False, False, False]
        self._lock = threading.Lock()
        self._connected_clients = set()
        self._loop = None
        self._thread = None
        self._target_changed = False
        self._obstacles_changed = False

    def start(self):
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self):
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        self._loop.run_until_complete(self._serve())

    async def _serve(self):
        async with websockets.serve(self._handler, self.host, self.port):
            await asyncio.Future()

    async def _handler(self, websocket):
        self._connected_clients.add(websocket)
        with self._lock:
            self._target_changed = True
            self._obstacles_changed = True
        try:
            async for message in websocket:
                data = json.loads(message)
                if data.get("type") == "set_target":
                    with self._lock:
                        self._new_target = (data["x"], data["y"])
                        self._target_updated = True
                elif data.get("type") == "set_obstacle":
                    idx = data["index"]
                    with self._lock:
                        self._new_obstacles[idx] = (data["x"], data["y"])
                        self._obstacle_updated[idx] = True
        finally:
            self._connected_clients.discard(websocket)

    def apply_target_if_updated(self, env):
        with self._lock:
            if self._target_updated and self._new_target:
                tx, ty = self._new_target
                env.target_pos[0, 0] = tx
                env.target_pos[0, 1] = ty
                self._target_updated = False
                self._target_changed = True

            for i in range(3):
                if self._obstacle_updated[i] and self._new_obstacles[i]:
                    ox, oy = self._new_obstacles[i]
                    env.obstacle_pos[0, i, 0] = ox
                    env.obstacle_pos[0, i, 1] = oy
                    self._obstacle_updated[i] = False
                    self._obstacles_changed = True

    def send_robot_state(self, env):
        if not self._connected_clients or self._loop is None:
            return
        state = {
            "joint_angles": env.simulator.dof_pos[0].cpu().tolist(),
            "robot_pos":    env.simulator.robot_root_states[0, :3].cpu().tolist(),
            "robot_quat":   env.simulator.robot_root_states[0, 3:7].cpu().tolist(),
        }
        asyncio.run_coroutine_threadsafe(
            self._broadcast(json.dumps(state)), self._loop
        )

    def send_static_state(self, env):
        if not self._connected_clients or self._loop is None:
            return
        if not self._target_changed and not self._obstacles_changed:
            return

        state = {}
        if self._target_changed:
            state["target_pos"] = env.target_pos[0].cpu().tolist()
            self._target_changed = False

        if self._obstacles_changed:
            obs = env.obstacle_pos[0].cpu().tolist()
            state["obstacle_pos"] = [v for obs_xy in obs for v in obs_xy]
            self._obstacles_changed = False

        asyncio.run_coroutine_threadsafe(
            self._broadcast(json.dumps(state)), self._loop
        )

    async def _broadcast(self, message):
        if self._connected_clients:
            await asyncio.gather(
                *[client.send(message) for client in self._connected_clients],
                return_exceptions=True
            )

    def stop(self):
        if self._loop is not None:
            self._loop.call_soon_threadsafe(self._loop.stop)
