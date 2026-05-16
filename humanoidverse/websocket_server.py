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
        self._lock = threading.Lock()
        self._connected_clients = set()
        self._loop = None
        self._thread = None

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
        try:
            async for message in websocket:
                data = json.loads(message)
                if data.get("type") == "set_target":
                    with self._lock:
                        self._new_target = (data["x"], data["y"])
                        self._target_updated = True
        finally:
            self._connected_clients.discard(websocket)

    def apply_target_if_updated(self, env):
        with self._lock:
            if self._target_updated and self._new_target is not None:
                tx, ty = self._new_target
                env.target_pos[0, 0] = tx
                env.target_pos[0, 1] = ty
                self._target_updated = False

    def send_robot_state(self, env):
        if not self._connected_clients or self._loop is None:
            return

        obs = env.obstacle_pos[0].cpu().tolist()  # [[x1,y1],[x2,y2],[x3,y3]]

        state = {
            "joint_angles": env.simulator.dof_pos[0].cpu().tolist(),
            "robot_pos": env.simulator.robot_root_states[0, :3].cpu().tolist(),
            "robot_quat": env.simulator.robot_root_states[0, 3:7].cpu().tolist(),
            "target_pos": env.target_pos[0].cpu().tolist(),
            "obstacle_pos": [v for obs_xy in obs for v in obs_xy],  # [x1,y1,x2,y2,x3,y3]
        }
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
