"""YTP Arena — 重建的環境

物理規則全部從 strategy.py 反推，不是猜的。對應行數標在各處。

已確認：
  - 牆壁完全彈性反彈，邊界 [ship_radius, L - ship_radius]   (strategy.py:382-393)
  - 重力井只在半徑內生效，a = G * d / r^2，G = 100          (strategy.py:1535-1556)
    注意 d 是未正規化分量，所以「大小」是 G / r，是 1/r 力場不是平方反比
  - 捕獲判定 dist <= ship_radius + chest_radius              (strategy.py:309, 396-399)
  - 速度上限用等比縮放，不是逐軸 clip                        (strategy.py:373-377)

未確認（開關留著）：
  - 船體碰撞：目前照「速度完全互換」實作，可切換成法線分量互換
  - 同 tick 雙方捕獲：目前 player 0 勝
"""

import csv
import math

EPS = 1e-9
GRAVITY_CONSTANT = 100.0

DEFAULT_CONFIG = {
    "L": 100.0,
    "dt": 0.01,
    "max_accel": 10.0,
    "max_speed": 20.0,
    "ship_radius": 0.5,
}

MAX_TICKS = 10000


# --------------------------------------------------------------------------
# scenario 讀取
# --------------------------------------------------------------------------

def load_scenario(path):
    """讀 scenario CSV。

    格式：
        L,100
        player,5,5
        player,95,95
        chest,30,28,1.20
        obstacle,25,25,10
    """
    config = dict(DEFAULT_CONFIG)
    players, chests, obstacles = [], [], []

    with open(path, newline="") as f:
        for row in csv.reader(f):
            if not row or row[0].strip().startswith("#"):
                continue
            tag = row[0].strip().lower()
            if tag == "l":
                config["L"] = float(row[1])
            elif tag == "player":
                players.append([float(row[1]), float(row[2])])
            elif tag == "chest":
                chests.append({
                    "center": [float(row[1]), float(row[2])],
                    "radius": float(row[3]),
                })
            elif tag == "obstacle":
                obstacles.append({
                    "center": [float(row[1]), float(row[2])],
                    "radius": float(row[3]),
                })
            elif tag in config:
                config[tag] = float(row[1])

    return {
        "config": config,
        "players": players,
        "chests": chests,
        "obstacles": obstacles,
    }


# --------------------------------------------------------------------------
# 環境
# --------------------------------------------------------------------------

class ArenaEnv:

    def __init__(self, scenario, swap_full_velocity=True, tie_goes_to_player0=True):
        self.scenario = scenario
        self.config = dict(scenario["config"])
        self.swap_full_velocity = swap_full_velocity
        self.tie_goes_to_player0 = tie_goes_to_player0
        self.reset()

    def reset(self, swap_spawns=False):
        spawns = list(self.scenario["players"])
        if swap_spawns:
            spawns = spawns[::-1]

        self.pos = [list(p) for p in spawns]
        self.vel = [[0.0, 0.0], [0.0, 0.0]]
        self.scores = [0, 0]
        self.tick = 0
        self.chests = [
            {"center": list(c["center"]), "radius": c["radius"]}
            for c in self.scenario["chests"]
        ]
        self.obstacles = [
            {"center": list(o["center"]), "radius": o["radius"]}
            for o in self.scenario["obstacles"]
        ]
        self.events = []
        return self.state_for(0), self.state_for(1)

    # ---------------- 給 agent 的 state ----------------

    def state_for(self, pid):
        """建 agent 看到的 state。

        鍵名跟 strategy.py 用的一致。注意 opponent_direction 這個鍵
        在 strategy.py:1842 被當成對手「位置」使用，不是方向——
        沿用主辦的命名。
        """
        opp = 1 - pid
        return {
            "tick": self.tick,
            "config": dict(self.config),
            "position": list(self.pos[pid]),
            "velocity": list(self.vel[pid]),
            "chests": [
                {"center": list(c["center"]), "radius": c["radius"]}
                for c in self.chests
            ],
            "obstacles": [
                {"center": list(o["center"]), "radius": o["radius"]}
                for o in self.obstacles
            ],
            "opponent_direction": list(self.pos[opp]),
            "scores": [self.scores[pid], self.scores[opp]],
        }

    # ---------------- 物理 ----------------

    def gravity_at(self, x, y):
        """strategy.py:1535-1556

        只有在重力井半徑「內」才受力。井外完全不受影響。
        """
        ax = ay = 0.0
        for o in self.obstacles:
            cx, cy = o["center"]
            r = o["radius"]
            dx, dy = cx - x, cy - y
            dist = math.hypot(dx, dy)
            if EPS < dist <= r:
                factor = GRAVITY_CONSTANT / (dist * dist)
                ax += factor * dx
                ay += factor * dy
        return ax, ay

    def thrust_escape_radius(self):
        """純靠推力就能抵抗重力的半徑。

        重力大小是 G / r（1/r 力場），令它等於 max_accel：
            G / r = max_accel  =>  r = G / max_accel = 10
        預設值下剛好等於井半徑本身 —— 井內任何位置推力都輸給重力。
        """
        return GRAVITY_CONSTANT / self.config["max_accel"]

    def momentum_escape_radius(self):
        """以最大速度向外衝時，還能逃出井的最小半徑。

        1/r 力場的位能 U = G ln(r)，從 r 逃到井緣 R 需要 G ln(R/r)。
        動能上限 v_max^2 / 2，解出 r = R * exp(-v_max^2 / (2G))。
        """
        R = max((o["radius"] for o in self.obstacles), default=0.0)
        v = self.config["max_speed"]
        return R * math.exp(-(v * v) / (2.0 * GRAVITY_CONSTANT))

    def _clamp_accel(self, a):
        max_accel = self.config["max_accel"]
        ax, ay = float(a[0]), float(a[1])
        mag = math.hypot(ax, ay)
        if mag <= max_accel or mag < EPS:
            return ax, ay
        scale = max_accel / mag
        return ax * scale, ay * scale

    def step(self, action0, action1):
        """推進一個 tick。action 是 [ax, ay]，超過 max_accel 會被縮放。"""
        dt = self.config["dt"]
        max_speed = self.config["max_speed"]
        ship_radius = self.config["ship_radius"]
        lo = ship_radius
        hi = self.config["L"] - ship_radius

        actions = [self._clamp_accel(action0), self._clamp_accel(action1)]

        # 1. 積分：加速度 + 重力
        for i in range(2):
            ax, ay = actions[i]
            gx, gy = self.gravity_at(*self.pos[i])

            vx = self.vel[i][0] + (ax + gx) * dt
            vy = self.vel[i][1] + (ay + gy) * dt

            speed = math.hypot(vx, vy)
            if speed > max_speed:
                scale = max_speed / speed
                vx *= scale
                vy *= scale

            self.vel[i] = [vx, vy]
            self.pos[i][0] += vx * dt
            self.pos[i][1] += vy * dt

        # 2. 牆壁反彈（strategy.py:382-393，位置鏡射、速度變號）
        for i in range(2):
            x, y = self.pos[i]
            vx, vy = self.vel[i]
            if x < lo:
                x, vx = 2.0 * lo - x, -vx
            elif x > hi:
                x, vx = 2.0 * hi - x, -vx
            if y < lo:
                y, vy = 2.0 * lo - y, -vy
            elif y > hi:
                y, vy = 2.0 * hi - y, -vy
            self.pos[i] = [x, y]
            self.vel[i] = [vx, vy]

        # 3. 船體碰撞
        collided = self._resolve_ship_collision()

        # 4. 捕獲判定
        gains = self._resolve_captures()

        self.tick += 1

        return {
            "gains": gains,
            "collided": collided,
            "done": self.done(),
        }

    def _resolve_ship_collision(self):
        r = self.config["ship_radius"]
        dx = self.pos[1][0] - self.pos[0][0]
        dy = self.pos[1][1] - self.pos[0][1]
        dist = math.hypot(dx, dy)

        if dist > 2.0 * r or dist < EPS:
            return False

        if self.swap_full_velocity:
            self.vel[0], self.vel[1] = self.vel[1], self.vel[0]
        else:
            # 只交換法線分量，切線保留（等質量彈性碰撞的正確版本）
            nx, ny = dx / dist, dy / dist
            v0n = self.vel[0][0] * nx + self.vel[0][1] * ny
            v1n = self.vel[1][0] * nx + self.vel[1][1] * ny
            d = v1n - v0n
            self.vel[0][0] += d * nx
            self.vel[0][1] += d * ny
            self.vel[1][0] -= d * nx
            self.vel[1][1] -= d * ny

        # 推開，避免下一 tick 還黏在一起重複觸發
        overlap = 2.0 * r - dist
        if overlap > 0:
            push = overlap / 2.0 + EPS
            nx, ny = dx / dist, dy / dist
            self.pos[0][0] -= push * nx
            self.pos[0][1] -= push * ny
            self.pos[1][0] += push * nx
            self.pos[1][1] += push * ny

        return True

    def _resolve_captures(self):
        ship_radius = self.config["ship_radius"]
        gains = [0, 0]
        remaining = []

        for chest in self.chests:
            cx, cy = chest["center"]
            cap_r = ship_radius + chest["radius"]

            hit = []
            for i in range(2):
                dx = cx - self.pos[i][0]
                dy = cy - self.pos[i][1]
                if dx * dx + dy * dy <= cap_r * cap_r:
                    hit.append(i)

            if not hit:
                remaining.append(chest)
                continue

            if len(hit) == 2:
                winner = 0 if self.tie_goes_to_player0 else 1
            else:
                winner = hit[0]

            gains[winner] += 1
            self.scores[winner] += 1
            self.events.append({
                "tick": self.tick,
                "player": winner,
                "chest": list(chest["center"]),
                "contested": len(hit) == 2,
            })

        self.chests = remaining
        return gains

    def done(self):
        return len(self.chests) == 0 or self.tick >= MAX_TICKS


# --------------------------------------------------------------------------
# 對局
# --------------------------------------------------------------------------

def run_match(scenario, agent0, agent1, swap_spawns=False, trace=False):
    """跑一整局。agent 需要有 act(state) -> [ax, ay]。"""
    env = ArenaEnv(scenario)
    s0, s1 = env.reset(swap_spawns=swap_spawns)

    for a, pid in ((agent0, 0), (agent1, 1)):
        if hasattr(a, "initialize"):
            a.initialize(pid)

    path = [[], []] if trace else None
    first_capture = [None, None]

    while not env.done():
        a0 = agent0.act(s0)
        a1 = agent1.act(s1)

        if trace:
            path[0].append(tuple(env.pos[0]))
            path[1].append(tuple(env.pos[1]))

        info = env.step(a0, a1)

        for i in range(2):
            if info["gains"][i] and first_capture[i] is None:
                first_capture[i] = env.tick

        s0, s1 = env.state_for(0), env.state_for(1)

    return {
        "scores": list(env.scores),
        "ticks": env.tick,
        "chests_left": len(env.chests),
        "first_capture": first_capture,
        "events": env.events,
        "path": path,
    }


# --------------------------------------------------------------------------
# 檢查用的簡單 agent
# --------------------------------------------------------------------------

class GreedyAgent:
    """永遠全力衝向最近的寶箱。當下限對照組用。"""

    def initialize(self, player_id):
        self.player_id = player_id

    def act(self, state):
        if not state["chests"]:
            return [0.0, 0.0]
        x, y = state["position"]
        best = min(
            state["chests"],
            key=lambda c: (c["center"][0] - x) ** 2 + (c["center"][1] - y) ** 2,
        )
        dx = best["center"][0] - x
        dy = best["center"][1] - y
        d = math.hypot(dx, dy)
        if d < EPS:
            return [0.0, 0.0]
        a = state["config"]["max_accel"]
        return [a * dx / d, a * dy / d]


class IdleAgent:
    def initialize(self, player_id):
        self.player_id = player_id

    def act(self, state):
        return [0.0, 0.0]
