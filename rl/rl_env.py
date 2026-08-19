"""RL 用的環境包裝。

在 env.ArenaEnv 外面加三件事：
  1. 狀態編碼（固定長度向量）
  2. Frame skip —— 每 N 個 tick 才決策一次
  3. 獎勵

## 為什麼要 frame skip

dt = 0.01、max_speed = 20，所以每 tick 最多移動 0.2 單位。
地圖 L = 100，跨越 30 單位要 150 tick。

γ = 0.99 的有效視野約 1/(1-γ) = 100 步：
    150 tick 後的獎勵 → 0.99^150 ≈ 0.22
    300 tick 後       → 0.99^300 ≈ 0.049

也就是說稍遠的寶箱在 agent 眼裡幾乎不存在，
但 strategy.py 的 beam search 規劃四顆寶箱之後的事。
不做 frame skip 的話，這個比較從一開始就不公平 ——
不是 RL 學不會長期規劃，是沒給它看長期的能力。

FRAME_SKIP = 10 之後：150 tick = 15 個決策步，0.99^15 ≈ 0.86。

副作用：每個決策的時間預算從 6 毫秒變成 60 毫秒。

## 狀態裡為什麼一定要有重力井

推力逃逸半徑 = G / max_accel = 10，剛好等於井半徑。
井內任何位置，推力都打不過重力 —— 只能靠動能衝出去。

所以「低速進入井」是不可逆的災難，而且要在進入前幾十個 tick 就避開。
狀態沒有井的資訊的話，agent 只會看到自己偶爾突然死掉，學不到原因。
"""

import math
import env

FRAME_SKIP = 10
K_CHESTS = 8
K_WELLS = 3

# 獎勵
R_CAPTURE = 1.0
R_OPPONENT_CAPTURE = -1.0
R_STEP = -0.002
R_WELL_DANGER = -0.5

# Potential-based shaping（Ng et al. 1999）
#   F(s,s') = gamma * Phi(s') - Phi(s),  Phi(s) = -SHAPING_SCALE * d_最近寶箱 / L
# 這個形式可證明「不改變最優策略」——它只改變學習速度，不改變學到什麼。
# 一般的 bonus（例如「靠近就給分」）沒有這個保證，會把 agent 導向貪心。
SHAPING_SCALE = 1.0
SHAPING_GAMMA = 0.99

# 動作：8 方向 × 2 強度 + 不動 = 17
N_ACTIONS = 17


def build_action_table(max_accel):
    table = [(0.0, 0.0)]
    for k in range(8):
        ang = 2.0 * math.pi * k / 8.0
        for strength in (1.0, 0.5):
            table.append((max_accel * strength * math.cos(ang),
                          max_accel * strength * math.sin(ang)))
    return table


class ArenaRLEnv:

    def __init__(self, scenario, opponent, frame_skip=FRAME_SKIP,
                 k_chests=K_CHESTS, k_wells=K_WELLS, max_steps=None,
                 shaping=True):
        self.scenario = scenario
        self.opponent = opponent
        self.frame_skip = frame_skip
        self.max_steps = max_steps
        self.shaping = shaping
        self.k_chests = k_chests
        self.k_wells = k_wells
        self.actions = build_action_table(scenario["config"]["max_accel"])
        self.core = env.ArenaEnv(scenario)

    # ---------------- 介面 ----------------

    @property
    def obs_dim(self):
        # 自己 6 + 寶箱 K*4 + 井 K*4 + 對手 5 + 全域 3
        return 6 + self.k_chests * 4 + self.k_wells * 4 + 5 + 3

    @property
    def n_actions(self):
        return len(self.actions)

    def reset(self, swap_spawns=False):
        self.core.reset(swap_spawns=swap_spawns)
        if hasattr(self.opponent, "initialize"):
            self.opponent.initialize(1)
        self.steps = 0
        self.was_in_danger = False
        self.danger_entries = 0
        self.prev_potential = self._potential()
        return self.encode()

    def step(self, action_idx):
        ax, ay = self.actions[action_idx]

        gained = 0
        lost = 0
        entered_danger = False

        for _ in range(self.frame_skip):
            if self.core.done():
                break
            opp_action = self.opponent.act(self.core.state_for(1))
            info = self.core.step([ax, ay], opp_action)
            gained += info["gains"][0]
            lost += info["gains"][1]

            # 只在「踏進去」的那一刻扣，不是每 tick 都扣。
            # 每 tick 扣的話，掉進井出不來會累積出巨大的負值，
            # 完全蓋過寶箱的 ±1 訊號 —— 實測隨機策略 1000 步扣了 418 分。
            now = self._in_danger()
            if now and not self.was_in_danger:
                entered_danger = True
                self.danger_entries += 1
            self.was_in_danger = now

        self.steps += 1

        reward = (R_CAPTURE * gained
                  + R_OPPONENT_CAPTURE * lost
                  + R_STEP
                  + (R_WELL_DANGER if entered_danger else 0.0))

        # shaping 項單獨算，方便評估時關掉看真實表現
        phi = self._potential()
        shaping = SHAPING_GAMMA * phi - self.prev_potential
        self.prev_potential = phi
        reward += shaping

        done = self.core.done()
        if self.max_steps is not None and self.steps >= self.max_steps:
            done = True
        return self.encode(), reward, done, {
            "gained": gained, "lost": lost,
            "entered_danger": entered_danger,
            "danger_entries": self.danger_entries,
            "in_danger": self.was_in_danger,
            "shaping": shaping,
            "scores": list(self.core.scores),
        }

    # ---------------- 狀態編碼 ----------------

    def _potential(self):
        """Phi(s) = -scale * (到最近寶箱的距離 / L)

        寶箱吃完時定義為 0，避免終局跳變。
        """
        if not self.shaping or not self.core.chests:
            return 0.0
        x, y = self.core.pos[0]
        d = min(math.hypot(c["center"][0] - x, c["center"][1] - y)
                for c in self.core.chests)
        return -SHAPING_SCALE * d / self.core.config["L"]

    def _in_danger(self):
        """低速待在井內 —— 逃不出去的那種狀態。"""
        x, y = self.core.pos[0]
        vx, vy = self.core.vel[0]
        speed = math.hypot(vx, vy)
        for o in self.core.obstacles:
            cx, cy = o["center"]
            r = math.hypot(cx - x, cy - y)
            if r <= o["radius"]:
                # 逃出需要的動能：G * ln(R / r)
                need = env.GRAVITY_CONSTANT * math.log(
                    max(o["radius"], 1e-9) / max(r, 1e-9))
                if 0.5 * speed * speed < need:
                    return True
        return False

    def encode(self):
        cfg = self.core.config
        L = cfg["L"]
        vmax = cfg["max_speed"]
        x, y = self.core.pos[0]
        vx, vy = self.core.vel[0]

        out = []

        # --- 自己 ---
        out += [x / L, y / L, vx / vmax, vy / vmax]
        speed = math.hypot(vx, vy)
        out += [speed / vmax]
        # 到最近牆的距離（正規化）
        out += [min(x, L - x, y, L - y) / (L / 2)]

        # --- 最近 K 顆寶箱：相對極座標 ---
        chests = sorted(
            self.core.chests,
            key=lambda c: (c["center"][0] - x) ** 2 + (c["center"][1] - y) ** 2,
        )[: self.k_chests]
        for c in chests:
            dx = c["center"][0] - x
            dy = c["center"][1] - y
            d = math.hypot(dx, dy)
            ang = math.atan2(dy, dx)
            # 用 sin/cos 而不是角度本身，避免 2π 處跳變
            out += [d / L, math.cos(ang), math.sin(ang), 1.0]
        for _ in range(self.k_chests - len(chests)):
            out += [0.0, 0.0, 0.0, 0.0]      # 最後一維是存在標記

        # --- 最近 K 個重力井 ---
        wells = sorted(
            self.core.obstacles,
            key=lambda o: (o["center"][0] - x) ** 2 + (o["center"][1] - y) ** 2,
        )[: self.k_wells]
        for o in wells:
            dx = o["center"][0] - x
            dy = o["center"][1] - y
            d = math.hypot(dx, dy)
            ang = math.atan2(dy, dx)
            inside = 1.0 if d <= o["radius"] else 0.0
            out += [d / L, math.cos(ang), math.sin(ang), inside]
        for _ in range(self.k_wells - len(wells)):
            out += [0.0, 0.0, 0.0, 0.0]

        # --- 對手 ---
        ox, oy = self.core.pos[1]
        dx, dy = ox - x, oy - y
        d = math.hypot(dx, dy)
        ang = math.atan2(dy, dx)
        out += [d / L, math.cos(ang), math.sin(ang)]
        ovx, ovy = self.core.vel[1]
        out += [ovx / vmax, ovy / vmax]

        # --- 全域 ---
        s0, s1 = self.core.scores
        total = max(1, s0 + s1 + len(self.core.chests))
        out += [
            len(self.core.chests) / total,
            (s0 - s1) / total,
            self.core.tick / env.MAX_TICKS,
        ]

        return out


if __name__ == "__main__":
    import strategy

    sc = env.load_scenario("scenarios/gravity.csv")
    e = ArenaRLEnv(sc, opponent=env.GreedyAgent())
    obs = e.reset()
    print(f"obs_dim = {e.obs_dim}（實際 {len(obs)}）  n_actions = {e.n_actions}")

    # 隨機策略跑一局，確認能跑完
    import random
    total = 0.0
    done = False
    while not done:
        obs, r, done, info = e.step(random.randrange(e.n_actions))
        total += r
    print(f"隨機策略：報酬 {total:.2f}  比分 {info['scores']}  "
          f"決策步數 {e.steps}  掉井次數 {info['danger_entries']}")

    # greedy 當上限對照
    class Greedy:
        def act(self, obs_unused): pass
    g = env.GreedyAgent(); g.initialize(0)
    e2 = ArenaRLEnv(sc, opponent=env.GreedyAgent())
    e2.reset()
    total2 = 0.0; done = False
    while not done:
        st = e2.core.state_for(0)
        ax, ay = g.act(st)
        # 找最接近的離散動作
        best = min(range(e2.n_actions),
                   key=lambda i: (e2.actions[i][0]-ax)**2 + (e2.actions[i][1]-ay)**2)
        _, r, done, info2 = e2.step(best)
        total2 += r
    print(f"greedy  ：報酬 {total2:.2f}  比分 {info2['scores']}  "
          f"決策步數 {e2.steps}  掉井次數 {info2['danger_entries']}")
